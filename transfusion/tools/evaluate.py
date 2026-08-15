"""
Official nuScenes detection evaluation for a trained TransFusion checkpoint.

Loads a checkpoint, runs inference over the val split, converts predictions
to the nuScenes submission format, and invokes the official nuscenes-devkit
evaluator to compute NDS / mAP / mATE / mASE / mAOE / mAVE / mAAE.

Additionally produces a per-class and per-range breakdown (0-20/20-30/30-40/
40-50m), since that conditional analysis is more informative than the
aggregate table alone.

Usage (from the folder CONTAINING the `transfusion` package):
    PYTHONPATH=. python transfusion/tools/evaluate.py \
        --config transfusion/configs/nuscenes.yaml \
        --data-root /path/to/nuscenes \
        --checkpoint transfusion/work_dirs/ablation_seed42/full/epoch_019.pth \
        --out-dir eval_results/full_seed42
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml
from pyquaternion import Quaternion
from torch.utils.data import DataLoader

from transfusion import TransFusion
from transfusion.data.nuscenes_dataset import (
    NUSCENES_CLASS_NAMES,
    NuScenesDataset,
    collate_fn,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("evaluate")

# Range bins (metres) for the per-range breakdown, matching the proposal.
RANGE_BINS = [(0, 20), (20, 30), (30, 40), (40, 50)]

# Default attribute per detection class (nuScenes requires SOME attribute
# string per box; without velocity/pose history we fall back to the most
# common attribute for each class rather than leaving it empty, which the
# official evaluator penalises less than a wrong-but-plausible guess).
DEFAULT_ATTRIBUTE = {
    "car": "vehicle.parked",
    "truck": "vehicle.parked",
    "construction_vehicle": "vehicle.parked",
    "bus": "vehicle.parked",
    "trailer": "vehicle.parked",
    "barrier": "",
    "motorcycle": "cycle.without_rider",
    "bicycle": "cycle.without_rider",
    "pedestrian": "pedestrian.standing",
    "traffic_cone": "",
}


def build_model(cfg: dict, checkpoint_path: str, device: torch.device,
                mode_override: str = None) -> TransFusion:
    mc = cfg["model"]
    dc = cfg["data"]

    # Load the checkpoint before constructing the model: the checkpoint
    # records which fusion_mode it was trained with, and that determines the
    # architecture to build. Resolution order:
    #   checkpoint field  >  --mode  >  config default
    # A disagreement between an explicit --mode and the checkpoint is fatal,
    # never silently resolved — evaluating a cls_only checkpoint as a full
    # model produces plausible, entirely wrong numbers.
    ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt_mode = ckpt.get("fusion_mode") if isinstance(ckpt, dict) else None

    if ckpt_mode is not None:
        if mode_override is not None and mode_override != ckpt_mode:
            raise SystemExit(
                f"FATAL: --mode '{mode_override}' contradicts the checkpoint, "
                f"which was trained with fusion_mode '{ckpt_mode}'.\n"
                f"  checkpoint: {checkpoint_path}\n"
                f"Refusing to evaluate — one of the two is wrong.")
        resolved_mode = ckpt_mode
        log.info("fusion_mode '%s' read from checkpoint", resolved_mode)
    else:
        resolved_mode = mode_override or mc.get("fusion_mode", "full")
        log.warning(
            "Checkpoint has no embedded fusion_mode (saved before this was "
            "added). Falling back to '%s' from %s. VERIFY this matches how the "
            "checkpoint was trained — an incorrect mode yields plausible but "
            "wrong metrics.", resolved_mode,
            "--mode" if mode_override else "the config file")
    model = TransFusion(
        bev_in_channels=mc["bev_in_channels"],
        num_cameras=mc["num_cameras"],
        num_classes=mc["num_classes"],
        num_queries=mc["num_queries"],
        d_model=mc["d_model"],
        nhead=mc["nhead"],
        num_lidar_decoder_layers=mc["num_lidar_decoder_layers"],
        num_fusion_decoder_layers=mc["num_fusion_decoder_layers"],
        dropout=mc["dropout"],
        img_size=tuple(dc["img_size"]),
        pc_range=tuple(mc["pc_range"]),
        voxel_size=tuple(mc["voxel_size"]),
        out_size_factor=mc["out_size_factor"],
        point_feat_channels=mc.get("point_feat_channels", 4),
        fusion_mode=resolved_mode,
        use_pillar_net=True,
        pretrained_img=False,  # loading trained weights next; skip ImageNet fetch
    ).to(device)

    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    log.info("Loaded checkpoint %s | missing=%d unexpected=%d",
             checkpoint_path, len(missing), len(unexpected))
    if len(missing) > 5 or len(unexpected) > 5:
        log.warning("Checkpoint load looks suspicious (missing=%d, unexpected=%d) "
                    "— verify the checkpoint matches this model config.",
                    len(missing), len(unexpected))
    model.eval()
    return model


def box_to_nusc_format(box: np.ndarray, lidar2global: np.ndarray,
                       sample_token: str, class_name: str, score: float) -> dict:
    """
    box: decoded metric box in the LIDAR SENSOR frame, ordered
         [x, y, z, l, w, h, sin, cos, vx, vy].

         NOTE on the l/w order: the dataset encodes dimensions as
         (log_l, log_w, log_h) — length first — following the
         CenterPoint/TransFusion convention, and decode_boxes() exponentiates
         those three positionally, so its output preserves (l, w, h).
         (decode_boxes' own docstring mislabels this as (w, l, h); the code is
         correct, the docstring is not.) The nuScenes submission format
         requires size as [width, length, height], so this function emits
         [w, l, h] — swapping back. Getting this wrong does not crash: it
         silently submits every box with length and width transposed, which
         inflates mASE and depresses mAP while still producing plausible-
         looking output.

    The official nuScenes evaluator requires submissions in the GLOBAL frame,
    so this transforms centre position, heading, and velocity through
    lidar2global (= T_ego2global @ T_lidarsensor2ego) before building the
    submission dict. Skipping this step silently produces near-zero, wrong
    metrics with no error raised, since the evaluator has no way to detect
    "boxes are in the wrong frame" — it just sees points far from any GT.
    """
    x, y, z, l, w, h, s, c, vx, vy = box
    yaw_lidar = math.atan2(s, c)

    R = lidar2global[:3, :3]
    t = lidar2global[:3, 3]

    # Centre: lidar-frame point -> global-frame point
    center_lidar = np.array([x, y, z], dtype=np.float64)
    center_global = R @ center_lidar + t

    # Heading: rotate the lidar-frame yaw direction vector into global frame,
    # then re-extract yaw. This correctly accounts for any yaw offset between
    # the LiDAR sensor mount and the vehicle's global heading.
    heading_vec_lidar = np.array([math.cos(yaw_lidar), math.sin(yaw_lidar), 0.0])
    heading_vec_global = R @ heading_vec_lidar
    yaw_global = math.atan2(heading_vec_global[1], heading_vec_global[0])
    quat = Quaternion(axis=[0, 0, 1], radians=yaw_global)

    # Velocity is a direction/magnitude, not a point — rotate only, don't
    # translate.
    vel_lidar = np.array([vx, vy, 0.0], dtype=np.float64)
    vel_global = R @ vel_lidar

    return {
        "sample_token": sample_token,
        "translation": [float(center_global[0]), float(center_global[1]), float(center_global[2])],
        "size": [float(w), float(l), float(h)],
        "rotation": [float(quat.w), float(quat.x), float(quat.y), float(quat.z)],
        "velocity": [float(vel_global[0]), float(vel_global[1])],
        "detection_name": class_name,
        "detection_score": float(score),
        "attribute_name": DEFAULT_ATTRIBUTE.get(class_name, ""),
    }


def run_inference(model: TransFusion, loader: DataLoader, device: torch.device,
                  score_threshold: float, nms_iou_threshold: float) -> dict:
    """Runs inference over the whole loader, returns nuScenes submission dict."""
    results: Dict[str, list] = defaultdict(list)
    n_samples = 0

    for i, batch in enumerate(loader):
        preds = model.predict(
            camera_imgs=batch["camera_imgs"].to(device),
            voxels=batch["voxels"].to(device),
            num_points=batch["num_points"].to(device),
            coords=batch["coords"].to(device),
            lidar2img=batch["lidar2img"].to(device),
            score_threshold=score_threshold,
            nms_iou_threshold=nms_iou_threshold,
        )
        tokens = batch["sample_token"]
        lidar2global = batch["lidar2global"].cpu().numpy()  # (B, 4, 4)

        for b, pred in enumerate(preds):
            token = tokens[b]
            # IMPORTANT: touch this key even if there are zero detections.
            # The official nuScenes evaluator requires every validation
            # sample_token to have an entry (even an empty list) — a sample
            # with no detections above threshold would otherwise be silently
            # absent from `results`, and DetectionEval raises an assertion
            # error ("samples in split doesn't match samples in predictions")
            # rather than treating a missing key as "no detections". This is
            # especially likely to bite on an early/undertrained checkpoint.
            results[token]  # noqa: B018 - intentional defaultdict touch

            scores = pred["scores"].cpu().numpy()
            labels = pred["labels"].cpu().numpy()
            boxes  = pred["boxes"].cpu().numpy()
            l2g    = lidar2global[b]
            for k in range(len(scores)):
                cls_name = NUSCENES_CLASS_NAMES[int(labels[k])]
                results[token].append(
                    box_to_nusc_format(boxes[k], l2g, token, cls_name, scores[k])
                )
            n_samples += 1

        if (i + 1) % 200 == 0:
            log.info("Inference: %d batches / %d samples processed", i + 1, n_samples)

    log.info("Inference complete: %d samples, %d total detections",
             n_samples, sum(len(v) for v in results.values()))
    return {
        "meta": {
            "use_camera": True,
            "use_lidar": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        },
        "results": dict(results),
    }


def run_official_eval(nusc, result_path: str, output_dir: str,
                      eval_set: str = "val") -> dict:
    """
    Invokes the official nuscenes-devkit DetectionEval.
    Returns the metrics summary dict (NDS, mAP, per-class AP, TP errors).
    """
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.evaluate import DetectionEval

    cfg = config_factory("detection_cvpr_2019")
    nusc_eval = DetectionEval(
        nusc, config=cfg, result_path=result_path,
        eval_set=eval_set, output_dir=output_dir, verbose=True,
    )
    metrics, _ = nusc_eval.evaluate()
    return metrics.serialize()


def per_range_breakdown(nusc, result_path: str, gt_boxes_by_token: dict,
                        output_dir: str, score_threshold: float = 0.3) -> dict:
    """
    Custom per-class, per-range-bin breakdown, since the official evaluator
    reports aggregate and per-class AP but not range-conditioned numbers.

    Method: predictions above `score_threshold` are matched greedily to GT in
    descending score order, within the same class and range bin, using a 2 m
    centre-distance threshold (nuScenes convention). Reports TP/FP/FN with
    precision, recall and F1 per (class, range bin).

    Two deliberate choices worth stating in the write-up:

    * This is an operating-point analysis at a fixed confidence threshold, NOT
      average precision. AP integrates over all thresholds; this reports what
      the detector actually does at one usable operating point, which is what
      makes per-range recall interpretable. Report it alongside the official
      per-class AP, not as a substitute for it.
    * Matching is score-ordered and greedy so that a confident correct
      detection claims a GT before a marginal one can. Unordered matching
      lets low-confidence predictions absorb GT and inflates recall.

    IMPORTANT: submission and GT translations are in the GLOBAL frame, whose
    origin is the map origin — often kilometres away. Range is therefore
    measured from the EGO VEHICLE position for each sample. Using global
    coordinates directly yields ranges in the thousands of metres, every box
    falls outside the 0-50 m bins, and the breakdown silently comes out empty.

    Returns {class_name: {range_bin: {tp, fp, fn, precision, recall, f1}}}
    """
    with open(result_path) as f:
        submission = json.load(f)["results"]

    breakdown = defaultdict(lambda: defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0}))
    MATCH_DIST_M = 2.0  # nuScenes-style centre-distance match threshold

    def _bin_for(dx: float, dy: float):
        """Range bin for an offset already relative to the ego vehicle."""
        r = math.hypot(dx, dy)
        for lo, hi in RANGE_BINS:
            if lo <= r < hi:
                return (lo, hi)
        return None

    for token, gt_list in gt_boxes_by_token.items():
        # Ego position in the global frame for this sample.
        sample = nusc.get("sample", token)
        lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ego_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
        ex, ey = ego_pose["translation"][0], ego_pose["translation"][1]

        # Filter by confidence, then match in descending score order so a
        # confident detection claims its GT before a marginal one can.
        preds = [p for p in submission.get(token, [])
                 if p.get("detection_score", 0.0) >= score_threshold]
        preds.sort(key=lambda p: p.get("detection_score", 0.0), reverse=True)

        # Bucket GT by (class, range bin), range measured from the ego vehicle.
        gt_by_bin: Dict[tuple, list] = defaultdict(list)
        for gt in gt_list:
            b = _bin_for(gt["translation"][0] - ex, gt["translation"][1] - ey)
            if b is not None:
                gt_by_bin[(gt["detection_name"], b)].append(gt)

        matched_gt = set()
        for p in preds:
            bin_found = _bin_for(p["translation"][0] - ex, p["translation"][1] - ey)
            if bin_found is None:
                continue  # outside the analysed range window
            key = (p["detection_name"], bin_found)
            bin_label = f"{bin_found[0]}-{bin_found[1]}m"
            candidates = gt_by_bin.get(key, [])
            best_dist, best_idx = 1e9, -1
            for gi, gt in enumerate(candidates):
                if (token, key, gi) in matched_gt:
                    continue
                d = math.hypot(p["translation"][0] - gt["translation"][0],
                               p["translation"][1] - gt["translation"][1])
                if d < best_dist:
                    best_dist, best_idx = d, gi
            if best_idx >= 0 and best_dist < MATCH_DIST_M:
                matched_gt.add((token, key, best_idx))
                breakdown[p["detection_name"]][bin_label]["tp"] += 1
            else:
                breakdown[p["detection_name"]][bin_label]["fp"] += 1

        for key, gts in gt_by_bin.items():
            cls_name, (lo, hi) = key
            n_matched = sum(1 for gi in range(len(gts)) if (token, key, gi) in matched_gt)
            breakdown[cls_name][f"{lo}-{hi}m"]["fn"] += len(gts) - n_matched

    # Precision / recall / F1 per cell.
    out = {"_meta": {"score_threshold": score_threshold,
                     "match_distance_m": MATCH_DIST_M,
                     "note": "operating-point analysis, not average precision"}}
    for cls_name, bins in sorted(breakdown.items()):
        out[cls_name] = {}
        for bin_label in [f"{lo}-{hi}m" for lo, hi in RANGE_BINS]:
            if bin_label not in bins:
                continue
            c = bins[bin_label]
            tp, fp, fn = c["tp"], c["fp"], c["fn"]
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall    = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            out[cls_name][bin_label] = {
                "tp": tp, "fp": fp, "fn": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }

    out_path = Path(output_dir) / "per_range_breakdown.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    log.info("Per-range breakdown written to %s (score_threshold=%.2f)",
             out_path, score_threshold)

    # Compact console summary: recall by range, averaged over classes.
    log.info("Per-range recall summary (avg over classes, score>=%.2f):", score_threshold)
    for lo, hi in RANGE_BINS:
        label = f"{lo}-{hi}m"
        vals = [out[c][label]["recall"] for c in out
                if c != "_meta" and label in out[c]]
        if vals:
            log.info("  %-10s recall=%.3f  (%d classes)",
                     label, sum(vals) / len(vals), len(vals))
    return out


def collect_gt_for_breakdown(nusc, sample_tokens: List[str]) -> dict:
    """Pulls GT boxes (in nuScenes submission-like shape) per sample token,
    for use only by the custom per-range breakdown above."""
    from nuscenes.utils.geometry_utils import transform_matrix
    from nuscenes.eval.detection.utils import category_to_detection_name

    gt_by_token = defaultdict(list)
    for token in sample_tokens:
        sample = nusc.get("sample", token)
        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            det_name = category_to_detection_name(ann["category_name"])
            if det_name is None:
                continue
            gt_by_token[token].append({
                "translation": ann["translation"],
                "detection_name": det_name,
            })
    return gt_by_token


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mode", default=None,
                    choices=["full", "cls_only", "dual_stream", "interact"],
                    help="fusion_mode to build the model with; overrides the "
                         "config value. Must match how the checkpoint was "
                         "TRAINED, not left at the config default.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--score-threshold", type=float, default=0.1,
                    help="Confidence threshold applied during inference. Keep "
                         "low (0.05-0.1): AP integrates over the full "
                         "precision-recall curve, so discarding low-confidence "
                         "detections early truncates the curve and lowers mAP.")
    ap.add_argument("--breakdown-score-threshold", type=float, default=0.3,
                    help="Separate, higher threshold for the custom per-range "
                         "precision/recall breakdown, which reports a single "
                         "operating point rather than an averaged curve.")
    ap.add_argument("--nms-iou-threshold", type=float, default=0.2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip-range-breakdown", action="store_true",
                    help="Skip the custom per-range analysis (faster, official "
                         "metrics only).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    log.info("Building val dataset...")
    val_ds = NuScenesDataset(
        data_root=args.data_root, version=args.version,
        split="val", augment=False,
    )
    loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )
    log.info("Val set: %d samples", len(val_ds))

    log.info("Building model and loading checkpoint...")
    if args.mode:
        log.info("fusion_mode overridden via --mode: %s", args.mode)
    model = build_model(cfg, args.checkpoint, device, mode_override=args.mode)

    log.info("Running inference over val split...")
    submission = run_inference(
        model, loader, device,
        score_threshold=args.score_threshold,
        nms_iou_threshold=args.nms_iou_threshold,
    )

    result_path = out_dir / "results_nusc.json"
    with open(result_path, "w") as f:
        json.dump(submission, f)
    log.info("Submission file written to %s", result_path)

    log.info("Running official nuScenes evaluation (NDS/mAP/TP-errors)...")
    metrics = run_official_eval(val_ds.nusc, str(result_path), str(out_dir))

    summary_path = out_dir / "metrics_summary.json"
    with open(summary_path, "w") as f:
        json.dump(metrics, f, indent=2)

    log.info("=" * 60)
    log.info("AGGREGATE RESULTS")
    log.info("=" * 60)
    log.info("NDS:  %.4f", metrics.get("nd_score", float("nan")))
    log.info("mAP:  %.4f", metrics.get("mean_ap", float("nan")))
    tp_errors = metrics.get("tp_errors", {})
    for tp_name, tp_val in tp_errors.items():
        log.info("%s: %.4f", tp_name, tp_val)
    log.info("Full breakdown written to %s", summary_path)

    if not args.skip_range_breakdown:
        log.info("Computing per-class, per-range breakdown...")
        sample_tokens = [s["token"] for s in val_ds.samples]
        gt_by_token = collect_gt_for_breakdown(val_ds.nusc, sample_tokens)
        # Pass the operating point explicitly. Inference already filtered at
        # --score-threshold; the breakdown applies its own (higher) threshold
        # on top, so the reported precision/recall correspond to a single,
        # stated operating point rather than an implicit default.
        per_range_breakdown(val_ds.nusc, str(result_path), gt_by_token,
                            str(out_dir),
                            score_threshold=args.breakdown_score_threshold)

    log.info("Evaluation complete. All outputs in %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
