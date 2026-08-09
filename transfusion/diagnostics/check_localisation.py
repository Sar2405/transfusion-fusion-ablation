#!/usr/bin/env python3
"""
Localisation check — how far are confident predictions from the nearest GT?

This is the diagnostic that mAP hides. mAP can be ~0 for two very different
reasons, and they need different fixes:

  (a) predictions land in roughly the right PLACE but not close enough to match
      (nuScenes matches at 0.5/1/2/4 m BEV centre distance)
  (b) predictions land somewhere unrelated entirely

Usage:
    PYTHONPATH=. python check_localisation.py \
        --checkpoint transfusion/work_dirs/ablation_seed42/cls_only/epoch_003.pth \
        --mode cls_only --samples 50
"""
import argparse, os, sys
import numpy as np
import torch
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mode", required=True,
                    choices=["full", "cls_only", "dual_stream"])
    ap.add_argument("--config", default="transfusion/configs/nuscenes.yaml")
    ap.add_argument("--data-root",
                    default="/data/aimotion/nuScenes-lidarseg/v1.0-trainval")
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--samples", type=int, default=50)
    ap.add_argument("--score-threshold", type=float, default=0.3,
                    help="only CONFIDENT predictions are diagnostic; weak ones "
                         "are expected to be scattered")
    ap.add_argument("--table", action="store_true",
                    help="one compact line of output (for the shell wrapper)")
    args = ap.parse_args()

    from transfusion.data.nuscenes_dataset import NuScenesDataset, collate_fn
    from transfusion.models.transfusion import TransFusion

    cfg = yaml.safe_load(open(args.config))
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    mc_ = cfg["model"]
    ds = NuScenesDataset(
        data_root=args.data_root, version=args.version, split="val",
        augment=False,
        pc_range=tuple(mc_["pc_range"]),
        voxel_size=tuple(mc_["voxel_size"]),
    )
    if not args.table:
        print(f"val samples: {len(ds)}   evaluating {args.samples}")

    # Construct exactly as transfusion/tools/evaluate.py does.
    mc = cfg["model"]
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
        pc_range=tuple(mc["pc_range"]),
        voxel_size=tuple(mc["voxel_size"]),
        out_size_factor=mc["out_size_factor"],
        point_feat_channels=mc.get("point_feat_channels", 4),
        fusion_mode=args.mode,          # overridden per arm
        use_pillar_net=True,
        pretrained_img=False,
    ).to(dev).eval()
    ck = torch.load(args.checkpoint, map_location=dev)
    sd = ck.get("model", ck.get("state_dict", ck))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if not args.table:
        print(f"loaded {os.path.basename(args.checkpoint)}   "
              f"missing={len(missing)} unexpected={len(unexpected)}")
    if missing or unexpected:
        print("  WARNING: key mismatch — architecture may not match the "
              "checkpoint. Numbers below would be meaningless.")
        for k in list(missing)[:5]:
            print(f"    missing:    {k}")
        for k in list(unexpected)[:5]:
            print(f"    unexpected: {k}")

    pr = ds.pc_range
    all_d, all_d_sameclass, n_conf, n_boxes = [], [], 0, 0
    per_range = {"0-20": [], "20-30": [], "30-50": []}

    with torch.no_grad():
        for i in range(min(args.samples, len(ds))):
            batch = collate_fn([ds[i]])
            batch = {k: (v.to(dev) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            out = model(
                camera_imgs=batch["camera_imgs"],
                voxels=batch["voxels"],
                num_points=batch["num_points"],
                coords=batch["coords"],
                lidar2img=batch["lidar2img"],
            )

            boxes = out["pred_boxes"][0].cpu().numpy()      # (Q,10) normalised
            logits = out["pred_logits"][0].cpu().numpy()    # (Q,C)
            scores = 1.0 / (1.0 + np.exp(-logits))
            cls = scores.argmax(1)
            conf = scores.max(1)

            gt = batch["gt_boxes"][0].cpu().numpy()
            gtl = batch["gt_labels"][0].cpu().numpy()
            if len(gt) == 0:
                continue

            def denorm(b):
                return np.stack([
                    b[:, 0] * (pr[3] - pr[0]) + pr[0],
                    b[:, 1] * (pr[4] - pr[1]) + pr[1]], axis=-1)

            p_xy, g_xy = denorm(boxes), denorm(gt)
            keep = conf >= args.score_threshold
            n_conf += int(keep.sum())
            n_boxes += len(gt)

            for j in np.where(keep)[0]:
                d = np.linalg.norm(g_xy - p_xy[j], axis=1)
                all_d.append(d.min())

                same = gtl == cls[j]
                if same.any():
                    dm = d[same].min()
                    all_d_sameclass.append(dm)
                    r = np.linalg.norm(p_xy[j])
                    key = "0-20" if r < 20 else ("20-30" if r < 30 else "30-50")
                    per_range[key].append(dm)

    if not all_d:
        print("\nNo predictions above the score threshold — try a lower "
              "--score-threshold.")
        return

    a, s = np.array(all_d), np.array(all_d_sameclass)

    if args.table:
        med = np.median(s) if len(s) else float("nan")
        near = np.median(per_range["0-20"]) if per_range["0-20"] else float("nan")
        mid  = np.median(per_range["20-30"]) if per_range["20-30"] else float("nan")
        far  = np.median(per_range["30-50"]) if per_range["30-50"] else float("nan")
        w2   = (s < 2.0).mean() if len(s) else 0.0
        verdict = ("GOOD" if med < 2 else "MARGINAL" if med < 4
                   else "COARSE" if med < 10 else "DISPLACED")
        flag = "  <-- key mismatch" if (missing or unexpected) else ""
        print(f"{args.mode:<14} {med:8.2f} {near:9.2f} {mid:9.2f} {far:9.2f} "
              f"{w2:8.1%}  {verdict}{flag}")
        return

    print(f"\nconfident predictions (score>={args.score_threshold}): {n_conf}"
          f"   GT boxes: {n_boxes}")
    print("\ndistance to NEAREST GT (any class), metres")
    print(f"   median {np.median(a):6.2f}   mean {a.mean():6.2f}   "
          f"p10 {np.percentile(a,10):6.2f}   p90 {np.percentile(a,90):6.2f}")
    if len(s):
        print("\ndistance to nearest SAME-CLASS GT, metres  <-- the diagnostic one")
        print(f"   median {np.median(s):6.2f}   mean {s.mean():6.2f}   "
              f"p10 {np.percentile(s,10):6.2f}   p90 {np.percentile(s,90):6.2f}")
        print("\n   fraction within each nuScenes matching threshold:")
        for t in (0.5, 1.0, 2.0, 4.0):
            print(f"     < {t:>3.1f} m : {(s < t).mean():6.1%}")
        print("\n   by ego distance (median, same-class):")
        for k, v in per_range.items():
            if v:
                print(f"     {k:>6} m : {np.median(v):6.2f} m   (n={len(v)})")

    med = np.median(s) if len(s) else np.median(a)
    print("\n" + "=" * 60)
    if med < 2.0:
        print(f"  GOOD — median {med:.2f} m. Inside the 2 m threshold; real")
        print("  matches should be forming and mAP should be non-trivial.")
    elif med < 4.0:
        print(f"  MARGINAL — median {med:.2f} m. Only the loosest (4 m)")
        print("  threshold will match, so mAP stays low but non-zero.")
    elif med < 10.0:
        print(f"  COARSE — median {med:.2f} m. Right area, wrong position.")
    else:
        print(f"  DISPLACED — median {med:.2f} m. Predictions are not near GT at all.")
    print("=" * 60)
    print("\nNote: this is an EARLY checkpoint. Localisation should tighten")
    print("over training — compare the same number across epochs rather than")
    print("judging convergence from one.")


if __name__ == "__main__":
    main()
