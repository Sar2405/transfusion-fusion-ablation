"""
Full pipeline validation on real nuScenes data.

For each fusion mode (full, cls_only, dual_stream):
  dataset -> batch -> model forward -> loss -> backward
and reports shapes, loss values, and gradient health.

Usage (from the folder CONTAINING the `transfusion` package):
    PYTHONPATH=. python transfusion/tools/test_pipeline.py \
        --data-root /fast_storage/sav8752/nuscenes [--device cuda]

Exit code 0 = every mode passed with finite losses and gradients.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

import torch

from transfusion import TransFusion, TransFusionLoss
from transfusion.data.nuscenes_dataset import NuScenesDataset, collate_fn

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("test_pipeline")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=2)
    args = ap.parse_args()

    device = torch.device(args.device)
    log.info("Device: %s", device)
    if device.type == "cuda":
        log.info("GPU: %s", torch.cuda.get_device_name(0))

    # ---- Dataset (partial-data filter runs inside __init__) ----
    t0 = time.time()
    ds = NuScenesDataset(
        data_root=args.data_root, version=args.version,
        split="train", augment=False,
    )
    log.info("Dataset ready in %.1fs — %d usable samples", time.time() - t0, len(ds))

    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, collate_fn=collate_fn,
    )
    batch = next(iter(loader))
    log.info("Batch: voxels %s | coords %s | imgs %s | lidar2img %s | GT boxes %s",
             tuple(batch["voxels"].shape), tuple(batch["coords"].shape),
             tuple(batch["camera_imgs"].shape), tuple(batch["lidar2img"].shape),
             [tuple(b.shape) for b in batch["gt_boxes"]])

    overall_ok = True
    for mode in ("full", "cls_only", "dual_stream"):
        log.info("=" * 60)
        log.info("FUSION MODE: %s", mode)
        try:
            # No bev_h/bev_w passed: new code derives them from geometry,
            # older code falls back to its consistent 64x64 defaults.
            model = TransFusion(
                point_feat_channels=4,
                use_pillar_net=True,
                fusion_mode=mode,
            ).to(device).train()
            n_params = sum(p.numel() for p in model.parameters()) / 1e6
            log.info("Model built: %.1fM params | BEV grid %dx%d",
                     n_params, model.bev_h, model.bev_w)

            out = model(
                camera_imgs=batch["camera_imgs"].to(device),
                voxels=batch["voxels"].to(device),
                num_points=batch["num_points"].to(device),
                coords=batch["coords"].to(device),
                lidar2img=batch["lidar2img"].to(device),
            )
            log.info("Forward OK: logits %s | boxes %s | heatmap %s",
                     tuple(out["pred_logits"].shape),
                     tuple(out["pred_boxes"].shape),
                     tuple(out["heatmap"].shape))

            criterion = TransFusionLoss(num_classes=10).to(device)
            gt_labels = [g.to(device) for g in batch["gt_labels"]]
            gt_boxes  = [g.to(device) for g in batch["gt_boxes"]]
            losses = criterion(out, gt_labels, gt_boxes,
                               model.bev_h, model.bev_w)
            loss_str = "  ".join(f"{k}={float(v):.4f}" for k, v in losses.items())
            log.info("Losses: %s", loss_str)

            total = losses["loss"]
            if not torch.isfinite(total):
                raise RuntimeError(f"NON-FINITE total loss: {float(total)}")

            model.zero_grad(set_to_none=True)
            total.backward()
            grads = [p.grad.abs().max() for p in model.parameters()
                     if p.grad is not None]
            gmax = float(torch.stack(grads).max())
            n_with_grad = len(grads)
            if not torch.isfinite(torch.tensor(gmax)):
                raise RuntimeError(f"NON-FINITE gradient (max={gmax})")
            log.info("Backward OK: %d tensors received grads | max|grad|=%.3e",
                     n_with_grad, gmax)
            log.info("MODE %s: PASS", mode)

        except Exception as e:  # noqa: BLE001
            overall_ok = False
            log.error("MODE %s: FAIL — %s", mode, e, exc_info=True)

        finally:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    log.info("=" * 60)
    log.info("OVERALL: %s", "ALL MODES PASS" if overall_ok else "FAILURES — see above")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
