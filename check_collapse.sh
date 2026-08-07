#!/usr/bin/env bash
# Query-collapse health check for the fusion ablation.
#
# Detects the failure mode that wasted the first training attempt: all 200
# object queries converging to a single predicted location, which produces
# ~1 detection per sample and zero mAP. A healthy model spreads its query
# predictions across the detection range (tens of metres) and reaches
# confident scores; a collapsing one shrinks toward 0 m spread with scores
# pinned near the focal-loss init value (~0.05).
#
# Usage:
#   bash check_collapse.sh                # newest checkpoint of every arm
#   bash check_collapse.sh epoch_003.pth  # a specific epoch, all arms
#
# Run from /fast_storage/sav8752 with the venv active.

set -u
CKPT_NAME="${1:-latest.pth}"
ROOT="transfusion/work_dirs"
DATA="/data/aimotion/nuScenes-lidarseg/v1.0-trainval"

echo "────────────────────────────────────────────────────────────"
echo " QUERY-COLLAPSE CHECK  —  $(date '+%Y-%m-%d %H:%M:%S')"
echo " checkpoint: $CKPT_NAME"
echo "────────────────────────────────────────────────────────────"
printf "%-14s %-5s %10s %10s %9s  %s\n" "MODE" "SEED" "X-SPREAD" "Y-SPREAD" "MAXSCORE" "VERDICT"
printf "%-14s %-5s %10s %10s %9s  %s\n" "----" "----" "--------" "--------" "--------" "-------"

PYTHONPATH=. python3 - "$ROOT" "$CKPT_NAME" "$DATA" << 'PY_EOF'
import sys, os, glob, warnings, logging
warnings.filterwarnings("ignore")
logging.disable(logging.INFO)

root, ckpt_name, data_root = sys.argv[1], sys.argv[2], sys.argv[3]

import torch, yaml
from transfusion.data.nuscenes_dataset import NuScenesDataset, collate_fn
from transfusion.tools.evaluate import build_model
from transfusion.utils.common import decode_boxes

cfg = yaml.safe_load(open("transfusion/configs/nuscenes.yaml"))

# One fixed val sample, reused for every arm so the comparison is like-for-like.
ds = NuScenesDataset(data_root=data_root, version="v1.0-trainval",
                     split="val", augment=False)
batch = collate_fn([ds[0]])
dev = torch.device("cpu")

for seed in (42, 43):
    for mode in ("full", "cls_only", "dual_stream"):
        path = os.path.join(root, f"ablation_seed{seed}", mode, ckpt_name)
        if not os.path.exists(path):
            continue
        try:
            cfg["model"]["fusion_mode"] = mode
            m = build_model(cfg, path, dev)
            with torch.no_grad():
                out = m(camera_imgs=batch["camera_imgs"], voxels=batch["voxels"],
                        num_points=batch["num_points"], coords=batch["coords"],
                        lidar2img=batch["lidar2img"])
            d = decode_boxes(out["pred_boxes"][0], cfg["model"]["pc_range"])
            xs = float(d[:, 0].max() - d[:, 0].min())
            ys = float(d[:, 1].max() - d[:, 1].min())
            sc = float(out["pred_logits"][0].sigmoid().max())

            # Thresholds: healthy runs showed ~44 m x / ~94 m y with score 0.83
            # at epoch 0. The collapsed run showed 0.00 m and score 0.05.
            if max(xs, ys) < 5.0:
                verdict = "!! COLLAPSED"
            elif max(xs, ys) < 20.0:
                verdict = "?  degrading"
            elif sc < 0.15:
                verdict = "?  low conf"
            else:
                verdict = "OK"
            print(f"{mode:<14} {seed:<5} {xs:9.2f}m {ys:9.2f}m {sc:9.3f}  {verdict}")
        except Exception as e:  # noqa: BLE001
            print(f"{mode:<14} {seed:<5} {'-':>10} {'-':>10} {'-':>9}  ERROR: {e}")
PY_EOF

echo "────────────────────────────────────────────────────────────"
echo " OK          spread > 20 m and confident scores"
echo " degrading   spread 5-20 m — re-check next epoch"
echo " COLLAPSED   spread < 5 m — stop the run, lower the LR"
echo "────────────────────────────────────────────────────────────"
