#!/usr/bin/env bash
# Query-collapse health check for the fusion ablation.
#
# Usage:
#   bash check_collapse.sh                           # latest.pth, all arms
#   bash check_collapse.sh epoch_003.pth             # specific epoch, all arms
#   bash check_collapse.sh all                       # all epochs, all arms
#   bash check_collapse.sh all seed=42               # all epochs, seed 42 only
#   bash check_collapse.sh all mode=full             # all epochs, mode full only
#   bash check_collapse.sh all seed=42 mode=full     # all epochs, one arm
#   bash check_collapse.sh epoch_003.pth seed=42     # single epoch, seed 42
#   bash check_collapse.sh epoch_003.pth mode=full   # single epoch, mode full
#
# Run from /fast_storage/sav8752 with the venv active.

set -u

# Parse arguments
CKPT_NAME="${1:-latest.pth}"
shift || true

SEED_FILTER=""
MODE_FILTER=""

for arg in "$@"; do
  case "$arg" in
    seed=*)
      SEED_FILTER="${arg#seed=}"
      ;;
    mode=*)
      MODE_FILTER="${arg#mode=}"
      ;;
    *)
      echo "Unknown argument: $arg (use seed=<N> and/or mode=<MODE>)"
      exit 1
      ;;
  esac
done

ROOT="transfusion/work_dirs"
DATA="/data/aimotion/nuScenes-lidarseg/v1.0-trainval"

echo "────────────────────────────────────────────────────────────"
echo " QUERY-COLLAPSE CHECK  —  $(date '+%Y-%m-%d %H:%M:%S')"
if [ "$CKPT_NAME" = "all" ]; then
  echo " checkpoints: ALL EPOCHS"
else
  echo " checkpoint: $CKPT_NAME"
fi
if [ -n "$SEED_FILTER" ]; then
  echo " seed filter: $SEED_FILTER"
fi
if [ -n "$MODE_FILTER" ]; then
  echo " mode filter: $MODE_FILTER"
fi
echo "────────────────────────────────────────────────────────────"

# Python helper that does the actual metric computation.
run_diag() {
  local ckpt_path="$1"
  PYTHONPATH=. python3 - "$ckpt_path" "$DATA" << 'PY_EOF'
import sys, os, warnings, logging
warnings.filterwarnings("ignore")
logging.disable(logging.INFO)

ckpt_path, data_root = sys.argv[1], sys.argv[2]

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

# Infer seed and mode from the path:
# e.g. transfusion/work_dirs/ablation_seed42/full/epoch_003.pth
parts = ckpt_path.replace("\\", "/").split("/")
# expect: ..., ablation_seed<SEED>, <MODE>, <CKPT>
seed_str = parts[-3]
mode = parts[-2]
seed = int(seed_str.replace("ablation_seed", ""))

try:
    m = build_model(cfg, ckpt_path, dev)
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
}

print_header() {
  printf "%-14s %-5s %10s %10s %9s  %s\n" "MODE" "SEED" "X-SPREAD" "Y-SPREAD" "MAXSCORE" "VERDICT"
  printf "%-14s %-5s %10s %10s %9s  %s\n" "----" "----" "--------" "--------" "--------" "-------"
}

# Lists of seeds and modes to iterate over
SEEDS="42 43"
MODES="full cls_only dual_stream"

# Apply filters if provided
if [ -n "$SEED_FILTER" ]; then
  SEEDS="$SEED_FILTER"
fi
if [ -n "$MODE_FILTER" ]; then
  MODES="$MODE_FILTER"
fi

if [ "$CKPT_NAME" != "all" ]; then
  # Single-epoch mode
  for seed in $SEEDS; do
    for mode in $MODES; do
      path="$ROOT/ablation_seed${seed}/${mode}/${CKPT_NAME}"
      if [ ! -f "$path" ]; then
        continue
      fi
      echo "Epoch checkpoint: ${path}"
      print_header
      run_diag "$path"
      echo ""
    done
  done
else
  # All-epochs mode: discover all epoch_*.pth in each arm and run sequentially.
  for seed in $SEEDS; do
    for mode in $MODES; do
      dir="$ROOT/ablation_seed${seed}/${mode}"
      if [ ! -d "$dir" ]; then
        continue
      fi
      # List epoch checkpoints in order.
      ckpts=$(ls "$dir"/epoch_*.pth 2>/dev/null | sort -V)
      if [ -z "$ckpts" ]; then
        continue
      fi

      echo "────────────────────────────────────────────────────────────"
      echo "Arm: ablation_seed${seed} / ${mode}  —  all epochs"
      echo "────────────────────────────────────────────────────────────"

      printf "%-12s %-14s %-5s %10s %10s %9s  %s\n" "EPOCH" "MODE" "SEED" "X-SPREAD" "Y-SPREAD" "MAXSCORE" "VERDICT"
      printf "%-12s %-14s %-5s %10s %10s %9s  %s\n" "-----" "----" "----" "--------" "--------" "--------" "-------"

      for ckpt in $ckpts; do
        epoch_base=$(basename "$ckpt")  # e.g. epoch_003.pth
        # Extract epoch number for nicer printing
        epoch_num=$(echo "$epoch_base" | sed 's/epoch_\([0-9]*\)\.pth/\1/')
        # Run diagnostic and prefix with epoch number
        result=$(run_diag "$ckpt")
        # result is like: "full         42      44.12m     94.03m     0.831  OK"
        printf "%-12s %s\n" "$epoch_num" "$result"
      done

      echo ""
    done
  done
fi

echo "────────────────────────────────────────────────────────────"
echo " OK          spread > 20 m and confident scores"
echo " degrading   spread 5-20 m — re-check next epoch"
echo " COLLAPSED   spread < 5 m — stop the run, lower the LR"
echo "────────────────────────────────────────────────────────────"
