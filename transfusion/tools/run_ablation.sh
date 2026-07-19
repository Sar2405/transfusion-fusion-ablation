#!/usr/bin/env bash
# Three-way fusion-philosophy ablation for TransFusion.
# Trains all three arms with identical settings, differing ONLY in fusion_mode.
#
# Usage:
#   bash tools/run_ablation.sh /path/to/nuscenes
#
# Each run writes to work_dirs/ablation/<fusion_mode>/ so nothing is overwritten.

set -e
DATA_ROOT=${1:?"Usage: run_ablation.sh <nuscenes_data_root>"}
CONFIG=configs/nuscenes.yaml
WORK=work_dirs/ablation

for MODE in full cls_only dual_stream; do
  echo "=============================================="
  echo "  Training fusion_mode = ${MODE}"
  echo "=============================================="
  python tools/train.py \
      --config ${CONFIG} \
      --data-root "${DATA_ROOT}" \
      --work-dir ${WORK} \
      --fusion-mode ${MODE}
done

echo "All three arms trained. Checkpoints in ${WORK}/{full,cls_only,dual_stream}/"
