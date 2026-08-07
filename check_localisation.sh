#!/usr/bin/env bash
# Localisation check across all arms for a given epoch.
#
# Companion to check_collapse.sh. Spread answers "has the model degenerated?";
# this answers "are the predictions actually ON objects?" — the positive test
# that spread cannot give you.
#
#   bash check_localisation.sh                 # epoch_003.pth, seed 42
#   bash check_localisation.sh epoch_009.pth
#   bash check_localisation.sh epoch_009.pth 43
#   SAMPLES=20 THRESH=0.5 bash check_localisation.sh epoch_009.pth
set -u
CKPT=${1:-epoch_003.pth}
SEED=${2:-42}
SAMPLES=${SAMPLES:-4}
THRESH=${THRESH:-0.5}
ROOT=/fast_storage/sav8752
WD=$ROOT/transfusion/work_dirs/ablation_seed${SEED}

cd "$ROOT" || exit 1
export PYTHONPATH=.

echo "────────────────────────────────────────────────────────────────────────"
echo " LOCALISATION CHECK  —  $(date '+%Y-%m-%d %H:%M:%S')"
echo " checkpoint: $CKPT   seed: $SEED   samples: $SAMPLES   score>=$THRESH"
echo "────────────────────────────────────────────────────────────────────────"
printf "%-14s %8s %9s %9s %9s %8s  %s\n" \
       "MODE" "MEDIAN" "0-20m" "20-30m" "30-50m" "<2m" "VERDICT"
printf "%-14s %8s %9s %9s %9s %8s  %s\n" \
       "----" "------" "-----" "------" "------" "---" "-------"

for arm in full cls_only dual_stream; do
    C="$WD/$arm/$CKPT"
    if [ ! -f "$C" ]; then
        printf "%-14s %8s\n" "$arm" "(no checkpoint)"
        continue
    fi
    python check_localisation.py --checkpoint "$C" --mode "$arm" \
        --samples "$SAMPLES" --score-threshold "$THRESH" --table 2>/dev/null \
        || printf "%-14s %8s\n" "$arm" "(failed)"
done

echo "────────────────────────────────────────────────────────────────────────"
echo " median distance from confident predictions to nearest SAME-CLASS GT"
echo ""
echo " GOOD       < 2 m   — inside the nuScenes 2 m threshold, mAP forming"
echo " MARGINAL   2-4 m   — only the loosest threshold matches"
echo " COARSE     4-10 m  — right area, wrong position"
echo " DISPLACED  > 10 m  — not near GT at all"
echo ""
echo " Far range (30-50 m) lags because of single-sweep LiDAR — few returns"
echo " per distant object. This is what multisweep would improve."
echo "────────────────────────────────────────────────────────────────────────"
