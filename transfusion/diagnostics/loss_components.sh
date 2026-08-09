#!/usr/bin/env bash
# Extract heat/cls/box loss components per epoch, across arms, from the
# training logs. Averages per-step lines within each epoch (the epoch summary
# line only prints total avg_loss, not the component breakdown).
#
#   bash loss_components.sh 43              # seed 43, all three arms
#   bash loss_components.sh 43 cls_only      # just one arm
#   bash loss_components.sh 42              # seed 42 (old tf_train_<jobid> logs)
#
# Seed 42 was launched before --job-name was used, so those logs are named
# tf_train_<jobid>.err rather than <arm><seed>_<jobid>.err. The job-ID mapping
# below is hardcoded for that reason — update it if new seed-42-style runs
# are launched without --job-name.
set -u
SEED=${1:-43}
ARM=${2:-}
ROOT=/fast_storage/sav8752
cd "$ROOT" || exit 1

declare -A NAMED_PATTERN
NAMED_PATTERN["full"]="full${SEED}"
NAMED_PATTERN["cls_only"]="cls${SEED}"
NAMED_PATTERN["dual_stream"]="ds${SEED}"

# Fallback map for seed 42's un-named jobs. Add entries here if other seeds
# also predate --job-name.
declare -A LEGACY_JOBID
if [ "$SEED" = "42" ]; then
    LEGACY_JOBID["full"]="1808"
    LEGACY_JOBID["cls_only"]="1798"
    LEGACY_JOBID["dual_stream"]="1809"
fi

find_log() {
    local arm=$1
    local f
    # 1) named pattern: <arm><seed>_<jobid>.err
    f=$(ls err/${NAMED_PATTERN[$arm]}_*.err 2>/dev/null | head -1)
    if [ -n "$f" ]; then echo "$f"; return; fi
    # 2) legacy: tf_train_<jobid>.err via hardcoded map
    if [ -n "${LEGACY_JOBID[$arm]:-}" ]; then
        f="err/tf_train_${LEGACY_JOBID[$arm]}.err"
        [ -f "$f" ] && { echo "$f"; return; }
    fi
    echo ""
}

for arm in full cls_only dual_stream; do
    [ -n "$ARM" ] && [ "$arm" != "$ARM" ] && continue

    logfile=$(find_log "$arm")
    if [ -z "$logfile" ]; then
        echo "no log found for $arm seed $SEED"
        echo "  tried: err/${NAMED_PATTERN[$arm]}_*.err${LEGACY_JOBID[$arm]:+ and err/tf_train_${LEGACY_JOBID[$arm]}.err}"
        continue
    fi

    echo "════════════════════════════════════════════════════════"
    echo " $arm   seed $SEED   ($logfile)"
    echo "════════════════════════════════════════════════════════"
    printf "%-6s %8s %8s %8s %8s\n" "EPOCH" "TOTAL" "HEAT" "CLS" "BOX"

    python3 - "$logfile" << 'PY'
import re, sys
from collections import defaultdict

path = sys.argv[1]
pat = re.compile(
    r"Epoch \[(\d+)\] Step.*?loss=([\d.]+)\s+\(heat=([\d.]+)\s+cls=([\d.]+)\s+box=([\d.]+)\)"
)
epoch_lines = defaultdict(list)
with open(path) as f:
    for line in f:
        m = pat.search(line)
        if m:
            ep = int(m.group(1))
            epoch_lines[ep].append(tuple(float(x) for x in m.groups()[1:]))

if not epoch_lines:
    print("  (no matching lines — log format may differ; check with:")
    print(f"   grep 'heat=' {path} | head -1)")
else:
    for ep in sorted(epoch_lines):
        rows = epoch_lines[ep]
        n = len(rows)
        tot  = sum(r[0] for r in rows) / n
        heat = sum(r[1] for r in rows) / n
        cls  = sum(r[2] for r in rows) / n
        box  = sum(r[3] for r in rows) / n
        print(f"{ep:<6d} {tot:8.4f} {heat:8.4f} {cls:8.4f} {box:8.4f}")
PY
    echo
done
