#!/usr/bin/env bash
# Health check for the TransFusion ablation runs — organised, aligned output.
# Only reports on jobs currently known to Slurm (running/pending), so old
# resolved crash logs never clutter the view.
#
# Usage: bash check_training.sh   (run from /fast_storage/sav8752)

set -u
DIVIDER="────────────────────────────────────────────────────────────"

echo "$DIVIDER"
echo " ABLATION STATUS  —  $(date '+%Y-%m-%d %H:%M:%S')"
echo "$DIVIDER"

# ---- Job table -------------------------------------------------------
printf "%-8s %-10s %-6s %-10s %-8s %s\n" "JOBID" "MODE" "SEED" "STATE" "TIME" "GPU"
printf "%-8s %-10s %-6s %-10s %-8s %s\n" "-----" "----" "----" "-----" "----" "---"

while read -r jid state name elapsed node; do
    [ -z "$jid" ] && continue
    mode=$(echo "$name" | sed -E 's/tf_//; s/_s[0-9]+$//')
    seed=$(echo "$name" | grep -oP '_s\K[0-9]+$')
    printf "%-8s %-10s %-6s %-10s %-8s %s\n" "$jid" "$mode" "$seed" "$state" "$elapsed" "$node"
done < <(squeue -u "$USER" -h -o "%i %T %j %M %N")

echo
echo "$DIVIDER"
echo " PROGRESS"
echo "$DIVIDER"

while read -r jid name; do
    [ -z "$jid" ] && continue
    logfile=$(ls err/*_"${jid}".err 2>/dev/null | head -1)
    mode=$(echo "$name" | sed -E 's/tf_//; s/_s[0-9]+$//')
    seed=$(echo "$name" | grep -oP '_s\K[0-9]+$')
    label=$(printf "%-12s seed %-3s" "$mode" "$seed")

    if [ -z "$logfile" ]; then
        printf "  %-22s  waiting to start (queued)\n" "$label"
        continue
    fi

    if tail -20 "$logfile" | grep -qi "Traceback\|CUDA out of memory\|loss=nan"; then
        printf "  %-22s  !! ISSUE — see err/%s\n" "$label" "$(basename "$logfile")"
        tail -3 "$logfile" | sed 's/^/        /'
        continue
    fi

    epoch=$(grep -oP 'Epoch \[\K[0-9]+' "$logfile" | tail -1)
    step_line=$(grep -oP 'Step \[\K[0-9]+/[0-9]+' "$logfile" | tail -1)
    loss=$(grep -oP 'loss=\K[0-9.]+' "$logfile" | tail -1)
    speed=$(grep -oP '\K[0-9.]+(?=s/step)' "$logfile" | tail -1)

    if [ -z "$step_line" ]; then
        printf "  %-22s  starting up (loading data / model)\n" "$label"
    else
        printf "  %-22s  epoch %-3s step %-14s loss=%-7s %ss/step\n" "$label" "$epoch" "$step_line" "$loss" "$speed"
    fi
done < <(squeue -u "$USER" -h -o "%i %j")

echo
echo "$DIVIDER"
echo " CHECKPOINTS"
echo "$DIVIDER"
n_ckpt=$(find transfusion/work_dirs -name "*.pth" 2>/dev/null | wc -l)
echo "  $n_ckpt checkpoint file(s) on disk"
find transfusion/work_dirs -name "latest.pth" 2>/dev/null | while read -r p; do
    rel=$(echo "$p" | sed 's|transfusion/work_dirs/||; s|/latest.pth||')
    mtime=$(stat -c '%y' "$p" 2>/dev/null | cut -d'.' -f1)
    printf "    %-30s  saved %s\n" "$rel" "$mtime"
done

echo "$DIVIDER"
