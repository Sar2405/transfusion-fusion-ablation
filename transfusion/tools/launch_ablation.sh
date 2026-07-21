#!/usr/bin/env bash
# Launch the full 3-mode x 2-seed ablation as six independent Slurm jobs.
# Run from the folder CONTAINING the package dir (e.g. /fast_storage/sav8752):
#   bash Transfusion/tools/launch_ablation.sh
set -e
mkdir -p out err

for SEED in 42 43; do
  for MODE in full cls_only dual_stream; do
    sbatch --job-name="tf_${MODE}_s${SEED}" \
           --export=ALL,MODE=${MODE},SEED=${SEED} \
           Transfusion/tools/train_job.sbatch
  done
done

echo "Submitted 6 jobs. Monitor with: squeue -u $USER"
echo "Logs: out/tf_<mode>_s<seed>_<jobid>.out"
