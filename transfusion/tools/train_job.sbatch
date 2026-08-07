#!/bin/bash
# One TransFusion training run on the `advance` QOS (3g.39gb MIG slice).
# Mode and seed come from --export on submission:
#   sbatch --export=ALL,MODE=full,SEED=42 transfusion/tools/train_job.sbatch
#
# NOTE on --gres naming: this cluster registers the 3g.39gb MIG profile under
# its FULL name `nvidia_a100_80gb_pcie_3g.39gb`, not the short `3g.39gb`.
#
# AUTO-RESUME: if a checkpoint already exists for this mode/seed's work dir
# (from a previous crash, preemption, or interrupted run), automatically
# resume from it instead of restarting from scratch. This is safe even on a
# genuinely fresh run, since no checkpoint exists yet and the check no-ops.
#SBATCH --job-name=tf_train
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_3g.39gb:1
#SBATCH --mem=64G
#SBATCH --time=120:00:00
#SBATCH --partition=advance
#SBATCH --qos=advance
#SBATCH --account=advance

MODE=${MODE:-full}
SEED=${SEED:-42}
CODE_DIR=${CODE_DIR:-/fast_storage/sav8752/transfusion}
DATA_ROOT=${DATA_ROOT:-/data/aimotion/nuScenes-lidarseg/v1.0-trainval}
VENV=${VENV:-/fast_storage/sav8752/tf}

source "$VENV/bin/activate"
echo "Python: $(which python)"
echo "Host: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) | mode=$MODE seed=$SEED"

cd "$CODE_DIR/.."
export PYTHONPATH="$PYTHONPATH:$(pwd)"

WORK_DIR="$CODE_DIR/work_dirs/ablation_seed${SEED}"
CKPT="$WORK_DIR/$MODE/latest.pth"

RESUME_ARGS=""
if [ -f "$CKPT" ]; then
    echo "Found existing checkpoint at $CKPT — auto-resuming."
    RESUME_ARGS="--resume $CKPT"
else
    echo "No existing checkpoint found — starting fresh."
fi

srun python "$CODE_DIR/tools/train.py" \
    --config "$CODE_DIR/configs/nuscenes.yaml" \
    --data-root "$DATA_ROOT" \
    --work-dir "$WORK_DIR" \
    --fusion-mode "$MODE" \
    --seed "$SEED" \
    --batch-size 2 \
    $RESUME_ARGS
