#!/bin/bash
# One TransFusion training run. Mode and seed come from --export on submission:
#   sbatch --export=ALL,MODE=full,SEED=42 transfusion/tools/train_job.sbatch
# Defaults below are used if not exported.
#SBATCH --job-name=tf_train
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:7g.79gb:1
#SBATCH --mem=100G
#SBATCH --time=96:00:00
#SBATCH --partition=debugging
#SBATCH --qos=debugging
#SBATCH --account=debugging

MODE=${MODE:-full}
SEED=${SEED:-42}
CODE_DIR=${CODE_DIR:-/fast_storage/sav8752/Transfusion}
DATA_ROOT=${DATA_ROOT:-/fast_storage/sav8752/nuscenes}
VENV=${VENV:-$HOME/venvs/tf}

echo "Host: $(hostname) | MIG: $CUDA_VISIBLE_DEVICES | mode=$MODE seed=$SEED"
source "$VENV/bin/activate"
cd "$CODE_DIR/.."          # the folder CONTAINING the package dir

export PYTHONPATH="$PYTHONPATH:$(pwd)"
srun python "$CODE_DIR/tools/train.py" \
    --config "$CODE_DIR/configs/nuscenes.yaml" \
    --data-root "$DATA_ROOT" \
    --work-dir "$CODE_DIR/work_dirs/ablation_seed${SEED}" \
    --fusion-mode "$MODE" \
    --seed "$SEED"
