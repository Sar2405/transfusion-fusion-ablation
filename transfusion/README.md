# TransFusion — Controlled Study of LiDAR-Camera Fusion Strategies

PyTorch implementation of TransFusion (Bai et al., CVPR 2022) for nuScenes,
extended with a **three-way fusion-philosophy ablation**: the fusion strategy is
a single config switch while backbone, data pipeline, and training recipe stay
identical across arms.

| `--fusion-mode` | Camera → cls | Camera → box regression | Philosophy |
|---|---|---|---|
| `full` (default) | yes | yes | TransFusion |
| `cls_only` | yes | **no** (LiDAR-only) | DAL |
| `dual_stream` | yes | via parallel LiDAR stream | DeepInteraction (+~1.05M params — report) |

## Layout
```
transfusion/
├── configs/nuscenes.yaml        hyperparameters · fusion_mode · data version
├── models/                      transfusion.py · backbones.py · head.py · loss.py
├── data/nuscenes_dataset.py     loader · voxeliser · projections · partial-data filter
├── utils/common.py              encodings · NMS · box decode
└── tools/
    ├── train.py                 AMP/DDP/OneCycleLR · --fusion-mode · per-mode work dirs
    ├── test_pipeline.py         dataset→forward→loss→backward per mode (run this first)
    ├── train_job.sbatch         one Slurm training job (MIG gres, qos/account triple)
    └── launch_ablation.sh       submits 3 modes × 2 seeds = 6 jobs
```

## Key properties
- **BEV grid is derived from geometry** (pc_range / voxel_size / out_size_factor
  → 64×64). Do not pass `bev_h`/`bev_w`; contradicting values raise at construction.
- **Partial-data tolerant:** samples with missing sensor files are dropped at
  init with a logged count (complete datasets drop 0).
- ResNet-50 loads ImageNet weights via a verified name remap — the log must say
  `Missing: 0, Unexpected: 0`; a large count triggers a loud warning.
- Real-data edge cases handled: NaN velocities zeroed, zero-annotation samples
  safe through matcher and losses. DDP-safe in all three modes
  (`find_unused_parameters=False`).

## Run
```bash
pip install -r transfusion/requirements.txt
# from the folder CONTAINING transfusion/:
PYTHONPATH=. python transfusion/tools/test_pipeline.py --data-root /path/to/nuscenes
PYTHONPATH=. python transfusion/tools/train.py \
    --config transfusion/configs/nuscenes.yaml \
    --data-root /path/to/nuscenes --fusion-mode full
```
Set `data.version` in the config to match the data on disk
(`v1.0-trainval` / `v1.0-mini`). Single-sweep by default; for sweeping set
`point_feat_channels: 5` and extend the loader (expect ghost-arc caveats).

## Cluster (Slurm, MIG)
```bash
mkdir -p out err
bash transfusion/tools/launch_ablation.sh     # 6 jobs on 7g.79gb slices
squeue -u $USER
```
Jobs require the site's `--partition/--qos/--account` triple (set in the sbatch).

## Known limitations
- `hard_voxelise` is pure Python (~50 ms/sample single-sweep): acceptable behind
  ≥8 dataloader workers, replace with a vectorised/compiled voxeliser before
  multi-sweep or if GPU utilisation sags.
- Official NDS/mAP evaluation pipeline not yet included (in progress).
- Box code: `[x_n, y_n, z_n, log_l, log_w, log_h, sin, cos, vx, vy]`,
  positions normalised over pc_range, decoded at inference.
