# TransFusion — LiDAR-Camera Fusion for 3D Object Detection

PyTorch implementation of *TransFusion: Robust LiDAR-Camera Fusion for 3D Object
Detection with Transformers* (Bai et al., CVPR 2022), built for the nuScenes dataset.

## Architecture

```
LiDAR point cloud ─► PillarFeatureNet ─► SECONDBackbone ─► BEV feature map ─┐
                                                                            │
                                                              Heatmap ─► 200 query seeds
                                                                            │
6 camera images ─► ResNet-50 ─► FPN ─► image features ──────────┐          │
                                                                │          ▼
                                                                │   Stage 1: LiDAR-only
                                                                │   transformer decoder
                                                                │          │
                                                                │   predicted 3D centres
                                                                │          ▼
                                                                └─► Stage 2: LiDAR-camera
                                                                    fusion decoder
                                                                            │
                                                                            ▼
                                                          class scores + 3D boxes + velocity
```

## Package layout

| File | Contents |
|------|----------|
| `models/transfusion.py`   | Top-level `TransFusion` model + `predict()` inference |
| `models/backbones.py`     | `PillarFeatureNet`, `SECONDBackbone`, `ResNet50`, `FPN`, `ImageBackbone` |
| `models/head.py`          | `TransFusionHead` — heatmap, stage-1 LiDAR decoder, stage-2 fusion decoder |
| `models/loss.py`          | `TransFusionLoss`, Hungarian matcher, Gaussian heatmap targets |
| `data/nuscenes_dataset.py`| nuScenes loader, voxeliser, projection matrices, augmentation |
| `utils/common.py`         | Positional encodings, NMS, box decoding, helpers |
| `tools/train.py`          | Training loop: DDP, AMP, OneCycleLR, checkpointing |
| `configs/nuscenes.yaml`   | Full configuration |

## Setup

```bash
pip install -r requirements.txt
```

Download nuScenes (v1.0-trainval) from https://www.nuscenes.org and point the
config / CLI at the data root.

## Training

```bash
# Single GPU
python tools/train.py --config configs/nuscenes.yaml --data-root /data/nuscenes

# 8 GPUs (DistributedDataParallel)
torchrun --nproc_per_node=8 tools/train.py \
    --config configs/nuscenes.yaml \
    --data-root /data/nuscenes
```

## Inference

```python
import torch
from transfusion import TransFusion

model = TransFusion(use_pillar_net=True).eval()
detections = model.predict(
    camera_imgs=imgs,        # (B, 6, 3, H, W)
    voxels=voxels,           # (M, P, 4)
    num_points=num_points,   # (M,)
    coords=coords,           # (M, 4)  [batch, z, y, x]
    lidar2img=lidar2img,     # (B, 6, 4, 4)
    score_threshold=0.1,
    nms_iou_threshold=0.2,
)
# detections[b] = {"scores": (K,), "labels": (K,), "boxes": (K, 10)}
```

## Box parameterisation

Each box is 10 values: `[x, y, z, log_l, log_w, log_h, sin_yaw, cos_yaw, vx, vy]`,
with x/y/z normalised to `[0, 1]` over the point-cloud range during training and
decoded back to metres at inference.

## Single-sweep vs multi-sweep

`configs/nuscenes.yaml` defaults to single-sweep. For 10-sweep training:
- set `data.num_sweeps: 10`
- set `model.bev_in_channels: 5` (adds the Δt time channel)
- raise `data.max_voxels_train` to 60000+

See the dataset docstring for the sweep-aggregation details.

## Notes for production use

The pure-Python voxeliser in `nuscenes_dataset.py` is correct but slow. For real
training throughput, replace `hard_voxelise` with the compiled voxeliser from
`mmcv` / `spconv`, or pre-cache voxelised sweeps to disk.
