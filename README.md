# transfusion-fusion-ablation
Controlled ablation of LiDAR-camera fusion strategies (TransFusion, DAL, DeepInteraction) inside one fixed architecture, for a master's thesis on 3D object detection
# Fusion Strategies for LiDAR–Camera 3D Object Detection

A controlled ablation of three LiDAR–camera fusion philosophies in a
transformer-based 3D detector, trained and evaluated on nuScenes.

Master's thesis project — AI Engineering of Autonomous Systems,
Technische Hochschule Ingolstadt.

---

## The question

Published fusion methods disagree on a specific design choice: **should camera
features influence 3D bounding-box regression, or only classification?**

- **TransFusion** [1] fuses image features into the object queries and lets
  both classification and regression read the fused representation.
- **DAL** [2] argues the opposite — box geometry should come from the point
  cloud alone, since monocular depth is ill-posed and letting image features
  into regression invites overfitting.

The papers are hard to compare directly: different backbones, schedules,
augmentation, data pipelines. This project isolates the fusion strategy —
one architecture, one training recipe, one dataset — behind a single
configuration switch that changes **only which feature source feeds which
prediction head**.

---

## The three arms

Classification always reads the fused LiDAR+camera query. Only the box
regression input differs:

```python
if self.fusion_mode == "full":
    cls_src, box_src = fused_query, fused_query          # TransFusion
elif self.fusion_mode == "cls_only":
    cls_src, box_src = fused_query, lidar_query           # DAL rule B
elif self.fusion_mode == "dual_stream":
    for layer in self.lidar_stream:
        lidar_query = layer(lidar_query, bev_flat, query_pos_emb, bev_pos)
    cls_src, box_src = fused_query, lidar_query            # DeepInteraction-style
```

`lidar_query` is snapshotted before the fusion decoder runs, so in `cls_only`
there is no path by which image features reach the box head. `full` and
`cls_only` use separate but identically-shaped heads, so they are
**parameter-matched** — only the input tensor differs.

## Architecture

```
6 cameras (B,6,3,448,800) ──► ResNet + FPN ──────► image features (B,2100,256)
                                                            │
LiDAR points ──► pillars (V,10,4) ──► PillarNet ──► BEV (B,256,H,W)
                                          │                 │
                                    class heatmap           │
                                          │                 │
                                    top-K local maxima      │
                                          ▼                 │
                              object queries (B,200,256)    │
                                          │                 │
                              LiDAR decoder (BEV attention) │
                                          │                 │
                              fusion decoder (SMCA) ◄───────┘
                                          │
                          ┌───────────────┴───────────────┐
                    class head                       box head
                   (B,200,10)                       (B,200,10)
```

Detection range ±51.2 m, pillar 0.2 m, 200 queries, `d_model` 256, 1 LiDAR +
1 fusion decoder layer.

---

## Repository layout

```
transfusion/
  models/
    transfusion.py       top-level model, forward pass
    head.py               query init, decoders, fusion routing (the ablation)
    loss.py               Hungarian matcher, focal / L1 / heatmap losses
    backbones.py          PillarFeatureNet, BEV backbone, image FPN
  data/
    nuscenes_dataset.py   loading, voxelisation, augmentation, targets
  tools/
    train.py              training loop, AMP, checkpointing
    evaluate.py            inference + nuScenes devkit evaluation
  configs/
    nuscenes.yaml          all hyper-parameters
tracker/
  track.py                 CenterPoint-style BEV tracker (AMOTA)
diagnostics/
  check_collapse.sh        query-degeneracy gate
  check_localisation.py    prediction-to-ground-truth distance analysis
  check_faithfulness.sh    paper-conformance audit
docs/
  CHANGES.md               deviations found and corrected, with impact
IMPROVEMENT_ROUND.md        pending changes: multisweep, CBGS, IoU matching
```

---

## Usage

```bash
# train one arm
python transfusion/tools/train.py \
    --config transfusion/configs/nuscenes.yaml \
    --mode cls_only --seed 42

# evaluate
python transfusion/tools/evaluate.py \
    --config transfusion/configs/nuscenes.yaml \
    --checkpoint work_dirs/ablation_seed42/cls_only/latest.pth \
    --out-dir eval_results/

# track
python tracker/track.py eval_results/results_nusc.json --out tracks.json --eval
```

---

## Diagnostics

Query-based detectors fail in ways the training loss does not reveal. During
development, all 200 object queries collapsed to a single point while training
loss declined normally for three days — the model emitted one detection per
scene and nothing in the log looked wrong.

**`check_collapse.sh`** — spatial spread of query predictions across a
checkpoint. A negative test: failing it means the model has degenerated;
passing only rules out that one failure mode. Runs in seconds.

**`check_localisation.py`** — median distance from confident predictions to
the nearest same-class ground truth, by range. The positive test spread cannot
give: a model can be well spread out and uniformly wrong (as `full` was,
pre-fix, at ~20 m).

**`check_faithfulness.sh`** — greps the codebase against a checklist of every
paper-specified hyperparameter and component, read-only.

---

## Known deviations from the reference papers

| Deviation | Status |
|---|---|
| Centre predicted as offset from query position | **fixed** |
| Matching-cost coefficients (0.15/0.25) | **fixed** |
| Category-aware query embedding | **fixed** |
| Local-max query selection, ped/cone exemption | **fixed** |
| Image resize augmentation | **fixed** |
| Global scaling range (0.9–1.1) | **fixed** |
| 10-sweep LiDAR accumulation | pending, patch written |
| CBGS class-balanced sampling | pending, patch written |
| IoU term in matching cost | pending, patch written |
| Copy-and-paste augmentation | not implemented |
| SMCA Gaussian attention mask | patch written, ungated |
| Image-guided query initialisation | not implemented |
| Two-stage training (20+6 epochs) | deliberately excluded — DAL Table 1 shows single-stage training is competitive |
| Backbone: PointPillars vs VoxelNet | deliberate choice, reported separately in both papers |

Full detail and published impact for each in `docs/CHANGES.md`.

---

## References

[1] Bai et al., *TransFusion: Robust LiDAR-Camera Fusion for 3D Object
Detection with Transformers*, CVPR 2022.
[2] Huang et al., *Detecting As Labeling: Rethinking LiDAR-camera Fusion in 3D
Object Detection*, arXiv:2311.07152, 2023.
[3] Yin et al., *Center-based 3D Object Detection and Tracking*, CVPR 2021.
[4] Zhu et al., *Class-balanced Grouping and Sampling for Point Cloud 3D
Object Detection*, arXiv:1908.09492, 2019.
[5] Caesar et al., *nuScenes: A Multimodal Dataset for Autonomous Driving*,
CVPR 2020.
