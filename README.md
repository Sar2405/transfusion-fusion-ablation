# Fusion Strategies for LiDAR–Camera 3D Object Detection

A controlled ablation of LiDAR–camera fusion philosophies in a
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
  into regression invites overfitting ("Rule B").
- **DeepInteraction** [3] takes a third position: never collapse LiDAR and
  camera into one representation at all — maintain both throughout, and let
  prediction alternate between them.

The papers are hard to compare directly: different backbones, schedules,
augmentation, data pipelines. This project isolates the fusion strategy —
one architecture, one training recipe, one dataset — behind a single
configuration switch that changes **only which feature source feeds which
prediction head**.

---

## The arms

Classification always reads the fused LiDAR+camera query. What differs is
the box regression input, and — for `interact` — how many times fusion
happens before prediction:

```python
if self.fusion_mode == "full":
    cls_src, box_src = fused_query, fused_query          # TransFusion
elif self.fusion_mode == "cls_only":
    cls_src, box_src = fused_query, lidar_query           # DAL Rule B
elif self.fusion_mode == "interact":
    cls_src, box_src = fused_query, fused_query           # alternating cascade
    # (fused_query here is the output of alternating LiDAR/image decoder
    # layers, not a single fusion pass — see note below)
```

`lidar_query` is snapshotted **before** the fusion decoder runs, so in
`cls_only` there is no path by which image features reach the box head —
not by convention, by construction. `full` and `cls_only` are
**parameter-matched**: same head shapes, only the input tensor differs.

| arm | status | camera reaches regression? | notes |
|---|---|---|---|
| `full` | stable | yes | TransFusion baseline |
| `cls_only` | stable | no | DAL Rule B |
| `dual_stream` | retired | no | superseded by `interact`; kept in results as a "did not replicate" case, see below |
| `interact` | patch written, **not yet verified against live code** | yes | alternating LiDAR/image decoder cascade, DeepInteraction-**inspired** |

> **`interact` is explicitly not a reproduction of DeepInteraction.** It
> captures only the alternating-cascade principle (their ablation prices
> this at the smallest of their three components, ~+0.4 mAP). It does not
> implement their MMRI encoder (dense bilateral cross-modal attention,
> their largest single contribution) or their MMPI decoder (RoI-align +
> dynamic convolution). Described as "DeepInteraction-inspired" throughout,
> never as a reproduction.

`dual_stream` results are retained in an appendix rather than deleted —
across two seeds its ordering relative to `cls_only` did not replicate,
which is itself a reported finding, not a discarded run.

---

## Architecture

```
6 cameras (B,6,3,448,800) ──► ResNet50 + FPN ────► image features (B,6,56,100,256)
                                                            │
LiDAR points ──► pillars (V,10,4) ──► PointPillars ──► BEV (B,256,H,W)
                                          │                 │
                                    class heatmap           │
                                          │                 │
                                    top-K local maxima      │
                                          ▼                 │
                              object queries (B,200,256)    │
                                          │                 │
                              LiDAR decoder (BEV attention) │
                                          │                 │
                          lidar_query snapshot ──────────┐  │
                                          │               │  │
                              fusion decoder ◄────────────┼──┘
                              (1 layer: full/cls_only,     │
                               4 alternating: interact)    │
                                          │               │
                          ┌───────────────┴───────────┐   │
                    class head                   box head │
                   (B,200,10)         reads fused_query OR │
                                       lidar_query, by arm ─┘
```

Detection range ±51.2 m, pillar 0.2 m, 200 queries, `d_model` 256, 1 LiDAR
decoder layer, 1 fusion decoder layer (4 alternating for `interact`).

Box code: `[x_norm, y_norm, z_norm, log_l, log_w, log_h, sin_yaw, cos_yaw,
vx, vy]` — centre predicted as an **offset** from the query's heatmap seed
position, not absolute (paper §3.3).

---

## Fidelity notes

Comparing arms to their reference papers surfaced two mechanisms present in
the papers but not (yet, or not correctly) in this codebase. Both are
tracked as scoped, gated patches rather than silent additions, since an
uncontrolled change to the fusion mechanism would compromise the arm
comparison this project exists to run.

**IoU term in the Hungarian matching cost** (TransFusion Eq. 1, `λ₃·L_iou`)
— **implemented and unit-verified**, currently gated for training-time
regression before being adopted in the main ablation. Matching previously
used classification + L1 centre/size distance only, with no term rewarding
actual box overlap; plausibly relevant given predictions are diagnosed as
recall-adequate but poorly ranked. Controlled by `cost_iou_w` in config
(default `0.0`, i.e. off, so existing runs are unaffected until explicitly
enabled).

**SMCA — Spatially Modulated Cross Attention** (TransFusion §3.4) — a 2D
Gaussian mask around each query's projected box, applied to the fusion
decoder's cross-attention so queries attend locally rather than over the
full image feature map. **Patch written, not yet applied.** Its integration
is intentionally deferred behind an `img_feat_h`/`img_feat_w` fix: those
values were hardcoded defaults (`14, 25`) disconnected from the image
backbone's real output stride (confirmed at runtime to be `56, 100`) — SMCA
would silently mask the wrong pixels if applied first. SMCA also only
reaches the box head in `full` and `interact`, not `cls_only` — by design,
since DAL's Rule B specifically forbids that path; this asymmetry is
reported explicitly rather than treated as a confound.

---

## Repository layout

```
transfusion/
  models/
    transfusion.py       top-level model, forward pass, BEV/image-feature
                         geometry derivation
    head.py               query init, decoders, fusion routing (the ablation)
    loss.py               Hungarian matcher, focal / L1 / heatmap losses
    iou_cost.py            BEV IoU matching-cost term (TransFusion Eq. 1)
    smca.py                spatially modulated cross-attention mask
    backbones.py          PillarFeatureNet, BEV backbone, image FPN
  data/
    nuscenes_dataset.py   loading, voxelisation, augmentation, targets
  tools/
    train.py              training loop, AMP/bf16, checkpointing
    evaluate.py            inference + nuScenes devkit evaluation, --mode
                          (must match the checkpoint's trained arm — no
                          safe default; embedded in the checkpoint dict)
  configs/
    nuscenes.yaml          base config (fp32)
    nuscenes_bf16.yaml      bf16 variant — ~3x faster, no accuracy cost
                          (validated), used for the ablation round
tracker/
  track.py                 CenterPoint-style BEV tracker (AMOTA / AMOTP)
diagnostics/
  check_collapse.sh        query-degeneracy gate (negative test)
  check_localisation.py    prediction-to-ground-truth distance by range
                          (positive test)
  check_faithfulness.sh    paper-conformance audit
  compare_seeds.sh          cross-seed replication table, all arms
  compare_precision.sh      fp32 vs bf16 comparison
  compare_range_class.sh    per-class AP / per-range recall, both seeds
  compare_reference.sh      results vs published reference numbers
  loss_components.sh        per-epoch heatmap/cls/box loss breakdown
```

---

## Usage

```bash
# train one arm
python transfusion/tools/train.py \
    --config transfusion/configs/nuscenes_bf16.yaml \
    --fusion-mode cls_only --seed 42 \
    --work-dir work_dirs/ablation_seed42

# evaluate — --mode must match the checkpoint's trained arm
python transfusion/tools/evaluate.py \
    --config transfusion/configs/nuscenes_bf16.yaml \
    --checkpoint work_dirs/ablation_seed42/cls_only/latest.pth \
    --mode cls_only \
    --out-dir eval_results/

# track
python tracker/track.py eval_results/results_nusc.json --out tracks.json --eval
```

---

## Diagnostics

Query-based detectors fail in ways the training loss does not reveal. During
development, all 200 object queries collapsed to a single point while
training loss declined normally for three days — the model emitted one
detection per scene and nothing in the log looked wrong. A separate bug had
`evaluate.py` silently defaulting to the wrong architecture regardless of
which checkpoint was loaded, which **inverted the measured arm ordering
twice** before discovery.

**`check_collapse.sh`** — spatial spread of query predictions across a
checkpoint. A negative test: failing it means the model has degenerated;
passing only rules out that one failure mode. Runs in seconds.

**`check_localisation.py`** — median distance from confident predictions to
the nearest same-class ground truth, by range. The positive test spread
cannot give: a model can be well spread out and uniformly wrong.

**`check_faithfulness.sh`** — read-only audit of whether config/code values
match the reference papers' specified values (voxel size, matching-cost
coefficients, augmentation ranges, etc.), independent of training results.

---

## Known limitations (in scope for the thesis, not silently fixed)

- **BEV resolution** is ~1.6 m/cell (`out_size_factor: 8`); TransFusion's
  VoxelNet configuration achieves 0.6 m. Query positions are quantised to
  roughly this cell size before the offset head can refine them.
- **Two-stage training, VoxelNet backbone, and DeepInteraction's full
  MMRI/MMPI mechanisms are deliberately excluded** — see thesis methodology
  for the per-item justification (compute budget, build risk, and — for
  MMRI/MMPI — the fact that adding representation-learning capacity to one
  arm only would break the parameter-matched comparison this project is
  built around).
- Absolute detection accuracy (NDS/mAP) trails the reference papers; the
  project's claims concern **relative ordering between arms**, replicated
  across seeds, not absolute state-of-the-art performance.

---

## References

[1] Bai et al., *TransFusion: Robust LiDAR-Camera Fusion for 3D Object
Detection with Transformers*, CVPR 2022.
[2] Huang et al., *Detecting As Labeling: Rethinking LiDAR-camera Fusion in
3D Object Detection*, arXiv:2311.07152, 2023.
[3] Yang et al., *DeepInteraction: 3D Object Detection via Modality
Interaction*, NeurIPS 2022.
[4] Yin et al., *Center-based 3D Object Detection and Tracking*, CVPR 2021.
[5] Zhu et al., *Class-balanced Grouping and Sampling for Point Cloud 3D
Object Detection*, arXiv:1908.09492, 2019.
[6] Caesar et al., *nuScenes: A Multimodal Dataset for Autonomous Driving*,
CVPR 2020.
