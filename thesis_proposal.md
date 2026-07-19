# Thesis Proposal

## A Controlled Study of Fusion Philosophies in Transformer-Based LiDAR-Camera 3D Object Detection

---

### Abstract

Multi-modal LiDAR-camera fusion underpins most state-of-the-art 3D object detectors, yet the field has not converged on a fundamental design question: where, and for which sub-task, camera information should be allowed to influence detection. Three influential methods embody three distinct fusion philosophies. TransFusion [1] predicts initial object boxes from LiDAR queries and then fuses those queries with image features in a second decoder stage, allowing camera information to influence the query representation used for detection. DAL [2] argues, from a data-labeling perspective, that regression should not involve camera features at all, treating metric geometry as the domain of LiDAR, and reports improved accuracy and speed-accuracy tradeoff under this design. DeepInteraction [3] criticises unified fusion for discarding modality-specific strengths and instead maintains separate per-modality representations that interact without merging. Crucially, these three methods also differ in their LiDAR backbones, image backbones, training schedules, augmentation strategies, and query mechanisms, so their reported differences cannot be attributed to fusion philosophy alone — the design choice of interest is entangled with many unrelated architectural factors.

This thesis investigates fusion philosophy as a controlled experimental variable. Building on a single fixed transformer-based detection architecture and holding the LiDAR backbone, image backbone, data pipeline, and training recipe constant, we implement three fusion variants that differ only in where and how camera information flows: (i) full fusion, in which camera informs both classification and regression, after TransFusion [1]; (ii) classification-only fusion, in which camera is restricted to classification while regression is driven by LiDAR, after DAL [2]; and (iii) dual-stream fusion, in which per-modality representations are maintained separately and interact without merging, after the design principle of DeepInteraction [3]. Because the other architectural factors are held fixed, observed performance differences are more cleanly attributable to fusion philosophy than in cross-paper comparisons, though attribution remains conditional on the single architecture studied.

We evaluate all three variants on the nuScenes benchmark, reporting both aggregate detection metrics and a per-class and per-range breakdown so that any differences in behaviour across object types and distances can be observed directly rather than obscured by aggregate scores. The intended contribution is a comparison of fusion philosophy under matched architectural conditions, reducing the confounds present in cross-paper comparisons, together with an empirical characterisation of how each philosophy behaves across object classes and ranges. This holds regardless of which variant achieves the highest aggregate score, reframing the discussion among TransFusion, DAL, and DeepInteraction from one of universal superiority toward a clearer, like-for-like understanding of the design tradeoff.

---

### Method Summary

| Variant | Camera to Classification | Camera to Regression | Modalities Merged | Source Philosophy |
|---|---|---|---|---|
| Full fusion | Yes | Yes | Yes (fused query) | TransFusion [1] |
| Classification-only | Yes | No | Yes for cls; LiDAR for reg | DAL [2] |
| Dual-stream | Yes | Via LiDAR stream | No (separate, interacting) | DeepInteraction [3] |

All three variants share an identical LiDAR backbone, image backbone (ResNet-50 + FPN), voxelisation, data augmentation, optimiser, learning-rate schedule, and training-epoch budget. The single point of variation is the routing of camera information within the detection head. The resulting attribution applies to this architecture and dataset; generalisation to other architectures is not claimed.

---

### Evaluation Plan

- **Benchmark:** nuScenes (validation split), reporting NDS, mAP, and the true-positive error metrics (mATE, mASE, mAOE, mAVE, mAAE).
- **Primary analysis:** per-class average precision and per-range bins (0–20 m, 20–30 m, 30–40 m, 40–50 m) for each variant.
- **Reliability:** each variant trained with at least two random seeds to separate genuine differences from training noise.
- **Optional robustness extension:** evaluation under degraded sensors (dropped cameras, added LiDAR noise) to observe how each philosophy fails.

---

### Indicative Timeline

| Phase | Work | Estimate |
|---|---|---|
| 1 | Environment setup, compiled voxeliser, single forward/backward pass on real data, evaluation pipeline | 3–5 weeks |
| 2 | Implement and unit-test the three fusion variants (full already done; classification-only and dual-stream) | 2–3 weeks |
| 3 | Train all variants to convergence (multiple seeds) | 2–4 weeks (wall-clock) |
| 4 | Per-class and per-range analysis, figures | 3 weeks |
| 5 | Writing, optional robustness extension | 3–4 weeks |

Estimated total: approximately 13–18 weeks, contingent on stable GPU access, which is the principal scheduling risk.

---

### References

[1] Bai, X., Hu, Z., Zhu, X., Huang, Q., Chen, Y., Fu, H., & Tai, C.-L. (2022). TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with Transformers. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 1080–1089. https://doi.org/10.1109/cvpr52688.2022.00116

[2] Huang, J., Ye, Y., Liang, Z., Shan, Y., & Du, D. (2023). Detecting As Labeling: Rethinking LiDAR-Camera Fusion in 3D Object Detection. arXiv:2311.07152. https://doi.org/10.48550/arxiv.2311.07152

[3] Yang, Z., Chen, J., Miao, Z., Li, W., Zhu, X., & Zhang, L. (2022). DeepInteraction: 3D Object Detection via Modality Interaction. Advances in Neural Information Processing Systems (NeurIPS), 35. arXiv:2208.11112. https://doi.org/10.48550/arxiv.2208.11112
