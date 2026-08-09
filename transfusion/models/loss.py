"""
TransFusion loss functions.

Components
----------
* GaussianFocalLoss   – CenterPoint-style modified focal loss for heatmaps
* FocalLoss           – sigmoid focal loss for query classification
* TransFusionMatcher  – true Hungarian matching via scipy
* TransFusionLoss     – full combined loss with auxiliary stage-1 term
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Loss primitives
# ---------------------------------------------------------------------------

class GaussianFocalLoss(nn.Module):
    """
    Modified focal loss for heatmap regression (CornerNet / CenterPoint).
    Positive pixels use standard focal loss; negative pixels near the Gaussian
    peak are down-weighted by (1-target)^beta.

    Reference:
        Law & Deng, "CornerNet: Detecting Objects as Paired Keypoints", ECCV 2018.
    """

    def __init__(self, alpha: float = 2.0, beta: float = 4.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta  = beta

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            pred:   (B, C, H, W) raw logits.
            target: (B, C, H, W) Gaussian heatmap in [0, 1].
        """
        pred_sigmoid = pred.sigmoid()
        eps = 1e-6

        pos_mask = (target == 1.0).float()
        neg_mask = 1.0 - pos_mask

        pos_loss = (
            -torch.log(pred_sigmoid.clamp(min=eps))
            * (1 - pred_sigmoid) ** self.alpha
            * pos_mask
        )
        neg_loss = (
            -torch.log((1 - pred_sigmoid).clamp(min=eps))
            * pred_sigmoid ** self.alpha
            * (1 - target) ** self.beta
            * neg_mask
        )

        num_pos = pos_mask.sum().clamp(min=1)
        return (pos_loss.sum() + neg_loss.sum()) / num_pos


class SigmoidFocalLoss(nn.Module):
    """Standard sigmoid focal loss for multi-label classification."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """pred: (N, C) logits; target: (N, C) binary float."""
        p = pred.sigmoid()
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        pt  = p * target + (1 - p) * (1 - target)
        fl  = self.alpha * (1 - pt) ** self.gamma * bce
        return fl.sum()


# ---------------------------------------------------------------------------
# Hungarian matcher
# ---------------------------------------------------------------------------

class TransFusionMatcher(nn.Module):
    """
    Bipartite matching between predicted and ground-truth boxes.

    Cost = w_cls * (-prob_gt_class) + w_box * L1(pred_box, gt_box)

    Uses scipy.optimize.linear_sum_assignment for the exact assignment.
    """

    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox:  float = 1.0,
    ) -> None:
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox  = cost_bbox

    @torch.no_grad()
    def forward(
        self,
        pred_logits: Tensor,        # (B, Q, C)
        pred_boxes:  Tensor,        # (B, Q, 10)
        gt_labels:   List[Tensor],  # list of (Gi,) int
        gt_boxes:    List[Tensor],  # list of (Gi, 10) float
    ) -> List[Tuple[Tensor, Tensor]]:
        from scipy.optimize import linear_sum_assignment

        indices = []
        B = pred_logits.shape[0]

        for b in range(B):
            G = len(gt_labels[b])
            if G == 0:
                indices.append((
                    torch.zeros(0, dtype=torch.long, device=pred_logits.device),
                    torch.zeros(0, dtype=torch.long, device=pred_logits.device),
                ))
                continue

            prob = pred_logits[b].softmax(-1)           # (Q, C)
            cls_cost = -prob[:, gt_labels[b]]           # (Q, G)

            # Normalise box cost to [0, 1] range
            pb = pred_boxes[b]                          # (Q, 10)
            gb = gt_boxes[b]                            # (G, 10)
            bbox_cost = torch.cdist(pb[:, :6], gb[:, :6], p=1) / 6.0  # (Q, G)

            cost = (
                self.cost_class * cls_cost +
                self.cost_bbox  * bbox_cost
            )
            # Defensive guard: an occasionally unstable prediction (e.g. an
            # under-trained box head briefly producing a very large raw
            # log_l/log_w/log_h value) can make cdist emit inf/nan, which
            # scipy's linear_sum_assignment rejects outright and crashes the
            # whole training job. Replace any non-finite entries with a large
            # but finite cost so that pairing is merely de-prioritised rather
            # than the batch aborting training.
            cost = torch.nan_to_num(cost, nan=1e6, posinf=1e6, neginf=-1e6)
            cost = cost.cpu().numpy()

            row_ind, col_ind = linear_sum_assignment(cost)
            indices.append((
                torch.as_tensor(row_ind, dtype=torch.long, device=pred_logits.device),
                torch.as_tensor(col_ind, dtype=torch.long, device=pred_logits.device),
            ))

        return indices


# ---------------------------------------------------------------------------
# Gaussian heatmap target builder
# ---------------------------------------------------------------------------

def build_heatmap_targets(
    gt_boxes_list: List[Tensor],       # list of (Gi, 10)  normalised + log-encoded
    gt_labels_list: List[Tensor],      # list of (Gi,) int
    num_classes: int,
    bev_h: int,
    bev_w: int,
    min_radius: int = 2,
    gaussian_overlap: float = 0.1,
    device: torch.device = torch.device("cpu"),
    pc_range: Tuple[float, ...] = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
) -> Tensor:
    """
    Construct per-batch Gaussian heatmap targets in [0, 1].
    Returns (B, C, H, W).

    Box code is [x_n, y_n, z_n, log_l, log_w, log_h, sin, cos, vx, vy], so:
      - centre comes from x_n, y_n  (already normalised to the BEV grid)
      - the Gaussian radius is derived from the REAL metric dimensions, which
        means the log-encoded length/width must be exponentiated first and then
        converted from metres to BEV pixels via the per-cell resolution.
    """
    import numpy as np

    span_x = pc_range[3] - pc_range[0]
    span_y = pc_range[4] - pc_range[1]
    px_per_m_x = bev_w / span_x
    px_per_m_y = bev_h / span_y

    B = len(gt_boxes_list)
    heatmaps = np.zeros((B, num_classes, bev_h, bev_w), dtype=np.float32)

    for b in range(B):
        boxes  = gt_boxes_list[b].cpu().numpy()   # (Gi, 10)
        labels = gt_labels_list[b].cpu().numpy()  # (Gi,)

        for i in range(len(labels)):
            cls = int(labels[i])
            # x_norm, y_norm → BEV pixel coords
            cx = int(np.clip(boxes[i, 0] * bev_w, 0, bev_w - 1))
            cy = int(np.clip(boxes[i, 1] * bev_h, 0, bev_h - 1))

            # log_l, log_w → metres → BEV pixels
            length_m = float(np.exp(boxes[i, 3]))   # log_l → metres
            width_m  = float(np.exp(boxes[i, 4]))   # log_w → metres
            l_px = max(1, int(length_m * px_per_m_y))
            w_px = max(1, int(width_m  * px_per_m_x))
            radius = max(min_radius, int(
                _gaussian_radius((l_px, w_px), min_overlap=gaussian_overlap)
            ))
            _draw_gaussian(heatmaps[b, cls], (cx, cy), radius)

    return torch.from_numpy(heatmaps).to(device)


def _gaussian_radius(det_size: Tuple[int, int], min_overlap: float = 0.5) -> float:
    h, w = det_size
    a1 = 1; b1 = h + w; c1 = w * h * (1 - min_overlap) / (1 + min_overlap)
    sq1 = (b1**2 - 4*a1*c1)**0.5
    r1  = (b1 - sq1) / (2 * a1)
    a2 = 4; b2 = 2*(h+w); c2 = (1-min_overlap)*w*h
    sq2 = (b2**2 - 4*a2*c2)**0.5
    r2  = (b2 - sq2) / (2 * a2)
    a3 = 4*min_overlap; b3 = -2*min_overlap*(h+w); c3 = (min_overlap-1)*w*h
    sq3 = (b3**2 - 4*a3*c3)**0.5
    r3  = (b3 + sq3) / (2 * a3)
    return min(r1, r2, r3)


def _draw_gaussian(heatmap: np.ndarray, center: Tuple[int, int], radius: int) -> None:
    diameter = 2 * radius + 1
    sigma = diameter / 6.0
    m = n = radius
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    gauss = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    gauss[gauss < np.finfo(gauss.dtype).eps * gauss.max()] = 0

    cx, cy = center
    H, W = heatmap.shape
    left  = min(cx, radius); right  = min(W - cx, radius + 1)
    top   = min(cy, radius); bottom = min(H - cy, radius + 1)
    h_crop = heatmap[cy - top : cy + bottom, cx - left : cx + right]
    g_crop = gauss[radius - top : radius + bottom, radius - left : radius + right]
    if h_crop.size > 0:
        np.maximum(h_crop, g_crop, out=h_crop)


# ---------------------------------------------------------------------------
# Full TransFusion loss
# ---------------------------------------------------------------------------

class TransFusionLoss(nn.Module):
    """
    Combined loss for TransFusion.

    Stage-1 auxiliary loss + stage-2 main loss, each consisting of:
      - Gaussian focal loss on BEV heatmap
      - Sigmoid focal loss on matched query classes
      - L1 regression loss on matched query boxes

    Args
    ----
    num_classes         : number of detection categories
    heatmap_weight      : λ for heatmap loss
    cls_weight          : λ for query classification loss
    bbox_weight         : λ for box regression loss
    aux_weight          : multiplier for stage-1 auxiliary losses
    focal_alpha / gamma : sigmoid focal loss hyper-parameters
    gaussian_overlap    : min IoU used to compute Gaussian radii
    min_radius          : minimum heatmap Gaussian radius (in pixels)
    """

    def __init__(
        self,
        num_classes: int        = 10,
        heatmap_weight: float   = 1.0,
        cls_weight: float       = 1.0,
        bbox_weight: float      = 0.25,
        aux_weight: float       = 0.5,
        focal_alpha: float      = 0.25,
        focal_gamma: float      = 2.0,
        gaussian_overlap: float = 0.1,
        min_radius: int         = 2,
        pc_range: Tuple[float, ...] = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.w_heat      = heatmap_weight
        self.w_cls       = cls_weight
        self.w_box       = bbox_weight
        self.w_aux       = aux_weight
        self.gaussian_overlap = gaussian_overlap
        self.min_radius  = min_radius
        self.pc_range    = pc_range

        self.heatmap_loss = GaussianFocalLoss()
        self.cls_loss     = SigmoidFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.matcher      = TransFusionMatcher(cost_class=0.15, cost_bbox=0.25)

    # ------------------------------------------------------------------ #
    def _cls_and_box_loss(
        self,
        pred_logits: Tensor,         # (B, Q, C)
        pred_boxes:  Tensor,         # (B, Q, 10)
        gt_labels:   List[Tensor],
        gt_boxes:    List[Tensor],
        indices:     Optional[List[Tuple[Tensor, Tensor]]] = None,
    ) -> Tuple[Tensor, Tensor, List[Tuple[Tensor, Tensor]]]:
        B, Q, C = pred_logits.shape
        device = pred_logits.device

        if indices is None:
            indices = self.matcher(pred_logits.detach(), pred_boxes.detach(),
                                   gt_labels, gt_boxes)

        # Classification target (B, Q, C) binary
        target_cls = torch.zeros(B, Q, C, device=device)
        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue
            target_cls[b, pred_idx, gt_labels[b][gt_idx]] = 1.0

        loss_cls = self.cls_loss(pred_logits, target_cls)
        num_total = max(sum(len(gt) for gt in gt_labels), 1)
        loss_cls = loss_cls / num_total * self.w_cls

        # Box regression: L1 on matched pairs
        loss_box = pred_logits.new_zeros(1)
        num_matched = 0
        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue
            loss_box = loss_box + F.l1_loss(
                pred_boxes[b, pred_idx],
                gt_boxes[b][gt_idx],
                reduction="sum",
            )
            num_matched += len(pred_idx)
        if num_matched > 0:
            loss_box = loss_box / num_matched * self.w_box

        return loss_cls, loss_box.squeeze(), indices

    # ------------------------------------------------------------------ #
    def forward(
        self,
        outputs: Dict[str, Tensor],
        gt_labels: List[Tensor],
        gt_boxes:  List[Tensor],
        bev_h: int,
        bev_w: int,
    ) -> Dict[str, Tensor]:
        """
        Args
        ----
        outputs   : model output dict from TransFusionHead.forward()
        gt_labels : list[Tensor] (Gi,) int  per batch element
        gt_boxes  : list[Tensor] (Gi,10)    per batch element, normalised
        bev_h/w   : BEV feature spatial size (for heatmap target generation)
        """
        pred_logits   = outputs["pred_logits"]     # (B, Q, C)
        pred_boxes    = outputs["pred_boxes"]      # (B, Q, 10)
        pred_logits_s1 = outputs["pred_logits_s1"]
        pred_boxes_s1  = outputs["pred_boxes_s1"]
        heatmap        = outputs["heatmap"]        # (B, C, H, W)

        device = pred_logits.device

        # --- Heatmap loss ---
        heat_target = build_heatmap_targets(
            gt_boxes, gt_labels, self.num_classes, bev_h, bev_w,
            self.min_radius, self.gaussian_overlap, device, self.pc_range,
        )
        loss_heat = self.heatmap_loss(heatmap, heat_target) * self.w_heat

        # --- Stage-2 main loss ---
        loss_cls, loss_box, indices = self._cls_and_box_loss(
            pred_logits, pred_boxes, gt_labels, gt_boxes
        )

        # --- Stage-1 auxiliary loss (reuse matching indices) ---
        loss_cls_aux, loss_box_aux, _ = self._cls_and_box_loss(
            pred_logits_s1, pred_boxes_s1, gt_labels, gt_boxes, indices
        )
        loss_cls_aux  = loss_cls_aux  * self.w_aux
        loss_box_aux  = loss_box_aux  * self.w_aux

        total = loss_heat + loss_cls + loss_box + loss_cls_aux + loss_box_aux

        return {
            "loss":          total,
            "loss_heatmap":  loss_heat,
            "loss_cls":      loss_cls,
            "loss_box":      loss_box,
            "loss_cls_aux":  loss_cls_aux,
            "loss_box_aux":  loss_box_aux,
        }
