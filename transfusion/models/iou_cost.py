"""
BEV IoU cost for TransFusionMatcher (TransFusion Eq. 1, the lambda_3 term).

    C_match = l1*L_cls + l2*L_reg + l3*L_iou

Box format (from head.py._assemble_box), normalised + log-encoded:
    [0] x_norm      centre x in [0,1] within pc_range
    [1] y_norm      centre y in [0,1]
    [2] z_norm      centre z in [0,1]
    [3] log d0      -> exp() = metres   (length or width: see dim_order)
    [4] log d1      -> exp() = metres
    [5] log h
    [6] sin_yaw
    [7] cos_yaw
    [8] vx
    [9] vy
"""

from __future__ import annotations

import torch
from torch import Tensor


# --------------------------------------------------------------------------
def decode_bev(
    boxes: Tensor,                 # (K, 10) normalised + log-encoded
    pc_range,
    dim_order: str = "lw",         # "lw": idx3=length, idx4=width
    log_clamp: float = 4.0,        # exp(4) = 54.6 m -- generous, blocks inf
):
    """
    (K,10) -> (cx_m, cy_m, l_m, w_m, yaw), each (K,)

    An under-trained box head can emit huge raw log-dims; exp() of those is
    inf, which would poison the IoU matrix before the matcher's nan_to_num
    guard runs. Clamp first.
    """
    cx = boxes[:, 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
    cy = boxes[:, 1] * (pc_range[4] - pc_range[1]) + pc_range[1]

    d0 = boxes[:, 3].clamp(-log_clamp, log_clamp).exp()
    d1 = boxes[:, 4].clamp(-log_clamp, log_clamp).exp()
    if dim_order == "lw":
        l, w = d0, d1
    elif dim_order == "wl":
        w, l = d0, d1
    else:
        raise ValueError(f"dim_order must be 'lw' or 'wl', got {dim_order}")

    yaw = torch.atan2(boxes[:, 6], boxes[:, 7])
    return cx, cy, l, w, yaw


# --------------------------------------------------------------------------
def bev_iou_axis_aligned(p_cx, p_cy, p_l, p_w,
                         g_cx, g_cy, g_l, g_w,
                         eps: float = 1e-7) -> Tensor:
    """Pairwise (N,M) IoU, yaw ignored. All inputs 1-D, metres."""
    n, m = p_cx.numel(), g_cx.numel()
    if n == 0 or m == 0:
        return p_cx.new_zeros((n, m))

    p_hl, p_hw = (p_l * .5).view(n, 1), (p_w * .5).view(n, 1)
    g_hl, g_hw = (g_l * .5).view(1, m), (g_w * .5).view(1, m)
    pcx, pcy = p_cx.view(n, 1), p_cy.view(n, 1)
    gcx, gcy = g_cx.view(1, m), g_cy.view(1, m)

    iw = (torch.min(pcx + p_hl, gcx + g_hl) -
          torch.max(pcx - p_hl, gcx - g_hl)).clamp(min=0)
    ih = (torch.min(pcy + p_hw, gcy + g_hw) -
          torch.max(pcy - p_hw, gcy - g_hw)).clamp(min=0)
    inter = iw * ih

    union = (p_l * p_w).view(n, 1) + (g_l * g_w).view(1, m) - inter
    return inter / (union + eps)


# --------------------------------------------------------------------------
def bev_iou_nearest(p_cx, p_cy, p_l, p_w, p_yaw,
                    g_cx, g_cy, g_l, g_w, g_yaw,
                    eps: float = 1e-7) -> Tensor:
    """
    Axis-aligned IoU on the AABB of each ROTATED box. Cheap, captures some
    orientation effect. Closest cheap analogue to mmdet3d's
    bbox_overlaps_nearest_3d; sensible default without a compiled rotated
    IoU kernel. True areas are kept in the union so IoU is not inflated.
    """
    n, m = p_cx.numel(), g_cx.numel()
    if n == 0 or m == 0:
        return p_cx.new_zeros((n, m))

    def aabb(l, w, yaw):
        c, s = torch.cos(yaw).abs(), torch.sin(yaw).abs()
        return l * c + w * s, l * s + w * c

    pex, pey = aabb(p_l, p_w, p_yaw)
    gex, gey = aabb(g_l, g_w, g_yaw)

    p_hx, p_hy = (pex * .5).view(n, 1), (pey * .5).view(n, 1)
    g_hx, g_hy = (gex * .5).view(1, m), (gey * .5).view(1, m)
    pcx, pcy = p_cx.view(n, 1), p_cy.view(n, 1)
    gcx, gcy = g_cx.view(1, m), g_cy.view(1, m)

    iw = (torch.min(pcx + p_hx, gcx + g_hx) -
          torch.max(pcx - p_hx, gcx - g_hx)).clamp(min=0)
    ih = (torch.min(pcy + p_hy, gcy + g_hy) -
          torch.max(pcy - p_hy, gcy - g_hy)).clamp(min=0)
    inter = iw * ih

    union = (p_l * p_w).view(n, 1) + (g_l * g_w).view(1, m) - inter
    return (inter / (union + eps)).clamp(0.0, 1.0)


# --------------------------------------------------------------------------
def iou_matching_cost(
    pred_boxes: Tensor,            # (Q, 10) normalised + log
    gt_boxes: Tensor,              # (G, 10) normalised + log
    pc_range,
    dim_order: str = "lw",
    mode: str = "nearest",         # "nearest" | "aabb"
    external_iou_fn=None,
) -> Tensor:
    """
    Returns (Q, G) IoU COST = -IoU, ready to add to the existing cost terms.

    external_iou_fn: if you have a compiled rotated-IoU routine (e.g. whatever
    backs nms_bev), pass it as fn(pred_bev, gt_bev) -> (Q,G) IoU with bev
    tensors packed [cx, cy, l, w, yaw]. Preferred over the fallbacks.
    """
    p = decode_bev(pred_boxes, pc_range, dim_order)
    g = decode_bev(gt_boxes,   pc_range, dim_order)

    if external_iou_fn is not None:
        iou = external_iou_fn(torch.stack(p, -1), torch.stack(g, -1))
    elif mode == "nearest":
        iou = bev_iou_nearest(*p, *g)
    else:
        iou = bev_iou_axis_aligned(p[0], p[1], p[2], p[3],
                                   g[0], g[1], g[2], g[3])
    return -iou
