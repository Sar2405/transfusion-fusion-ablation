"""Shared utility functions for TransFusion."""
from __future__ import annotations

import math
import logging
from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Positional encodings
# ---------------------------------------------------------------------------

def pos2posemb2d(
    pos: Tensor,
    num_pos_feats: int = 128,
    temperature: int = 10_000,
) -> Tensor:
    """
    Sinusoidal 2-D positional embedding.

    Args:
        pos: (..., 2) normalised coordinates in [0, 1].
        num_pos_feats: half the output dimension (total = 2 * num_pos_feats).
        temperature: wavelength base.

    Returns:
        (..., 2 * num_pos_feats)
    """
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    pos_x = pos[..., 0:1] / dim_t
    pos_y = pos[..., 1:2] / dim_t

    pos_x = torch.stack(
        [pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], dim=-1
    ).flatten(-2)
    pos_y = torch.stack(
        [pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], dim=-1
    ).flatten(-2)
    return torch.cat([pos_x, pos_y], dim=-1)


def pos2posemb3d(
    pos: Tensor,
    num_pos_feats: int = 128,
    temperature: int = 10_000,
) -> Tensor:
    """Sinusoidal 3-D positional embedding. Returns (..., 3 * num_pos_feats)."""
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    def encode(v: Tensor) -> Tensor:
        v = v / dim_t
        return torch.stack([v[..., 0::2].sin(), v[..., 1::2].cos()], dim=-1).flatten(-2)

    return torch.cat([encode(pos[..., i:i+1]) for i in range(3)], dim=-1)


# ---------------------------------------------------------------------------
# Network building blocks
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Configurable multi-layer perceptron."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        layers: List[nn.Module] = []
        for i in range(num_layers):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ConvBnAct(nn.Module):
    """Conv2d → BatchNorm2d → ReLU block."""

    def __init__(
        self,
        in_c: int,
        out_c: int,
        k: int = 3,
        s: int = 1,
        p: int = 1,
        groups: int = 1,
        act: bool = True,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(in_c, out_c, k, s, p, groups=groups, bias=False),
            nn.BatchNorm2d(out_c, momentum=0.1, eps=1e-5),
        ]
        if act:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Gaussian heatmap helpers
# ---------------------------------------------------------------------------

def gaussian_2d(shape: tuple, sigma: float = 1.0):
    """Generate a 2-D Gaussian kernel (numpy)."""
    import numpy as np
    m = (shape[0] - 1) / 2.0
    n = (shape[1] - 1) / 2.0
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def gaussian_radius(det_size: tuple, min_overlap: float = 0.5) -> float:
    """Compute minimum Gaussian radius given box size and overlap threshold."""
    height, width = det_size
    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = (b1 ** 2 - 4 * a1 * c1) ** 0.5
    r1 = (b1 - sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = (b2 ** 2 - 4 * a2 * c2) ** 0.5
    r2 = (b2 - sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = (b3 ** 2 - 4 * a3 * c3) ** 0.5
    r3 = (b3 + sq3) / (2 * a3)
    return min(r1, r2, r3)


def draw_heatmap_gaussian(heatmap, center: tuple, radius: int, k: int = 1):
    """Draw a Gaussian blob on a numpy heatmap array."""
    import numpy as np
    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap  = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[radius - top : radius + bottom, radius - left : radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


# ---------------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------------

def nms_bev(
    boxes: Tensor,
    scores: Tensor,
    iou_threshold: float = 0.2,
) -> Tensor:
    """
    BEV NMS using axis-aligned IoU approximation.
    boxes : (N, 7+)  first 5 dims are [x, y, z, w, l, ...]
    scores: (N,)
    """
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)

    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break

        cx1, cy1, w1, l1 = boxes[i, 0], boxes[i, 1], boxes[i, 3], boxes[i, 4]
        cx2 = boxes[order[1:], 0]
        cy2 = boxes[order[1:], 1]
        w2  = boxes[order[1:], 3]
        l2  = boxes[order[1:], 4]

        inter_w = (torch.minimum(cx1 + w1 / 2, cx2 + w2 / 2) -
                   torch.maximum(cx1 - w1 / 2, cx2 - w2 / 2)).clamp(min=0)
        inter_l = (torch.minimum(cy1 + l1 / 2, cy2 + l2 / 2) -
                   torch.maximum(cy1 - l1 / 2, cy2 - l2 / 2)).clamp(min=0)
        inter   = inter_w * inter_l
        union   = w1 * l1 + w2 * l2 - inter
        iou     = inter / union.clamp(min=1e-6)

        order = order[1:][iou <= iou_threshold]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


# ---------------------------------------------------------------------------
# Box decoding
# ---------------------------------------------------------------------------

def decode_boxes(pred_boxes: Tensor, pc_range: List[float]) -> Tensor:
    """
    Decode normalised box predictions to metric coordinates.
    pred_boxes: (N, 10) [x_norm, y_norm, z_norm, log_w, log_l, log_h, sin, cos, vx, vy]
    Returns:    (N, 10) [x, y, z, w, l, h, sin, cos, vx, vy]
    """
    out = pred_boxes.clone()
    out[:, 0] = pred_boxes[:, 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
    out[:, 1] = pred_boxes[:, 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
    out[:, 2] = pred_boxes[:, 2] * (pc_range[5] - pc_range[2]) + pc_range[2]
    out[:, 3:6] = pred_boxes[:, 3:6].exp()   # log_w, log_l, log_h → w, l, h
    return out


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def get_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    log = logging.getLogger(name)
    if log.handlers:          # avoid duplicate handlers on re-import
        return log
    log.setLevel(level)
    fmt = logging.Formatter("[%(asctime)s %(name)s %(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    log.addHandler(ch)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        log.addHandler(fh)
    return log
