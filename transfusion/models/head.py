"""
TransFusion Detection Head.

Implements:
  1. Heatmap-based query initialisation
  2. LiDAR-only transformer decoder (stage 1)
  3. Image cross-attention fusion decoder (stage 2)
  4. Per-query prediction heads (class + 10-DoF box)
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..utils.common import MLP, ConvBnAct, pos2posemb2d, pos2posemb3d


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class FFN(nn.Module):
    """Post-attention feed-forward network with residual + layernorm."""

    def __init__(self, d_model: int, dim_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x + self.net(x))


class LiDARDecoderLayer(nn.Module):
    """
    One TransFusion decoder layer for the LiDAR-only stage.

    Order: cross-attn (BEV) → self-attn (queries) → FFN
    (matches the paper's Fig. 3 – BEV cross-attention first).
    """

    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.self_attn  = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.norm_ca = nn.LayerNorm(d_model)
        self.norm_sa = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, dim_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: Tensor,      # (B, Q, d)
        bev_feat: Tensor,     # (B, HW, d)
        query_pos: Tensor,    # (B, Q, d)  sinusoidal query position
        bev_pos: Tensor,      # (B, HW, d) sinusoidal BEV position
    ) -> Tensor:
        # 1. Cross-attention: queries → BEV
        q = queries + query_pos
        k = bev_feat + bev_pos
        ca_out, _ = self.cross_attn(query=q, key=k, value=bev_feat)
        queries = self.norm_ca(queries + self.dropout(ca_out))

        # 2. Self-attention among queries
        q = k = queries + query_pos
        sa_out, _ = self.self_attn(query=q, key=k, value=queries)
        queries = self.norm_sa(queries + self.dropout(sa_out))

        # 3. FFN
        return self.ffn(queries)


class ImageFusionLayer(nn.Module):
    """
    TransFusion image cross-attention fusion layer (stage 2).

    For each query its predicted 3-D centre is projected onto the image plane
    of each camera. A soft camera-selection score (learned MLP over the 3-D
    position) gates the contribution from each camera, then a standard
    cross-attention over image tokens enriches the query.

    When camera projection matrices are provided the gate is refined by a
    visibility prior: cameras where the point projects outside the image
    receive a large negative bias, which is equivalent to SMCA spatial
    constraint.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_ff: int,
        num_cameras: int,
        dropout: float = 0.1,
        img_h: int = 14,   # image feature spatial size (after backbone)
        img_w: int = 25,
    ) -> None:
        super().__init__()
        self.num_cameras = num_cameras
        self.img_h = img_h
        self.img_w = img_w

        # Camera selection: predicted xyz → per-camera weight
        self.cam_gate = MLP(3, d_model // 2, num_cameras, num_layers=2)

        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm_ca = nn.LayerNorm(d_model)
        self.norm_sa = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = FFN(d_model, dim_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def _camera_visibility_bias(
        self,
        pred_centers: Tensor,    # (B, Q, 3)  world xyz
        lidar2img: Optional[Tensor],  # (B, N, 4, 4) or None
    ) -> Optional[Tensor]:
        """Return (B, Q, N) additive logit bias; large-negative = invisible."""
        if lidar2img is None:
            return None
        B, Q, _ = pred_centers.shape
        N = lidar2img.shape[1]
        # Homogeneous world coords
        ones = pred_centers.new_ones(B, Q, 1)
        pts_h = torch.cat([pred_centers, ones], dim=-1)   # (B,Q,4)
        # Project: (B,N,4,4) @ (B,1,4,Q) → (B,N,4,Q)
        pts_cam = torch.einsum("bnij,bqj->bnqi", lidar2img, pts_h)  # (B,N,Q,4)
        depth = pts_cam[..., 2]                  # (B,N,Q)
        u = pts_cam[..., 0] / depth.clamp(min=1e-3)
        v = pts_cam[..., 1] / depth.clamp(min=1e-3)

        # Normalise to [0,1] image space (assume img_w x img_h)
        u_norm = u / (self.img_w * 32)          # rough stride-32 assumption
        v_norm = v / (self.img_h * 32)

        in_front = depth > 0
        in_bounds = (u_norm >= 0) & (u_norm <= 1) & (v_norm >= 0) & (v_norm <= 1)
        visible = (in_front & in_bounds).float()  # (B,N,Q)
        # Convert to additive bias: 0 for visible, -1e4 for invisible
        bias = (visible - 1.0) * 1e4             # (B,N,Q)
        return bias.permute(0, 2, 1)             # (B,Q,N)

    def forward(
        self,
        queries: Tensor,          # (B, Q, d)
        img_feats: Tensor,        # (B, N, HW, d)
        query_pos: Tensor,        # (B, Q, d)
        metric_centers: Tensor,   # (B, Q, 3)  predicted xyz in METRES (for projection)
        lidar2img: Optional[Tensor] = None,   # (B, N, 4, 4)
        norm_centers: Optional[Tensor] = None,  # (B, Q, 3) xyz in [0,1] (for gate MLP)
    ) -> Tensor:
        B, Q, d = queries.shape
        N, HW = img_feats.shape[1], img_feats.shape[2]

        # Camera gate weights. Feed the NORMALISED centres to the learned MLP
        # (well-conditioned inputs in [0,1]); use the METRIC centres only for
        # the geometric projection / visibility test.
        gate_input = norm_centers if norm_centers is not None else metric_centers
        gate_logits = self.cam_gate(gate_input)      # (B, Q, N)
        vis_bias = self._camera_visibility_bias(metric_centers, lidar2img)
        if vis_bias is not None:
            gate_logits = gate_logits + vis_bias
        cam_weights = gate_logits.softmax(dim=-1)    # (B, Q, N)

        # Weighted aggregation of image features per query
        # img_feats: (B, N, HW, d)  cam_weights: (B, Q, N)
        # → (B, Q, HW, d)
        img_agg = torch.einsum("bqn,bnhd->bqhd", cam_weights, img_feats)

        # Flatten batch × queries for multi-head attention
        img_flat = img_agg.view(B * Q, HW, d)
        q_flat   = (queries + query_pos).view(B * Q, 1, d)

        ca_out, _ = self.cross_attn(query=q_flat, key=img_flat, value=img_flat)
        ca_out = ca_out.view(B, Q, d)
        queries = self.norm_ca(queries + self.dropout(ca_out))

        # Self-attention among queries
        q = k = queries + query_pos
        sa_out, _ = self.self_attn(query=q, key=k, value=queries)
        queries = self.norm_sa(queries + self.dropout(sa_out))

        return self.ffn(queries)


# ---------------------------------------------------------------------------
# Prediction heads
# ---------------------------------------------------------------------------

class SeparateHead(nn.Module):
    """
    Separate prediction heads for each output (class, xyz, wlh, yaw, velocity).

    """

    def __init__(
        self,
        in_channels: int,
        heads: Dict[str, Tuple[int, int]],  # name → (out_dim, num_conv)
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleDict()
        for name, (out_dim, num_conv) in heads.items():
            layers: List[nn.Module] = []
            c = in_channels
            for i in range(num_conv - 1):
                layers += [nn.Linear(c, c), nn.ReLU(inplace=True)]
            layers.append(nn.Linear(c, out_dim))
            self.heads[name] = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for name, head in self.heads.items():
            if "heatmap" in name or "cls" in name:
                # Focal-loss initialisation
                bias_init = -math.log((1 - 0.01) / 0.01)
                nn.init.constant_(head[-1].bias, bias_init)
            else:
                nn.init.normal_(head[-1].weight, std=0.001)
                nn.init.zeros_(head[-1].bias)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        return {name: head(x) for name, head in self.heads.items()}


# ---------------------------------------------------------------------------
# Main TransFusion head
# ---------------------------------------------------------------------------

class TransFusionHead(nn.Module):
    """
    Full TransFusion detection head.

    Architecture
    ============
    1. BEV projection (1×1 conv → d_model)
    2. Heatmap prediction → top-K spatial positions seed object queries
    3. Stage-1 LiDAR-only transformer decoder (N_L layers)
    4. Stage-1 box prediction (auxiliary loss target)
    5. Stage-2 image cross-attention fusion decoder (N_I layers)
    6. Stage-2 final box + class prediction

    Box parameterisation (10 values per query)
    ------------------------------------------
    [x_norm, y_norm, z_norm, log_w, log_l, log_h, sin_yaw, cos_yaw, vx, vy]
    """

    BOX_CODE_SIZE = 10
    HEAD_SPEC = {
        "cls":    (10, 2),   # (out_dim, num_conv_layers)
        "center": (2, 2),    # x, y (normalised)
        "height": (1, 2),    # z
        "dim":    (3, 2),    # log w, l, h
        "rot":    (2, 2),    # sin, cos
        "vel":    (2, 2),    # vx, vy
    }

    def __init__(
        self,
        bev_feat_dim: int,
        img_feat_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_lidar_decoder_layers: int = 1,
        num_fusion_decoder_layers: int = 1,
        num_queries: int = 200,
        num_classes: int = 10,
        num_cameras: int = 6,
        bev_h: int = 128,
        bev_w: int = 128,
        img_feat_h: int = 14,
        img_feat_w: int = 25,
        pc_range: Tuple[float, ...] = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        dropout: float = 0.1,
        fusion_mode: str = "full",   # "full" | "cls_only" | "dual_stream"
    ) -> None:
        super().__init__()
        assert fusion_mode in ("full", "cls_only", "dual_stream"), \
            f"unknown fusion_mode: {fusion_mode}"
        self.fusion_mode  = fusion_mode
        self.num_queries  = num_queries
        self.num_classes  = num_classes
        self.d_model      = d_model
        self.bev_h        = bev_h
        self.bev_w        = bev_w
        self.pc_range     = pc_range

        # --- BEV projection ---
        self.bev_proj = nn.Sequential(
            ConvBnAct(bev_feat_dim, d_model, k=3, s=1, p=1),
            nn.Conv2d(d_model, d_model, 1),
        )

        # --- Heatmap (for query seeding) ---
        self.heatmap_head = nn.Sequential(
            ConvBnAct(d_model, d_model, k=3, s=1, p=1),
            nn.Conv2d(d_model, num_classes, 1),
        )

        # --- Learnable query content embeddings ---
        self.query_feat = nn.Embedding(num_queries, d_model)

        # --- BEV positional embeddings (fixed sinusoidal grid) ---
        self.register_buffer(
            "bev_pos_embed",
            self._build_bev_pos(bev_h, bev_w, d_model),
        )

        # --- Stage 1: LiDAR-only decoder ---
        self.lidar_decoder = nn.ModuleList([
            LiDARDecoderLayer(d_model, nhead, d_model * 4, dropout)
            for _ in range(num_lidar_decoder_layers)
        ])

        # --- Stage-1 auxiliary heads ---
        self.aux_heads = SeparateHead(d_model, self.HEAD_SPEC)

        # --- Image feature projection ---
        self.img_proj = nn.Sequential(
            nn.Linear(img_feat_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # --- Stage 2: LiDAR-Camera fusion decoder ---
        self.fusion_decoder = nn.ModuleList([
            ImageFusionLayer(d_model, nhead, d_model * 4, num_cameras, dropout,
                             img_h=img_feat_h, img_w=img_feat_w)
            for _ in range(num_fusion_decoder_layers)
        ])

        # Dual-stream mode keeps a SEPARATE LiDAR stream that continues to
        # refine via self-attention + FFN in parallel with the camera stream,
        # so the two modalities interact (through shared query positions /
        # predicted centres) without ever merging into one feature. Only built
        # when needed to avoid unused parameters in the other two modes.
        if self.fusion_mode == "dual_stream":
            self.lidar_stream = nn.ModuleList([
                LiDARDecoderLayer(d_model, nhead, d_model * 4, dropout)
                for _ in range(num_fusion_decoder_layers)
            ])
        else:
            self.lidar_stream = None

        # --- Stage-2 final heads ---
        # For the controlled fusion-philosophy comparison we keep separate
        # class and box heads so each can read from a different feature source
        # depending on `fusion_mode`:
        #   full        : both heads read the fused (LiDAR+camera) query
        #   cls_only    : class head reads fused query; box head reads the
        #                 LiDAR-only (stage-1) query  -> DAL-style
        #   dual_stream : two maintained streams; class reads camera-attended,
        #                 box reads LiDAR-attended    -> DeepInteraction-style
        self.final_cls_head = SeparateHead(d_model, {"cls": self.HEAD_SPEC["cls"]})
        self.final_box_head = SeparateHead(d_model, {
            k: v for k, v in self.HEAD_SPEC.items() if k != "cls"
        })
        # NOTE: no combined `final_heads` module — every parameter constructed
        # here must participate in the forward pass, because train.py wraps the
        # model in DDP with find_unused_parameters=False (unused params crash
        # the first backward). The split heads above are used in all three
        # fusion modes.

        self._init_weights()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_bev_pos(H: int, W: int, d_model: int) -> Tensor:
        ys = torch.linspace(0, 1, H)
        xs = torch.linspace(0, 1, W)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        pos2d = torch.stack([gx, gy], dim=-1).view(-1, 2)    # (HW, 2)
        return pos2posemb2d(pos2d, d_model // 2).unsqueeze(0)  # (1, HW, d)

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.query_feat.weight)
        # Focal-loss bias for heatmap
        bias_val = -math.log((1 - 0.01) / 0.01)
        nn.init.constant_(self.heatmap_head[-1].bias, bias_val)
        for m in self.bev_proj.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------ #
    def _query_pos_from_heatmap(self, heatmap: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Sample top-K positions from the multi-class heatmap.

        Returns
        -------
        query_pos_emb : (B, K, d_model) – sinusoidal positional embedding
        top_scores    : (B, K)
        top_cls       : (B, K) int64 – predicted class index for each query seed
        """
        B, C, H, W = heatmap.shape
        # Take max over classes first (class-agnostic peaks), then top-K
        heat_max = heatmap.max(dim=1).values                 # (B, H, W)
        flat = heat_max.view(B, -1)                           # (B, HW)
        scores, idx = flat.topk(self.num_queries, dim=-1)    # (B, K)

        top_cls = heatmap.view(B, C, -1).permute(0, 2, 1)   # (B, HW, C)
        top_cls = top_cls[torch.arange(B)[:, None], idx].argmax(-1)  # (B, K)

        hy = (idx // W).float() / H                          # (B, K)  normalised
        hx = (idx %  W).float() / W

        pos2d = torch.stack([hx, hy], dim=-1)                # (B, K, 2)
        pos_emb = pos2posemb2d(pos2d, self.d_model // 2)     # (B, K, d)
        return pos_emb, scores, top_cls

    # ------------------------------------------------------------------ #
    def _assemble_box(self, head_out: Dict[str, Tensor]) -> Tensor:
        """Concatenate per-attribute predictions into (B, Q, 10) box tensor."""
        return torch.cat([
            head_out["center"],   # (B,Q,2)  x,y
            head_out["height"],   # (B,Q,1)  z
            head_out["dim"],      # (B,Q,3)  log w,l,h
            head_out["rot"],      # (B,Q,2)  sin,cos
            head_out["vel"],      # (B,Q,2)  vx,vy
        ], dim=-1)

    # ------------------------------------------------------------------ #
    def forward(
        self,
        bev_feat: Tensor,                          # (B, C_bev, H, W)
        img_feats: List[Tensor],                   # list[(B*N, C_img, h, w)]
        num_cameras: int,
        lidar2img: Optional[Tensor] = None,        # (B, N, 4, 4)
    ) -> Dict[str, Tensor]:
        B = bev_feat.shape[0]

        # ---- 1. Project BEV features ----
        bev = self.bev_proj(bev_feat)              # (B, d, H, W)
        H, W = bev.shape[-2:]
        if (H, W) != (self.bev_h, self.bev_w):
            raise ValueError(
                f"BEV feature map is {H}x{W} but the head was constructed with "
                f"bev_h={self.bev_h}, bev_w={self.bev_w}. These must match — "
                f"bev_h/bev_w should equal (pc_range span / voxel_size / "
                f"out_size_factor). With the default config that is 64x64. "
                f"Fix the bev_h/bev_w you pass to TransFusion(...)."
            )
        bev_flat = bev.flatten(2).permute(0, 2, 1) # (B, HW, d)
        bev_pos  = self.bev_pos_embed.expand(B, -1, -1)

        # ---- 2. Heatmap ----
        heatmap = self.heatmap_head(bev)           # (B, C, H, W)
        query_pos_emb, _, _ = self._query_pos_from_heatmap(heatmap.detach())

        # ---- 3. Initialise queries ----
        queries = self.query_feat.weight.unsqueeze(0).expand(B, -1, -1)  # (B, Q, d)

        # ---- 4. Stage-1 LiDAR decoder ----
        for layer in self.lidar_decoder:
            queries = layer(queries, bev_flat, query_pos_emb, bev_pos)

        aux_out  = self.aux_heads(queries)
        pred_boxes_s1 = self._assemble_box(aux_out)   # (B, Q, 10)
        pred_logits_s1 = aux_out["cls"]                # (B, Q, C)

        # ---- 5. Project image features ----
        # Use the finest-scale FPN level (index 0 = P3, stride 8)
        img_f = img_feats[0]              # (B*N, C_img, h, w)
        BN, C_img, ih, iw = img_f.shape
        N = num_cameras
        img_f = img_f.view(B, N, C_img, ih * iw).permute(0, 1, 3, 2)  # (B, N, HW, C)
        img_f = self.img_proj(img_f)                                   # (B, N, HW, d)

        # Decode normalised (x_n, y_n, z_n) → metric (x, y, z) in the LiDAR
        # frame BEFORE projecting onto cameras. The projection matrix lidar2img
        # expects metres, so feeding [0,1] normalised values would project to
        # garbage pixels and disable the visibility gate. (Fix for the
        # coordinate-frame mismatch between the box head output and lidar2img.)
        pr = self.pc_range
        norm_centers = pred_boxes_s1[..., :3].detach()      # (B, Q, 3) in [0,1]
        metric_centers = torch.empty_like(norm_centers)
        metric_centers[..., 0] = norm_centers[..., 0] * (pr[3] - pr[0]) + pr[0]
        metric_centers[..., 1] = norm_centers[..., 1] * (pr[4] - pr[1]) + pr[1]
        metric_centers[..., 2] = norm_centers[..., 2] * (pr[5] - pr[2]) + pr[2]

        # Snapshot the LiDAR-only query (output of stage 1) BEFORE any image
        # information is mixed in. This is what the box head reads in cls_only
        # and dual_stream modes.
        lidar_query = queries.clone()

        # ---- 6. Stage-2 fusion decoder ----
        for layer in self.fusion_decoder:
            queries = layer(queries, img_f, query_pos_emb, metric_centers,
                            lidar2img, norm_centers)
        fused_query = queries   # LiDAR + camera

        # ---- 7. Predict according to fusion philosophy ----
        if self.fusion_mode == "full":
            # Baseline TransFusion: both heads read the fused query.
            cls_src, box_src = fused_query, fused_query

        elif self.fusion_mode == "cls_only":
            # DAL-style: camera informs classification only; box geometry comes
            # from the LiDAR-only representation (camera forbidden in regression).
            cls_src, box_src = fused_query, lidar_query

        elif self.fusion_mode == "dual_stream":
            # DeepInteraction-style: two maintained streams. The LiDAR stream
            # keeps refining on the BEV map in parallel (never sees the camera),
            # the camera stream is the fused query. Classification reads the
            # camera-attended stream; regression reads the refined LiDAR stream.
            lidar_stream_q = lidar_query
            for layer in self.lidar_stream:
                lidar_stream_q = layer(lidar_stream_q, bev_flat,
                                       query_pos_emb, bev_pos)
            cls_src, box_src = fused_query, lidar_stream_q
        else:
            cls_src, box_src = fused_query, fused_query

        pred_logits = self.final_cls_head(cls_src)["cls"]   # (B, Q, C)
        box_out     = self.final_box_head(box_src)          # dict of box parts
        pred_boxes  = self._assemble_box(box_out)           # (B, Q, 10)

        return {
            "heatmap":        heatmap,
            "pred_logits":    pred_logits,
            "pred_boxes":     pred_boxes,
            "pred_logits_s1": pred_logits_s1,
            "pred_boxes_s1":  pred_boxes_s1,
            "fusion_mode":    self.fusion_mode,
        }
