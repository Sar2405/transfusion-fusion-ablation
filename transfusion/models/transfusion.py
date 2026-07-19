"""
TransFusion: end-to-end model combining LiDAR + camera backbones with the
TransFusion detection head.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .backbones import PillarFeatureNet, SECONDBackbone, ImageBackbone
from .head import TransFusionHead


class TransFusion(nn.Module):
    """
    End-to-end TransFusion 3-D object detector.

    Input modes
    -----------
    The model supports two LiDAR input modes controlled by ``use_pillar_net``:

    * ``use_pillar_net=True``  (default / production):
        Accepts raw voxelised tensors – ``voxels``, ``num_points``, ``coords``.
        PillarFeatureNet → SECONDBackbone → BEV feature map.

    * ``use_pillar_net=False`` (debugging / unit tests):
        Accepts a pre-computed BEV pseudo-image tensor ``bev_input``
        of shape (B, bev_in_channels, H, W) and runs it directly
        through SECONDBackbone.

    Args
    ----
    bev_in_channels         : pillar feature channels (PointPillars output)
    num_cameras             : number of surround-view cameras
    num_classes             : detection categories
    num_queries             : number of object queries
    d_model                 : transformer hidden dimension
    nhead                   : number of attention heads
    num_lidar_decoder_layers: stage-1 decoder depth
    num_fusion_decoder_layers: stage-2 decoder depth
    dropout                 : transformer dropout
    bev_h, bev_w            : BEV feature map spatial dims (after backbone stride)
    img_feat_h, img_feat_w  : image FPN feature spatial dims (stride-8 level)
    pc_range                : [xmin,ymin,zmin,xmax,ymax,zmax]
    voxel_size              : [dx,dy,dz]
    fpn_out_channels        : FPN output channels
    pretrained_img          : load torchvision ResNet-50 pretrained weights
    frozen_img_stages       : number of ResNet stages to freeze (0 = none)
    use_pillar_net          : if False, accept BEV pseudo-image directly
    """

    def __init__(
        self,
        bev_in_channels: int            = 64,
        num_cameras: int                = 6,
        num_classes: int                = 10,
        num_queries: int                = 200,
        d_model: int                    = 256,
        nhead: int                      = 8,
        num_lidar_decoder_layers: int   = 1,
        num_fusion_decoder_layers: int  = 1,
        dropout: float                  = 0.1,
        bev_h: int                      = 64,
        bev_w: int                      = 64,
        img_feat_h: int                 = 14,
        img_feat_w: int                 = 25,
        pc_range: Tuple[float, ...]     = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        voxel_size: Tuple[float, ...]   = (0.2, 0.2, 8.0),
        out_size_factor: int            = 8,
        fpn_out_channels: int           = 256,
        pretrained_img: bool            = True,
        frozen_img_stages: int          = 1,
        use_pillar_net: bool            = True,
        point_feat_channels: int        = 4,   # raw point dims: 4 (single-sweep) or 5 (+Δt)
        fusion_mode: str                = "full",  # "full" | "cls_only" | "dual_stream"
    ) -> None:
        super().__init__()
        self.num_cameras = num_cameras
        self.bev_h       = bev_h
        self.bev_w       = bev_w
        self.use_pillar_net = use_pillar_net

        # --- LiDAR pipeline ---
        if use_pillar_net:
            self.pillar_net = PillarFeatureNet(
                in_channels=point_feat_channels,
                feat_channels=(bev_in_channels,),
                voxel_size=voxel_size,
                pc_range=pc_range,
            )
            lidar_in = bev_in_channels
        else:
            self.pillar_net = None
            lidar_in = bev_in_channels

        self.lidar_backbone = SECONDBackbone(
            in_channels=lidar_in,
            layer_nums=(5, 5),
            layer_strides=(1, 2),
            num_filters=(128, 256),
            upsample_strides=(1, 2),
            num_upsample_filters=(256, 256),
            out_stride=out_size_factor,
        )

        # --- Image pipeline ---
        self.image_backbone = ImageBackbone(
            pretrained=pretrained_img,
            frozen_stages=frozen_img_stages,
            fpn_out_channels=fpn_out_channels,
        )

        # --- TransFusion head ---
        self.head = TransFusionHead(
            bev_feat_dim=self.lidar_backbone.out_channels,
            img_feat_dim=fpn_out_channels,
            d_model=d_model,
            nhead=nhead,
            num_lidar_decoder_layers=num_lidar_decoder_layers,
            num_fusion_decoder_layers=num_fusion_decoder_layers,
            num_queries=num_queries,
            num_classes=num_classes,
            num_cameras=num_cameras,
            bev_h=bev_h,
            bev_w=bev_w,
            img_feat_h=img_feat_h,
            img_feat_w=img_feat_w,
            pc_range=pc_range,
            dropout=dropout,
            fusion_mode=fusion_mode,
        )

    # ------------------------------------------------------------------ #
    def extract_lidar_feat(
        self,
        bev_input: Optional[Tensor] = None,
        voxels: Optional[Tensor] = None,
        num_points: Optional[Tensor] = None,
        coords: Optional[Tensor] = None,
    ) -> Tensor:
        if self.use_pillar_net:
            assert voxels is not None and num_points is not None and coords is not None
            bev_canvas = self.pillar_net(voxels, num_points, coords)
        else:
            assert bev_input is not None
            bev_canvas = bev_input
        return self.lidar_backbone(bev_canvas)

    def extract_image_feat(self, camera_imgs: Tensor) -> List[Tensor]:
        """
        Args:
            camera_imgs: (B, N, 3, H, W)
        Returns:
            list of multi-scale features [(B*N, C, h, w), ...]
        """
        B, N, C, H, W = camera_imgs.shape
        imgs_flat = camera_imgs.view(B * N, C, H, W)
        return self.image_backbone(imgs_flat)

    # ------------------------------------------------------------------ #
    def forward(
        self,
        camera_imgs: Tensor,                       # (B, N, 3, H, W)
        bev_input: Optional[Tensor] = None,        # (B, C, H0, W0)  if use_pillar_net=False
        voxels: Optional[Tensor] = None,           # (M, P, C_in)
        num_points: Optional[Tensor] = None,       # (M,)
        coords: Optional[Tensor] = None,           # (M, 4)
        lidar2img: Optional[Tensor] = None,        # (B, N, 4, 4)
    ) -> Dict[str, Tensor]:
        """
        Forward pass.

        Returns
        -------
        Dict with keys:
            heatmap        (B, num_classes, H_bev, W_bev)
            pred_logits    (B, Q, num_classes)
            pred_boxes     (B, Q, 10)
            pred_logits_s1 (B, Q, num_classes)   [stage-1 auxiliary]
            pred_boxes_s1  (B, Q, 10)            [stage-1 auxiliary]
        """
        bev_feat  = self.extract_lidar_feat(bev_input, voxels, num_points, coords)
        img_feats = self.extract_image_feat(camera_imgs)

        return self.head(
            bev_feat=bev_feat,
            img_feats=img_feats,
            num_cameras=self.num_cameras,
            lidar2img=lidar2img,
        )

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict(
        self,
        camera_imgs: Tensor,
        score_threshold: float = 0.1,
        nms_iou_threshold: float = 0.2,
        max_per_scene: int = 500,
        **forward_kwargs,
    ) -> List[Dict[str, Tensor]]:
        """
        Run inference and return decoded, NMS-filtered detections.

        Returns a list (one per batch item) of dicts:
            scores   (K,)
            labels   (K,) int
            boxes    (K, 10)  decoded
        """
        from ..utils.common import nms_bev, decode_boxes

        self.eval()
        outputs = self.forward(camera_imgs, **forward_kwargs)

        pred_logits = outputs["pred_logits"]   # (B, Q, C)
        pred_boxes  = outputs["pred_boxes"]    # (B, Q, 10)

        B = pred_logits.shape[0]
        results = []

        for b in range(B):
            scores_all, labels = pred_logits[b].sigmoid().max(dim=-1)  # (Q,), (Q,)
            mask = scores_all > score_threshold

            scores = scores_all[mask]
            labs   = labels[mask]
            boxes  = pred_boxes[b][mask]

            # Decode box coordinates
            boxes_dec = decode_boxes(boxes, list(self.head.pc_range))

            # BEV NMS (approximate axis-aligned)
            keep = nms_bev(boxes_dec, scores, iou_threshold=nms_iou_threshold)
            keep = keep[:max_per_scene]

            results.append({
                "scores": scores[keep],
                "labels": labs[keep],
                "boxes":  boxes_dec[keep],
            })

        return results
