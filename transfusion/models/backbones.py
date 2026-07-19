"""
Production backbones for TransFusion.

LiDAR path:  PointPillar voxelisation → SECOND-style 2-D backbone → BEV feature map
Image path:  ResNet-50 + FPN
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..utils.common import ConvBnAct


# ---------------------------------------------------------------------------
# LiDAR: PointPillars pillar feature network
# ---------------------------------------------------------------------------

class PillarFeatureNet(nn.Module):
    """
    PointPillars pillar encoder.

    Input:  voxels    (M, P, C_in)       – padded pillar points
            num_points(M,)               – actual point count per pillar
            coords    (M, 4) int         – [batch_idx, z, y, x]
    Output: BEV pseudo-image (B, C_out, ny, nx)

    FIX (bug #7): pfn_in now computed as in_channels + in_channels + 2
        = in_channels raw features
        + in_channels offset-from-mean features   (same dim as raw)
        + 2 offset-from-pillar-centre (xy only)
    This is always correct regardless of in_channels.
    """

    def __init__(
        self,
        in_channels: int = 4,
        feat_channels: Tuple[int, ...] = (64,),
        voxel_size: Tuple[float, ...] = (0.2, 0.2, 8.0),
        pc_range: Tuple[float, ...] = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
    ) -> None:
        super().__init__()
        self.voxel_size = voxel_size
        self.pc_range   = pc_range

        # raw(C) + offset_from_mean(C) + offset_from_centre_xy(2)
        pfn_in = in_channels * 2 + 2
        pfn_layers: List[nn.Module] = []
        for c_out in feat_channels:
            pfn_layers += [
                nn.Linear(pfn_in, c_out, bias=False),
                nn.BatchNorm1d(c_out, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
            ]
            pfn_in = c_out
        self.pfn = nn.Sequential(*pfn_layers)
        self.out_channels = feat_channels[-1]

        self.nx = int(round((pc_range[3] - pc_range[0]) / voxel_size[0]))
        self.ny = int(round((pc_range[4] - pc_range[1]) / voxel_size[1]))

    def forward(
        self,
        voxels: Tensor,      # (M, P, C_in)
        num_points: Tensor,  # (M,)
        coords: Tensor,      # (M, 4) [batch_idx, z, y, x]
    ) -> Tensor:
        M, P, C = voxels.shape

        # Mask padding points
        mask   = torch.arange(P, device=voxels.device)[None] < num_points[:, None]   # (M, P)
        mask_f = mask.float().unsqueeze(-1)                                            # (M, P, 1)

        # Mean of valid points per pillar
        mean = (voxels * mask_f).sum(1) / num_points.float().clamp(min=1).unsqueeze(-1)  # (M, C)
        offset_mean = voxels - mean.unsqueeze(1)                                           # (M, P, C)

        # Offset from pillar centre (xy in metric space)
        # coords: [batch_idx, z, y, x]  → x=[:,3], y=[:,2]
        cx = (coords[:, 3].float() + 0.5) * self.voxel_size[0] + self.pc_range[0]
        cy = (coords[:, 2].float() + 0.5) * self.voxel_size[1] + self.pc_range[1]
        centre = torch.stack([cx, cy], dim=-1)                      # (M, 2)
        offset_centre = voxels[..., :2] - centre.unsqueeze(1)       # (M, P, 2)

        feat = torch.cat([voxels, offset_mean, offset_centre], dim=-1)  # (M, P, 2C+2)
        feat = feat.view(M * P, -1)
        feat = self.pfn(feat).view(M, P, -1)
        feat = (feat * mask_f).max(dim=1).values                    # (M, C_out)

        # Scatter to BEV canvas
        batch_size = int(coords[:, 0].max().item()) + 1
        canvas = voxels.new_zeros(batch_size, self.out_channels, self.ny, self.nx)
        b_idx  = coords[:, 0].long()
        y_idx  = coords[:, 2].long().clamp(0, self.ny - 1)
        x_idx  = coords[:, 3].long().clamp(0, self.nx - 1)
        canvas[b_idx, :, y_idx, x_idx] = feat
        return canvas


# ---------------------------------------------------------------------------
# LiDAR: SECOND-style 2-D backbone
# ---------------------------------------------------------------------------

class SECONDBackbone(nn.Module):
    """
    SECOND-style 2-D feature extractor on the BEV pseudo-image.
    Produces a single fused BEV feature map via FPN-style up-sampling.
    """

    def __init__(
        self,
        in_channels: int = 64,
        layer_nums: Tuple[int, ...] = (5, 5),
        layer_strides: Tuple[int, ...] = (1, 2),
        num_filters: Tuple[int, ...] = (128, 256),
        upsample_strides: Tuple[int, ...] = (1, 2),
        num_upsample_filters: Tuple[int, ...] = (256, 256),
        out_stride: int = 8,
    ) -> None:
        super().__init__()
        # Downsampling stem: reduces the pillar canvas (e.g. 512x512) by
        # `out_stride` so the final BEV feature map matches the head's
        # expected (bev_h, bev_w). Each strided conv halves resolution, so
        # we need log2(out_stride) stem layers. The two residual blocks below
        # then run at this reduced resolution and the multi-scale fusion brings
        # everything back to the stem-output resolution.
        import math as _math
        n_stem = max(0, int(round(_math.log2(out_stride))))
        stem_layers: List[nn.Module] = []
        c = in_channels
        stem_out = num_filters[0]
        for _ in range(n_stem):
            stem_layers.append(ConvBnAct(c, stem_out, k=3, s=2, p=1))
            c = stem_out
        self.stem = nn.Sequential(*stem_layers) if stem_layers else nn.Identity()
        stem_c = c if n_stem > 0 else in_channels

        self.blocks   = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        c_in = stem_c

        for i in range(len(layer_nums)):
            block_layers: List[nn.Module] = [
                ConvBnAct(c_in, num_filters[i], k=3, s=layer_strides[i], p=1)
            ]
            for _ in range(layer_nums[i] - 1):
                block_layers.append(ConvBnAct(num_filters[i], num_filters[i], k=3, s=1, p=1))
            self.blocks.append(nn.Sequential(*block_layers))
            c_in = num_filters[i]

            us = upsample_strides[i]
            uf = num_upsample_filters[i]
            if us == 1:
                self.deblocks.append(ConvBnAct(num_filters[i], uf, k=3, s=1, p=1))
            else:
                self.deblocks.append(nn.Sequential(
                    nn.ConvTranspose2d(num_filters[i], uf, us, stride=us, bias=False),
                    nn.BatchNorm2d(uf, eps=1e-3, momentum=0.01),
                    nn.ReLU(inplace=True),
                ))

        self.out_channels = sum(num_upsample_filters)

    def forward(self, bev: Tensor) -> Tensor:
        x = self.stem(bev)          # downsample by out_stride
        outs = []
        for block, deblock in zip(self.blocks, self.deblocks):
            x = block(x)
            outs.append(deblock(x))

        target_h, target_w = outs[0].shape[-2:]
        aligned = [outs[0]]
        for feat in outs[1:]:
            if feat.shape[-2:] != (target_h, target_w):
                feat = F.interpolate(feat, size=(target_h, target_w),
                                     mode="bilinear", align_corners=False)
            aligned.append(feat)
        return torch.cat(aligned, dim=1)


# ---------------------------------------------------------------------------
# Image: ResNet-50 backbone
# ---------------------------------------------------------------------------

class ResNetBottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_c: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.conv1     = ConvBnAct(in_c, planes, k=1, s=1, p=0)
        self.conv2     = ConvBnAct(planes, planes, k=3, s=stride, p=1)
        self.conv3     = ConvBnAct(planes, planes * self.expansion, k=1, s=1, p=0, act=False)
        self.relu      = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        return self.relu(self.conv3(self.conv2(self.conv1(x))) + identity)


class ResNet50(nn.Module):
    """
    ResNet-50 returning (C3, C4, C5) feature maps at stride 8/16/32.
    Optionally loads torchvision pretrained weights.
    """

    def __init__(self, pretrained: bool = True, frozen_stages: int = 1) -> None:
        super().__init__()
        self.frozen_stages = frozen_stages

        self.stem   = nn.Sequential(
            ConvBnAct(3, 64, k=7, s=2, p=3),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64,   64,  3, stride=1)
        self.layer2 = self._make_layer(256,  128, 4, stride=2)
        self.layer3 = self._make_layer(512,  256, 6, stride=2)
        self.layer4 = self._make_layer(1024, 512, 3, stride=2)

        if pretrained:
            self._load_pretrained()
        self._freeze_stages()

    @staticmethod
    def _make_layer(in_c: int, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        exp = ResNetBottleneck.expansion
        ds  = None
        if stride != 1 or in_c != planes * exp:
            ds = nn.Sequential(
                nn.Conv2d(in_c, planes * exp, 1, stride, bias=False),
                nn.BatchNorm2d(planes * exp),
            )
        layers = [ResNetBottleneck(in_c, planes, stride, ds)]
        for _ in range(1, blocks):
            layers.append(ResNetBottleneck(planes * exp, planes))
        return nn.Sequential(*layers)

    def _load_pretrained(self) -> None:
        try:
            import torchvision.models as tvm
            tv_model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
            # Map torchvision layer names → ours
            tv_state = tv_model.state_dict()
            # Our stem.0.block.{0,1} = torchvision conv1, bn1
            new_state = {}
            for k, v in tv_state.items():
                if k.startswith("conv1."):
                    new_state[k.replace("conv1.", "stem.0.block.0.", 1)] = v
                elif k.startswith("bn1."):
                    new_state[k.replace("bn1.", "stem.0.block.1.", 1)] = v
                elif k.startswith(("layer1.", "layer2.", "layer3.", "layer4.")):
                    new_state[k] = v
            missing, unexpected = self.load_state_dict(new_state, strict=False)
            import logging
            logging.getLogger(__name__).info(
                "Pretrained ResNet-50 loaded. Missing: %d, Unexpected: %d",
                len(missing), len(unexpected),
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("Could not load pretrained ResNet-50: %s", e)

    def _freeze_stages(self) -> None:
        if self.frozen_stages >= 0:
            self.stem.eval()
            for p in self.stem.parameters():
                p.requires_grad_(False)
        for i in range(1, self.frozen_stages + 1):
            layer = getattr(self, f"layer{i}")
            layer.eval()
            for p in layer.parameters():
                p.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_stages()
        return self

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        x  = self.stem(x)    # stride 4
        x  = self.layer1(x)  # stride 4
        c3 = self.layer2(x)  # stride 8
        c4 = self.layer3(c3) # stride 16
        c5 = self.layer4(c4) # stride 32
        return c3, c4, c5


# ---------------------------------------------------------------------------
# FPN neck
# ---------------------------------------------------------------------------

class FPN(nn.Module):
    def __init__(
        self,
        in_channels: Tuple[int, ...] = (512, 1024, 2048),
        out_channels: int = 256,
        num_outs: int = 3,
    ) -> None:
        super().__init__()
        self.num_outs = num_outs
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(c, out_channels, 1) for c in in_channels]
        )
        self.fpn_convs = nn.ModuleList(
            [nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels]
        )
        self.out_channels = out_channels
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features: Tuple[Tensor, Tensor, Tensor]) -> List[Tensor]:
        laterals = [l(f) for l, f in zip(self.lateral_convs, features)]
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], scale_factor=2, mode="nearest"
            )
        outs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]
        return outs[: self.num_outs]


class ImageBackbone(nn.Module):
    """ResNet-50 + FPN returning multi-scale feature list."""

    def __init__(
        self,
        pretrained: bool = True,
        frozen_stages: int = 1,
        fpn_out_channels: int = 256,
        fpn_num_outs: int = 3,
    ) -> None:
        super().__init__()
        self.backbone    = ResNet50(pretrained=pretrained, frozen_stages=frozen_stages)
        self.neck        = FPN(
            in_channels=(512, 1024, 2048),
            out_channels=fpn_out_channels,
            num_outs=fpn_num_outs,
        )
        self.out_channels = fpn_out_channels

    def forward(self, imgs: Tensor) -> List[Tensor]:
        """imgs: (B*N, 3, H, W) → list of (B*N, C, h, w) per FPN level."""
        c3, c4, c5 = self.backbone(imgs)
        return self.neck((c3, c4, c5))
