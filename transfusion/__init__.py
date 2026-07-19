"""TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with Transformers."""
from .models.transfusion import TransFusion
from .models.loss import TransFusionLoss

__version__ = "1.0.0"
__all__ = ["TransFusion", "TransFusionLoss"]
