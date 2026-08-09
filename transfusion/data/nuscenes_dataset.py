"""
nuScenes dataset for TransFusion.

Notes on frame conventions
---------------------------
Points and GT boxes are kept in the LiDAR sensor frame (correct for
voxelisation). The lidar→image projection chain is:
    K @ T_cam_sensor⁻¹ @ T_cam_ego⁻¹ @ T_lidar_ego @ T_lidar_sensor

nuScenes annotation size is [width, length, height]; we store dimensions in
the paper-standard [l, w, h] order (CenterPoint convention).

hard_voxelise keys and outputs coords consistently as (z, y, x), so
PillarFeatureNet receives [batch, z, y, x] after collate_fn prepends the
batch index.
"""
from __future__ import annotations

import os
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Hard voxeliser
# ---------------------------------------------------------------------------

def hard_voxelise(
    points: np.ndarray,          # (N, C)  first 3 cols = x, y, z
    voxel_size: np.ndarray,      # (3,)    dx, dy, dz
    coors_range: np.ndarray,     # (6,)    xmin,ymin,zmin,xmax,ymax,zmax
    max_points: int = 10,
    max_voxels: int = 30_000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    voxels     : (M, max_points, C)   padded pillar tensors
    coords     : (M, 3)  integer voxel indices in (z, y, x) order
    num_points : (M,)    valid point count per voxel
    """
    C = points.shape[1]
    grid_size = np.round(
        (coors_range[3:] - coors_range[:3]) / voxel_size
    ).astype(np.int32)   # [nx, ny, nz]

    # Range filter
    mask = (
        (points[:, 0] >= coors_range[0]) & (points[:, 0] < coors_range[3]) &
        (points[:, 1] >= coors_range[1]) & (points[:, 1] < coors_range[4]) &
        (points[:, 2] >= coors_range[2]) & (points[:, 2] < coors_range[5])
    )
    points = points[mask]

    # Voxel index per point: xi, yi, zi  (all in 0..grid-1)
    xi = np.floor((points[:, 0] - coors_range[0]) / voxel_size[0]).astype(np.int32)
    yi = np.floor((points[:, 1] - coors_range[1]) / voxel_size[1]).astype(np.int32)
    zi = np.floor((points[:, 2] - coors_range[2]) / voxel_size[2]).astype(np.int32)
    xi = np.clip(xi, 0, grid_size[0] - 1)
    yi = np.clip(yi, 0, grid_size[1] - 1)
    zi = np.clip(zi, 0, grid_size[2] - 1)

    # key and stored coord both use (z, y, x) consistently
    voxel_dict: Dict[Tuple[int,int,int], int] = {}
    voxels_list: List[List[np.ndarray]]       = []
    coords_list: List[np.ndarray]             = []

    for i in range(len(points)):
        key = (int(zi[i]), int(yi[i]), int(xi[i]))   # (z, y, x)
        if key not in voxel_dict:
            if len(voxel_dict) >= max_voxels:
                continue
            voxel_dict[key] = len(voxels_list)
            voxels_list.append([])
            coords_list.append(np.array(key, dtype=np.int32))  # stored as (z,y,x)
        idx = voxel_dict[key]
        if len(voxels_list[idx]) < max_points:
            voxels_list[idx].append(points[i])

    M = len(voxels_list)
    voxels     = np.zeros((M, max_points, C), dtype=np.float32)
    num_pts    = np.zeros(M, dtype=np.int32)
    coors_out  = np.zeros((M, 3), dtype=np.int32)

    for i, (pts, c) in enumerate(zip(voxels_list, coords_list)):
        n = len(pts)
        voxels[i, :n] = np.stack(pts)
        num_pts[i]    = n
        coors_out[i]  = c     # (z, y, x)

    return voxels, coors_out, num_pts


# ---------------------------------------------------------------------------
# Class mappings
# ---------------------------------------------------------------------------

NUSCENES_CLASS_NAMES = [
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]

# Maps nuScenes category name → our class name (prefix match)
NUSCENES_CATS: Dict[str, str] = {
    "vehicle.car":                            "car",
    "vehicle.truck":                          "truck",
    "vehicle.construction":                   "construction_vehicle",
    "vehicle.bus.bendy":                      "bus",
    "vehicle.bus.rigid":                      "bus",
    "vehicle.trailer":                        "trailer",
    "movable_object.barrier":                 "barrier",
    "vehicle.motorcycle":                     "motorcycle",
    "vehicle.bicycle":                        "bicycle",
    "human.pedestrian.adult":                 "pedestrian",
    "human.pedestrian.child":                 "pedestrian",
    "human.pedestrian.wheelchair":            "pedestrian",
    "human.pedestrian.stroller":              "pedestrian",
    "human.pedestrian.personal_mobility":     "pedestrian",
    "human.pedestrian.police_officer":        "pedestrian",
    "human.pedestrian.construction_worker":   "pedestrian",
    "movable_object.trafficcone":             "traffic_cone",
}

CAMERA_NAMES = [
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT",
    "CAM_BACK",  "CAM_BACK_LEFT",   "CAM_FRONT_LEFT",
]


# ---------------------------------------------------------------------------
# nuScenes Dataset
# ---------------------------------------------------------------------------

class NuScenesDataset(Dataset):
    """
    nuScenes 3-D object detection dataset.

    Args
    ----
    data_root  : nuScenes root directory (contains `samples/`, `v1.0-*/`)
    version    : 'v1.0-trainval' | 'v1.0-mini' | 'v1.0-test'
    split      : 'train' | 'val'
    pc_range   : [xmin, ymin, zmin, xmax, ymax, zmax]
    voxel_size : [dx, dy, dz]
    max_points : max points per voxel
    max_voxels : max voxels per sample
    img_size   : (H, W) to resize camera images to
    augment    : enable LiDAR + image augmentation
    """

    def __init__(
        self,
        data_root: str,
        version: str             = "v1.0-trainval",
        split: str               = "train",
        pc_range: List[float]    = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        voxel_size: List[float]  = [0.2, 0.2, 8.0],
        max_points: int          = 10,
        max_voxels: int          = 30_000,
        img_size: Tuple[int,int] = (448, 800),   # (H, W)
        img_resize_range: Tuple[float,float] = (0.95, 1.35),
        augment: bool            = False,
    ) -> None:
        super().__init__()
        from nuscenes.nuscenes import NuScenes
        from nuscenes.utils import splits as nu_splits

        self.nusc       = NuScenes(version=version, dataroot=data_root, verbose=False)
        self.pc_range   = np.array(pc_range,  dtype=np.float64)
        self.voxel_size = np.array(voxel_size, dtype=np.float64)
        self.max_points = max_points
        self.max_voxels = max_voxels
        self.img_size   = img_size
        # Random image-resize augmentation range (multiplier on the base scale).
        # DAL uses a wide range; too wide with a small backbone can hurt, so this
        # is configurable and should be gated. Set (1.0, 1.0) to disable.
        self.img_resize_range = img_resize_range
        self.augment    = augment
        self.class_names = NUSCENES_CLASS_NAMES
        self.cls2idx     = {c: i for i, c in enumerate(self.class_names)}

        # Scene names for the requested split
        split_scenes = set(nu_splits.train if split == "train" else nu_splits.val)
        candidates = [
            s for s in self.nusc.sample
            if self.nusc.get("scene", s["scene_token"])["name"] in split_scenes
        ]

        # ---- Partial-data filter -------------------------------------------
        # Partial copies of nuScenes (e.g. Kaggle re-uploads) ship the COMPLETE
        # metadata tables but only a fraction of the sensor blobs. The tables
        # therefore index samples whose files do not exist on disk, which would
        # crash the dataloader mid-epoch with FileNotFoundError. Here we keep
        # only samples whose LiDAR keyframe AND all camera keyframes are
        # actually present. On a complete dataset this drops nothing.
        cam_names = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT",
                     "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT"]
        self.samples = []
        dropped = 0
        for s in candidates:
            ok = True
            for sensor in ["LIDAR_TOP"] + cam_names:
                sd = self.nusc.get("sample_data", s["data"][sensor])
                if not os.path.exists(os.path.join(self.nusc.dataroot, sd["filename"])):
                    ok = False
                    break
            if ok:
                self.samples.append(s)
            else:
                dropped += 1
        import logging
        logging.getLogger(__name__).info(
            "NuScenesDataset[%s/%s]: kept %d samples, dropped %d with missing "
            "sensor files%s", version, split, len(self.samples), dropped,
            "" if dropped == 0 else " (partial dataset copy detected)")
        if len(self.samples) == 0:
            raise RuntimeError(
                f"No usable samples in split '{split}': all {len(candidates)} "
                f"indexed samples have missing sensor files under "
                f"{self.nusc.dataroot}. Check --data-root and that samples/ "
                f"contains the sensor blobs.")

    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _lidar_sensor_to_global(self, sample: dict):
        """
        Return 4×4 transforms:
            T_lidar_sensor : lidar sensor frame → ego body frame
            T_lidar_ego    : ego body frame → global frame
        (at the LiDAR timestamp)
        """
        from pyquaternion import Quaternion
        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_sd    = self.nusc.get("sample_data", lidar_token)
        lidar_cs    = self.nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
        lidar_ep    = self.nusc.get("ego_pose",           lidar_sd["ego_pose_token"])

        def to_mat(rotation, translation):
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = Quaternion(rotation).rotation_matrix
            T[:3,  3] = np.array(translation, dtype=np.float64)
            return T

        T_ls  = to_mat(lidar_cs["rotation"], lidar_cs["translation"])   # lidar_sensor→ego
        T_le  = to_mat(lidar_ep["rotation"], lidar_ep["translation"])   # ego→global
        return T_ls, T_le

    def _load_pointcloud(self, sample: dict) -> np.ndarray:
        """Return (N, 4) point cloud [x, y, z, intensity] in LiDAR sensor frame."""
        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_sd    = self.nusc.get("sample_data", lidar_token)
        path = os.path.join(self.nusc.dataroot, lidar_sd["filename"])
        pts  = np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :4]  # (N,4)
        return pts   # stays in LiDAR sensor frame for voxelisation

    def _load_images_and_proj(
        self, sample: dict, T_lidar_sensor: np.ndarray, T_lidar_ego: np.ndarray
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Load 6 camera images and compute lidar→image projection matrices.

        Projection chain (LiDAR sensor frame → image pixels):
            K @ T_cam_sensor⁻¹ @ T_cam_ego⁻¹ @ T_lidar_ego @ T_lidar_sensor

        where:
            T_lidar_sensor : lidar sensor → lidar ego body    (at LiDAR time)
            T_lidar_ego    : lidar ego body → global           (at LiDAR time)
            T_cam_ego      : cam ego body → global             (at cam time)
            T_cam_sensor   : cam sensor → cam ego body         (calibration)
        """
        import cv2
        from pyquaternion import Quaternion

        def to_mat(rotation, translation) -> np.ndarray:
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = Quaternion(rotation).rotation_matrix
            T[:3,  3] = np.array(translation, dtype=np.float64)
            return T

        imgs: List[np.ndarray]      = []
        lidar2img_list: List[np.ndarray] = []
        tH, tW = self.img_size

        for cam_name in CAMERA_NAMES:
            cam_token = sample["data"][cam_name]
            cam_sd    = self.nusc.get("sample_data", cam_token)
            cam_cs    = self.nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
            cam_ep    = self.nusc.get("ego_pose",           cam_sd["ego_pose_token"])

            # Load + resize image
            img_path = os.path.join(self.nusc.dataroot, cam_sd["filename"])
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Camera image not found: {img_path}")
            H, W = img.shape[:2]

            # ---- Random resize augmentation (DAL Tab.5: +3.91 mAP) ----------
            # Train: resize by a random factor around the base scale, then
            # random-crop/pad back to (tH, tW). Eval: deterministic base scale.
            # The intrinsics are scaled by the SAME factor and shifted by the
            # crop offset — otherwise lidar2img projects to the wrong pixels and
            # the camera fusion silently attends to unrelated image regions.
            base_sy, base_sx = tH / H, tW / W
            if self.augment:
                lo, hi = self.img_resize_range           # e.g. (0.95, 1.35)
                jitter = float(np.random.uniform(lo, hi))
            else:
                jitter = 1.0
            sy, sx = base_sy * jitter, base_sx * jitter
            rh, rw = max(int(round(H * sy)), 1), max(int(round(W * sx)), 1)
            img = cv2.resize(img, (rw, rh))

            # crop (if larger than target) or pad (if smaller) to exactly tH x tW
            if rh >= tH:
                oy = int(np.random.randint(0, rh - tH + 1)) if self.augment else (rh - tH) // 2
                img = img[oy:oy + tH]
            else:
                oy = -((tH - rh) // 2)
                pad = np.zeros((tH, img.shape[1], 3), dtype=img.dtype)
                pad[-oy:-oy + rh] = img
                img = pad
            if rw >= tW:
                ox = int(np.random.randint(0, rw - tW + 1)) if self.augment else (rw - tW) // 2
                img = img[:, ox:ox + tW]
            else:
                ox = -((tW - rw) // 2)
                pad = np.zeros((img.shape[0], tW, 3), dtype=img.dtype)
                pad[:, -ox:-ox + rw] = img
                img = pad
            imgs.append(img)

            # Intrinsics: scale by the applied factor, then shift by crop offset
            K = np.eye(4, dtype=np.float64)
            K[:3, :3] = np.array(cam_cs["camera_intrinsic"], dtype=np.float64)
            K[0, :] *= sx       # scale fx, cx
            K[1, :] *= sy       # scale fy, cy
            K[0, 2] -= ox       # shift principal point by horizontal crop
            K[1, 2] -= oy       # shift principal point by vertical crop

            # Extrinsics
            T_cam_sensor = to_mat(cam_cs["rotation"], cam_cs["translation"])  # cam_sensor→cam_ego
            T_cam_ego    = to_mat(cam_ep["rotation"], cam_ep["translation"])  # cam_ego→global

            # lidar_sensor → global: T_lidar_ego @ T_lidar_sensor
            # global → cam_ego:      inv(T_cam_ego)
            # cam_ego → cam_sensor:  inv(T_cam_sensor)
            # cam_sensor → pixel:    K
            lidar2img = (
                K
                @ np.linalg.inv(T_cam_sensor)
                @ np.linalg.inv(T_cam_ego)
                @ T_lidar_ego
                @ T_lidar_sensor
            )
            lidar2img_list.append(lidar2img.astype(np.float32))

        return np.stack(imgs), lidar2img_list   # imgs: (6, tH, tW, 3) BGR

    def _load_annotations(
        self, sample: dict, T_lidar_sensor: np.ndarray, T_lidar_ego: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load GT boxes (in global frame) and transform to LiDAR sensor frame.

        nuScenes size = [width, length, height].
        Box code = [x_norm, y_norm, z_norm, log_l, log_w, log_h, sin_yaw, cos_yaw, vx, vy]
        (l=longitudinal length, w=lateral width – matches CenterPoint/TransFusion convention)

        Returns
        -------
        boxes  : (G, 10) float32
        labels : (G,)    int64
        """
        from pyquaternion import Quaternion

        boxes_list, labels_list = [], []
        T_global_to_lidar = np.linalg.inv(T_lidar_ego @ T_lidar_sensor)  # global→LiDAR sensor

        for ann_token in sample["anns"]:
            ann = self.nusc.get("sample_annotation", ann_token)
            cat = ann["category_name"]

            # Category lookup (prefix match)
            cls = None
            for key, val in NUSCENES_CATS.items():
                if cat.startswith(key):
                    cls = val
                    break
            if cls is None:
                continue   # unknown / ignore category
            label = self.cls2idx[cls]

            # Box centre in global frame → LiDAR sensor frame
            xyz_global = np.array([*ann["translation"], 1.0], dtype=np.float64)
            xyz_lidar  = T_global_to_lidar @ xyz_global               # (4,)
            x, y, z    = xyz_lidar[:3]

            # nuScenes size = [width, length, height]
            width, length, height = ann["size"]                        # w, l, h (metric)

            # Yaw: global quaternion → LiDAR-frame yaw
            q_global   = Quaternion(ann["rotation"])
            q_lidar_cs = Quaternion(matrix=T_lidar_sensor[:3, :3])
            q_lidar_ep = Quaternion(matrix=T_lidar_ego[:3, :3])
            q_lidar    = (q_lidar_ep * q_lidar_cs).inverse * q_global
            yaw        = q_lidar.yaw_pitch_roll[0]

            # Velocity in LiDAR sensor frame (nuScenes gives global-frame vel)
            velo_global = self.nusc.box_velocity(ann_token)  # (3,) or (2,)
            if np.any(np.isnan(velo_global)):
                vx, vy = 0.0, 0.0
            else:
                velo_h = np.array([velo_global[0], velo_global[1], 0.0, 0.0])
                velo_lidar = T_global_to_lidar @ velo_h
                vx, vy = float(velo_lidar[0]), float(velo_lidar[1])

            # Normalise position to [0, 1] within pc_range
            pr   = self.pc_range
            x_n  = float((x - pr[0]) / (pr[3] - pr[0]))
            y_n  = float((y - pr[1]) / (pr[4] - pr[1]))
            z_n  = float((z - pr[2]) / (pr[5] - pr[2]))

            # Log-encode dimensions (l, w, h)
            log_l = math.log(max(length, 0.1))
            log_w = math.log(max(width,  0.1))
            log_h = math.log(max(height, 0.1))

            boxes_list.append([x_n, y_n, z_n, log_l, log_w, log_h,
                                math.sin(yaw), math.cos(yaw), vx, vy])
            labels_list.append(label)

        if not boxes_list:
            return np.zeros((0, 10), dtype=np.float32), np.zeros(0, dtype=np.int64)
        return (np.array(boxes_list, dtype=np.float32),
                np.array(labels_list, dtype=np.int64))

    # ------------------------------------------------------------------ #
    # Augmentation
    # ------------------------------------------------------------------ #

    def _sample_augmentation(self) -> Dict:
        """Sample augmentation parameters once per scene."""
        angle = float(np.random.uniform(-0.3925, 0.3925))
        scale = float(np.random.uniform(0.95, 1.05))
        flip  = bool(np.random.rand() < 0.5)
        return {"angle": angle, "scale": scale, "flip": flip}

    def _augment_points(self, pts: np.ndarray, aug: Dict) -> np.ndarray:
        """Apply rotation+scale+flip to (N,4) point cloud in-place."""
        c, s = math.cos(aug["angle"]), math.sin(aug["angle"])
        rot  = np.array([[c, -s], [s, c]], dtype=np.float32)
        pts[:, :2] = (pts[:, :2] @ rot.T) * aug["scale"]
        if aug["flip"]:
            pts[:, 1] = -pts[:, 1]
        return pts

    def _augment_boxes(self, boxes: np.ndarray, aug: Dict) -> np.ndarray:
        """
        Apply the identical transform used on the point cloud to GT boxes.
        boxes: (G, 10) [x_n,y_n,z_n,log_l,log_w,log_h,sin,cos,vx,vy]
        We must undo normalisation, rotate, renormalise.
        """
        if len(boxes) == 0:
            return boxes
        pr = self.pc_range
        # Decode xy to metric
        x  = boxes[:, 0] * (pr[3] - pr[0]) + pr[0]
        y  = boxes[:, 1] * (pr[4] - pr[1]) + pr[1]

        c, s = math.cos(aug["angle"]), math.sin(aug["angle"])
        rot  = np.array([[c, -s], [s, c]], dtype=np.float32)
        xy   = np.stack([x, y], axis=-1) @ rot.T * aug["scale"]

        if aug["flip"]:
            xy[:, 1] = -xy[:, 1]

        boxes = boxes.copy()
        boxes[:, 0] = (xy[:, 0] - pr[0]) / (pr[3] - pr[0])
        boxes[:, 1] = (xy[:, 1] - pr[1]) / (pr[4] - pr[1])

        # Rotate yaw: sin(θ+α), cos(θ+α)
        sin_t, cos_t = boxes[:, 6], boxes[:, 7]
        new_sin = sin_t * math.cos(aug["angle"]) + cos_t * math.sin(aug["angle"])
        new_cos = cos_t * math.cos(aug["angle"]) - sin_t * math.sin(aug["angle"])
        if aug["flip"]:
            new_sin = -new_sin          # flip negates yaw
        boxes[:, 6] = new_sin
        boxes[:, 7] = new_cos

        # Rotate velocity
        vx, vy = boxes[:, 8].copy(), boxes[:, 9].copy()
        vxy    = np.stack([vx, vy], axis=-1) @ rot.T * aug["scale"]
        if aug["flip"]:
            vxy[:, 1] = -vxy[:, 1]
        boxes[:, 8] = vxy[:, 0]
        boxes[:, 9] = vxy[:, 1]
        return boxes

    def _augment_image(self, img: np.ndarray) -> np.ndarray:
        """Random colour jitter (BGR uint8 → uint8)."""
        img = img.astype(np.float32)
        img *= np.random.uniform(0.6, 1.4)
        return np.clip(img, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------ #
    # __getitem__
    # ------------------------------------------------------------------ #

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        # Compute LiDAR sensor→ego and ego→global transforms once
        T_ls, T_le = self._lidar_sensor_to_global(sample)

        # --- Point cloud (LiDAR sensor frame) ---
        pts = self._load_pointcloud(sample)  # (N, 4)

        # --- Annotations (LiDAR sensor frame, normalised) ---
        gt_boxes, gt_labels = self._load_annotations(sample, T_ls, T_le)

        # --- Augmentation (same params applied to pts and boxes) ---
        if self.augment:
            aug = self._sample_augmentation()
            pts      = self._augment_points(pts, aug)
            gt_boxes = self._augment_boxes(gt_boxes, aug)

        # --- Voxelise ---
        voxels, coords, num_pts = hard_voxelise(
            pts.astype(np.float32),
            self.voxel_size.astype(np.float32),
            self.pc_range.astype(np.float32),
            self.max_points,
            self.max_voxels,
        )

        # --- Images + projection matrices ---
        imgs, lidar2img = self._load_images_and_proj(sample, T_ls, T_le)

        if self.augment:
            imgs = np.stack([self._augment_image(im) for im in imgs])

        # BGR uint8 → RGB float32 normalised
        imgs_rgb = imgs[:, :, :, ::-1].copy().astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        imgs_rgb = (imgs_rgb - mean) / std               # (N, H, W, 3)
        imgs_rgb = imgs_rgb.transpose(0, 3, 1, 2)        # (N, 3, H, W)

        # Composed transform for evaluation: LiDAR sensor frame -> global frame.
        # Required because the official nuScenes evaluator expects submitted
        # boxes in the GLOBAL frame, while the model predicts in the LiDAR
        # sensor frame (same frame as pc_range/voxel_size). Without this, an
        # evaluation script would silently submit boxes in the wrong frame and
        # produce near-zero, meaningless metrics with no error raised.
        lidar2global = T_le @ T_ls  # (4, 4)

        return {
            "voxels":       torch.from_numpy(voxels).float(),                # (M, P, 4)
            "num_points":   torch.from_numpy(num_pts).long(),                # (M,)
            "coords":       torch.from_numpy(coords).long(),                 # (M, 3) z,y,x
            "camera_imgs":  torch.from_numpy(imgs_rgb).float(),              # (6, 3, H, W)
            "lidar2img":    torch.from_numpy(np.stack(lidar2img)).float(),   # (6, 4, 4)
            "gt_boxes":     torch.from_numpy(gt_boxes).float(),              # (G, 10)
            "gt_labels":    torch.from_numpy(gt_labels).long(),              # (G,)
            "sample_token": sample["token"],
            "lidar2global": torch.from_numpy(lidar2global).float(),          # (4, 4)
        }


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn(batch: List[Dict]) -> Dict:
    """
    Stack fixed-size tensors; concatenate variable-size voxel tensors
    and prepend the batch index to coords → [batch_idx, z, y, x].
    """
    result: Dict = {}
    for k in batch[0].keys():
        if k == "sample_token":
            result[k] = [b[k] for b in batch]
        elif k in ("gt_boxes", "gt_labels"):
            result[k] = [b[k] for b in batch]
        elif k == "coords":
            indexed = []
            for i, b in enumerate(batch):
                bi = torch.full((b[k].shape[0], 1), i, dtype=b[k].dtype)
                indexed.append(torch.cat([bi, b[k]], dim=1))   # (M_i, 4)
            result[k] = torch.cat(indexed, dim=0)              # (M_total, 4)
        elif k in ("voxels", "num_points"):
            result[k] = torch.cat([b[k] for b in batch], dim=0)
        else:
            result[k] = torch.stack([b[k] for b in batch])
    return result
