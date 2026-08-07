#!/usr/bin/env python3
"""
CenterPoint-style BEV tracker for nuScenes (tracking-by-detection).

Takes a nuScenes *detection* submission JSON (the same results_nusc.json the
detection eval already produces), links detections across frames within each
scene, and writes a nuScenes *tracking* submission that can be scored with the
official devkit for AMOTA/AMOTP.

Method (CenterPoint, Yin et al. 2021 — the tracker TransFusion also used):
  1. predict  each active track forward by  xy + velocity * dt
  2. associate detections to predicted positions by BEV centre distance,
     greedily, within each class independently
  3. update / birth / kill

Why velocity-based prediction matters here: nuScenes keyframes are 0.5 s apart,
during which a car at 50 km/h moves ~7 m — further than the gap between adjacent
lanes. Predicting forward before matching shrinks the effective search radius
enormously. Because the detector predicts vx, vy directly, a track is
predictable from its FIRST detection, unlike history-based motion models.

Usage:
    python track.py detections.json --out tracks.json \
        --dataroot /data/aimotion/nuScenes-lidarseg/v1.0-trainval
    python track.py detections.json --out tracks.json --eval    # + AMOTA
    python track.py detections.json --out tracks.json --ekf     # + EKF smoothing
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

# nuScenes tracks 7 of the 10 detection classes. barrier / traffic_cone /
# construction_vehicle are excluded because they are static or not tracked.
TRACKING_CLASSES = [
    "bicycle", "bus", "car", "motorcycle", "pedestrian", "trailer", "truck",
]

# Per-class association gates in metres (CenterPoint's values). These reflect
# how far a class can plausibly move in 0.5 s AFTER velocity prediction, plus
# localisation error. Pedestrians get a tight gate because they are dense and
# slow; motorcycles a loose one because their velocity is often poorly
# estimated and they move fast.
DIST_THRESH: Dict[str, float] = {
    "car": 4.0, "truck": 4.0, "bus": 5.5, "trailer": 3.0,
    "pedestrian": 1.0, "motorcycle": 13.0, "bicycle": 3.0,
}

# Frames a track survives without being matched before it is deleted. Too low
# and a brief occlusion causes an ID switch; too high and ghost tracks drift
# through empty space. 3 keyframes = 1.5 s.
MAX_AGE = 3


class Track:
    """A single object trajectory."""

    _next_id = 0

    def __init__(self, det: dict, score_decay: float = 0.0):
        self.id = Track._next_id
        Track._next_id += 1

        self.cls = det["detection_name"]
        self.xyz = np.array(det["translation"], dtype=np.float64)
        self.size = list(det["size"])
        self.rotation = list(det["rotation"])
        self.velocity = np.array(det.get("velocity", [0.0, 0.0]), dtype=np.float64)
        if not np.all(np.isfinite(self.velocity)):
            self.velocity = np.zeros(2)
        self.score = float(det["detection_score"])

        self.time_since_update = 0
        self.hits = 1
        self.score_decay = score_decay

        # EKF state (optional): [x, y, vx, vy] with a diagonal covariance.
        self.P: Optional[np.ndarray] = None

    # -- prediction ------------------------------------------------------
    def predict(self, dt: float) -> np.ndarray:
        """Predicted BEV centre at the next frame (does not mutate state)."""
        return self.xyz[:2] + self.velocity * dt

    def mark_missed(self, dt: float) -> None:
        """No detection matched: coast forward on the motion model."""
        self.xyz[:2] = self.xyz[:2] + self.velocity * dt
        self.time_since_update += 1
        # Decay confidence so coasted tracks rank below observed ones.
        self.score *= (1.0 - self.score_decay)

    # -- update ----------------------------------------------------------
    def update(self, det: dict, dt: float, use_ekf: bool = False) -> None:
        z = np.array(det["translation"], dtype=np.float64)
        v = np.array(det.get("velocity", [0.0, 0.0]), dtype=np.float64)
        if not np.all(np.isfinite(v)):
            v = self.velocity

        if use_ekf:
            self._ekf_update(z[:2], v, dt)
            self.xyz[2] = z[2]
        else:
            # Plain CenterPoint behaviour: take the detection as truth.
            self.xyz = z
            self.velocity = v

        self.size = list(det["size"])
        self.rotation = list(det["rotation"])
        self.score = float(det["detection_score"])
        self.time_since_update = 0
        self.hits += 1

    def _ekf_update(self, z_xy: np.ndarray, z_v: np.ndarray, dt: float) -> None:
        """
        Constant-velocity Kalman update on [x, y, vx, vy].

        Measured as a delta over the plain tracker: it smooths jittery detector
        output and lets a track coast sensibly through a miss, at the cost of
        lagging genuine accelerations. Whether it helps is an empirical question
        on a given detector — report the AMOTA difference rather than assuming.
        """
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=np.float64)
        H = np.eye(4)                       # detector observes position AND velocity
        Q = np.diag([0.5, 0.5, 1.0, 1.0])   # process noise
        R = np.diag([0.5, 0.5, 2.0, 2.0])   # measurement noise (velocity noisier)

        x = np.concatenate([self.xyz[:2], self.velocity])
        if self.P is None:
            self.P = np.diag([1.0, 1.0, 10.0, 10.0])

        # predict
        x = F @ x
        P = F @ self.P @ F.T + Q
        # update
        z = np.concatenate([z_xy, z_v])
        y = z - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        self.P = (np.eye(4) - K @ H) @ P

        self.xyz[:2] = x[:2]
        self.velocity = x[2:]

    # -- output ----------------------------------------------------------
    def to_submission(self, sample_token: str) -> dict:
        return {
            "sample_token": sample_token,
            "translation": [float(v) for v in self.xyz],
            "size": [float(v) for v in self.size],
            "rotation": [float(v) for v in self.rotation],
            "velocity": [float(v) for v in self.velocity],
            "tracking_id": str(self.id),
            "tracking_name": self.cls,
            "tracking_score": float(self.score),
        }


def greedy_match(dets: List[dict], tracks: List[Track], dt: float):
    """
    Greedy nearest-neighbour association on BEV centre distance, per class.

    Greedy rather than Hungarian: it is what CenterPoint uses, it is O(n^2) with
    no dependency, and at nuScenes densities the optimal assignment and the
    greedy one almost always agree. Pairs are committed in ascending distance,
    so the most confident correspondences are fixed first.
    """
    matches, unmatched_dets = [], list(range(len(dets)))
    if not tracks or not dets:
        return matches, unmatched_dets, list(range(len(tracks)))

    pairs = []
    for ti, trk in enumerate(tracks):
        pred = trk.predict(dt)
        gate = DIST_THRESH.get(trk.cls, 4.0)
        for di, det in enumerate(dets):
            if det["detection_name"] != trk.cls:
                continue                       # never associate across classes
            d = float(np.linalg.norm(np.array(det["translation"][:2]) - pred))
            if d < gate:
                pairs.append((d, ti, di))

    pairs.sort(key=lambda p: p[0])
    used_t, used_d = set(), set()
    for d, ti, di in pairs:
        if ti in used_t or di in used_d:
            continue
        used_t.add(ti)
        used_d.add(di)
        matches.append((ti, di))

    unmatched_dets = [i for i in range(len(dets)) if i not in used_d]
    unmatched_trks = [i for i in range(len(tracks)) if i not in used_t]
    return matches, unmatched_dets, unmatched_trks


def track_scene(frames, use_ekf=False, min_score=0.0, score_decay=0.1):
    """
    Run the tracker over one scene's frames, in timestamp order.

    frames: list of (sample_token, timestamp_us, [detections])
    Returns {sample_token: [track submission dicts]}
    """
    out: Dict[str, list] = {}
    tracks: List[Track] = []
    prev_ts = None

    for token, ts, dets in frames:
        dt = 0.5 if prev_ts is None else max((ts - prev_ts) / 1e6, 1e-3)
        prev_ts = ts

        dets = [d for d in dets
                if d["detection_name"] in TRACKING_CLASSES
                and d["detection_score"] >= min_score]

        matches, un_d, un_t = greedy_match(dets, tracks, dt)

        for ti, di in matches:
            tracks[ti].update(dets[di], dt, use_ekf=use_ekf)
        for ti in un_t:
            tracks[ti].mark_missed(dt)
        for di in un_d:
            tracks.append(Track(dets[di], score_decay=score_decay))

        tracks = [t for t in tracks if t.time_since_update <= MAX_AGE]
        out[token] = [t.to_submission(token) for t in tracks]

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("detections", help="nuScenes detection submission JSON")
    ap.add_argument("--out", default="tracking_results.json")
    ap.add_argument("--dataroot",
                    default="/data/aimotion/nuScenes-lidarseg/v1.0-trainval")
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--eval-set", default="val")
    ap.add_argument("--ekf", action="store_true",
                    help="enable EKF smoothing (measure the AMOTA delta)")
    ap.add_argument("--min-score", type=float, default=0.0,
                    help="drop detections below this before tracking")
    ap.add_argument("--eval", action="store_true",
                    help="run the official TrackingEval afterwards")
    args = ap.parse_args()

    from nuscenes import NuScenes

    print("loading detections ...")
    sub = json.load(open(args.detections))["results"]
    print(f"  {len(sub)} samples, {sum(len(v) for v in sub.values())} detections")

    print("loading nuScenes (for scene grouping) ...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    # ---- group frames by SCENE and order by timestamp ------------------
    # Tracking is only meaningful within a scene. The detection submission is an
    # unordered dict of samples; processing it as one stream would link objects
    # across unrelated scenes and produce nonsense.
    scenes: Dict[str, list] = defaultdict(list)
    for token in sub:
        s = nusc.get("sample", token)
        scenes[s["scene_token"]].append((token, s["timestamp"], sub[token]))
    for k in scenes:
        scenes[k].sort(key=lambda f: f[1])
    print(f"  grouped into {len(scenes)} scenes")

    # ---- run ------------------------------------------------------------
    results: Dict[str, list] = {}
    for i, (scene_token, frames) in enumerate(scenes.items(), 1):
        Track._next_id = 0            # ids need only be unique within a scene
        results.update(track_scene(frames, use_ekf=args.ekf,
                                   min_score=args.min_score))
        if i % 25 == 0:
            print(f"  {i}/{len(scenes)} scenes")

    n_tracks = sum(len(v) for v in results.values())
    print(f"tracking done: {n_tracks} track boxes over {len(results)} samples")

    meta = {"use_camera": True, "use_lidar": True, "use_radar": False,
            "use_map": False, "use_external": False}
    with open(args.out, "w") as f:
        json.dump({"meta": meta, "results": results}, f)
    print(f"written {args.out}")

    if args.eval:
        from nuscenes.eval.tracking.evaluate import TrackingEval
        from nuscenes.eval.common.config import config_factory
        out_dir = os.path.splitext(args.out)[0] + "_eval"
        ev = TrackingEval(config=config_factory("tracking_nips_2019"),
                          result_path=os.path.abspath(args.out),
                          eval_set=args.eval_set, output_dir=out_dir,
                          nusc_version=args.version, nusc_dataroot=args.dataroot)
        m = ev.main(render_curves=False)
        print("\n" + "=" * 46)
        print(f"  AMOTA : {m['amota']:.4f}")
        print(f"  AMOTP : {m['amotp']:.4f}")
        print("=" * 46)


if __name__ == "__main__":
    main()
