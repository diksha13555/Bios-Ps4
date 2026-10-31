"""Fixed world frame via the camera-motion fallback, plus world wrist trajectories.

World frame = camera at frame 0 (OpenCV axes: +x right, +y down, +z forward).
Rotation is exact; translation is up-to-scale (monocular scale ambiguity).

HaWoR seam: any per-frame camera trajectory can replace CameraMotionEstimator
output in attach_world, and a HaWoR runner can write world-space hands straight
into `world_fixed`, mirroring the WiLoR mesh pattern.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp

from src.camera_frame import intrinsics
from src.export_sim import wrist_pose


def _relative_pose(prev_pts: np.ndarray, curr_pts: np.ndarray, k: np.ndarray):
    """Essential-matrix relative pose from 2D correspondences. None on failure."""
    if len(prev_pts) < 8 or len(curr_pts) < 8:
        return None
    try:
        essential, mask = cv2.findEssentialMat(prev_pts, curr_pts, k, cv2.RANSAC, 0.999, 1.0)
        if essential is None:
            return None
        inliers, rot, tvec, _ = cv2.recoverPose(essential, prev_pts, curr_pts, k, mask=mask)
    except cv2.error:
        return None
    if inliers < 8 or not np.all(np.isfinite(rot)) or not np.all(np.isfinite(tvec)):
        return None
    return rot, tvec.reshape(3), int(inliers)


def _rotation_from_homography(prev_pts: np.ndarray, curr_pts: np.ndarray, k: np.ndarray, min_inliers: int = 12):
    """Rotation-only fallback for rotation-dominant/low-parallax frames.

    recoverPose rejects those frames on cheirality; under (near) pure rotation
    translation is unobservable anyway, so only rotation is applied.
    """
    if len(prev_pts) < min_inliers:
        return None
    try:
        homography, mask = cv2.findHomography(prev_pts, curr_pts, cv2.RANSAC, 1.5)
        if homography is None or mask is None or int(mask.sum()) < min_inliers:
            return None
        _, rotations, _, _ = cv2.decomposeHomographyMat(homography, k)
    except cv2.error:
        return None
    if not rotations:
        return None
    # Small inter-frame motion prior: pick the smallest-angle decomposition.
    best = min(rotations, key=lambda r: Rotation.from_matrix(r).magnitude())
    if Rotation.from_matrix(best).magnitude() > 0.5 or not np.all(np.isfinite(best)):
        return None
    return best, int(mask.sum())


class CameraMotionEstimator:
    """Monocular visual odometry on background features; hands are masked out."""

    def __init__(self, width: int, height: int, *, max_side: int = 640, min_features: int = 8):
        scale = min(1.0, float(max_side) / float(max(width, height))) if max_side else 1.0
        self.width = max(1, int(round(width * scale)))
        self.height = max(1, int(round(height * scale)))
        self.k = intrinsics(width, height)
        self.k[:2] *= scale
        self.min_features = int(min_features)
        self._rot = np.eye(3)
        self._t = np.zeros(3)
        self._prev_gray: np.ndarray | None = None
        self._prev_pts: np.ndarray | None = None
        self._poses: list[dict] = []

    def _background_mask(self, hand_boxes) -> np.ndarray:
        mask = np.full((self.height, self.width), 255, dtype=np.uint8)
        for x1, y1, x2, y2 in hand_boxes:
            px1 = int(max(0.0, x1) * self.width)
            py1 = int(max(0.0, y1) * self.height)
            px2 = int(min(1.0, x2) * self.width) + 1
            py2 = int(min(1.0, y2) * self.height) + 1
            cv2.rectangle(mask, (px1, py1), (px2, py2), 0, thickness=-1)
        return mask

    def _apply_relative(self, rot: np.ndarray, tvec: np.ndarray) -> None:
        # curr_cam = rot @ prev_cam + tvec, so T_w_c_new = T_w_c_prev @ inv([rot|tvec]).
        self._rot = self._rot @ rot.T
        self._t = self._t - self._rot @ tvec

    def step(self, bgr: np.ndarray, hand_boxes=()) -> None:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if (gray.shape[1], gray.shape[0]) != (self.width, self.height):
            gray = cv2.resize(gray, (self.width, self.height))
        mask = self._background_mask(hand_boxes)
        valid = False
        inliers = 0
        mode = "carry"
        if self._prev_gray is None:
            valid = True  # frame 0 anchors the world frame
            mode = "anchor"
        elif self._prev_pts is not None and len(self._prev_pts) >= self.min_features:
            pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, gray, self._prev_pts, None, winSize=(21, 21), maxLevel=3
            )
            if pts is not None:
                good = status.reshape(-1) == 1
                prev = self._prev_pts.reshape(-1, 2)[good]
                curr = pts.reshape(-1, 2)[good]
                inside = (
                    (curr[:, 0] >= 0) & (curr[:, 0] < self.width)
                    & (curr[:, 1] >= 0) & (curr[:, 1] < self.height)
                )
                prev, curr = prev[inside], curr[inside]
                relative = _relative_pose(prev, curr, self.k)
                if relative is not None:
                    rot, tvec, inliers = relative
                    self._apply_relative(rot, tvec)
                    valid = True
                    mode = "essential"
                else:
                    fallback = _rotation_from_homography(prev, curr, self.k)
                    if fallback is not None:
                        rot, inliers = fallback
                        self._apply_relative(rot, np.zeros(3))
                        valid = True
                        mode = "rotation"
        self._poses.append(
            {
                "rotation": self._rot.copy(),
                "translation": self._t.copy(),
                "valid": valid,
                "inliers": inliers,
                "mode": mode,
            }
        )
        seeded = cv2.goodFeaturesToTrack(
            gray, maxCorners=800, qualityLevel=0.01, minDistance=8, mask=mask
        )
        self._prev_pts = None if seeded is None else seeded.astype(np.float32)
        self._prev_gray = gray

    def trajectory(self) -> list[dict]:
        return self._poses


def attach_world(frames: list[dict], poses: list[dict]) -> None:
    """Rigidly map each hand's camera-frame joints into the fixed world frame."""
    for index, frame in enumerate(frames):
        pose = poses[index] if index < len(poses) else None
        if pose is None:
            frame["camera_pose_world"] = None
            for hand in frame["hands"]:
                hand["world_fixed"] = None
            continue
        rot = np.asarray(pose["rotation"], dtype=np.float64)
        tvec = np.asarray(pose["translation"], dtype=np.float64).reshape(3)
        frame["camera_pose_world"] = {
            "rotation": rot.tolist(),
            "translation": tvec.tolist(),
            "valid": bool(pose["valid"]),
        }
        for hand in frame["hands"]:
            camera = hand.get("camera")
            if camera is None:
                hand["world_fixed"] = None
            else:
                hand["world_fixed"] = ((rot @ np.asarray(camera, dtype=np.float64).T).T + tvec).tolist()


def build_world_trajectories(frames: list[dict]) -> dict[str, list[dict]]:
    """Per-label wrist position + orientation time series from world_fixed joints."""
    trajectories: dict[str, list[dict]] = {}
    for frame in frames:
        for hand in frame["hands"]:
            points = hand.get("world_fixed")
            if points is None:
                continue
            wrist, quat = wrist_pose(np.asarray(points, dtype=np.float64))
            trajectories.setdefault(hand["label"], []).append(
                {
                    "frame": frame["frame"],
                    "t_sec": frame["t_sec"],
                    "wrist_xyz": wrist,
                    "quat_xyzw": quat,
                    "tracked": hand.get("tracked", True),
                    "interpolated": hand.get("interpolated", False),
                    "smoothed": False,
                }
            )
    return trajectories


def _fill_between(start: dict, end: dict) -> list[dict]:
    gap = end["frame"] - start["frame"]
    out = []
    for step in range(1, gap):
        alpha = step / float(gap)
        wrist = ((1 - alpha) * np.asarray(start["wrist_xyz"]) + alpha * np.asarray(end["wrist_xyz"])).tolist()
        quat = None
        if start["quat_xyzw"] is not None and end["quat_xyzw"] is not None:
            rotations = Rotation.from_quat([start["quat_xyzw"], end["quat_xyzw"]])
            quat = Slerp([0.0, 1.0], rotations)([alpha]).as_quat()[0].tolist()
        out.append(
            {
                "frame": start["frame"] + step,
                "t_sec": round((1 - alpha) * start["t_sec"] + alpha * end["t_sec"], 4),
                "wrist_xyz": wrist,
                "quat_xyzw": quat,
                "tracked": False,
                "interpolated": True,
                "smoothed": False,
            }
        )
    return out


def _segments(samples: list[dict], max_gap: int) -> list[list[dict]]:
    segments: list[list[dict]] = []
    current = [samples[0]]
    for prev, sample in zip(samples, samples[1:]):
        gap = sample["frame"] - prev["frame"]
        if gap > max_gap + 1:
            segments.append(current)
            current = [sample]
            continue
        current.extend(_fill_between(prev, sample))
        current.append(sample)
    segments.append(current)
    return segments


def _aligned_quats(samples: list[dict]) -> np.ndarray | None:
    filled = [sample["quat_xyzw"] for sample in samples]
    last = None
    for i, quat in enumerate(filled):
        if quat is None:
            filled[i] = last
        else:
            last = quat
    following = None
    for i in range(len(filled) - 1, -1, -1):
        if filled[i] is None:
            filled[i] = following
        else:
            following = filled[i]
    if any(quat is None for quat in filled):
        return None
    arr = np.asarray(filled, dtype=np.float64)
    for i in range(1, len(arr)):
        if float(np.dot(arr[i - 1], arr[i])) < 0:
            arr[i] = -arr[i]
    return arr


def _savgol(values: np.ndarray, window: int, polyorder: int) -> np.ndarray | None:
    n = len(values)
    size = min(window, n if n % 2 == 1 else n - 1)
    # window == polyorder + 1 fits the polynomial exactly, i.e. no smoothing.
    if size < 3 or size < polyorder + 2:
        return None
    return savgol_filter(values, size, polyorder, axis=0)


def smooth_world_trajectories(
    trajectories: dict[str, list[dict]], *, window: int = 11, polyorder: int = 2, max_gap: int = 5
) -> None:
    """Smooth wrist positions (savgol) and orientations (sign-aligned savgol + renormalize)."""
    for label, samples in trajectories.items():
        if not samples:
            continue
        samples.sort(key=lambda sample: sample["frame"])
        smoothed: list[dict] = []
        for segment in _segments(samples, max_gap):
            positions = _savgol(np.asarray([s["wrist_xyz"] for s in segment], dtype=np.float64), window, polyorder)
            quats = _aligned_quats(segment)
            quats = None if quats is None else _savgol(quats, window, polyorder)
            if positions is None and quats is None:
                smoothed.extend(segment)
                continue
            for i, sample in enumerate(segment):
                if positions is not None:
                    sample["wrist_xyz"] = positions[i].tolist()
                if quats is not None:
                    norm = float(np.linalg.norm(quats[i]))
                    if norm > 1e-8:
                        sample["quat_xyzw"] = (quats[i] / norm).tolist()
                sample["smoothed"] = True
            smoothed.extend(segment)
        trajectories[label] = smoothed


def putdown_pickup_drift(frames: list[dict], *, max_gap: int = 90, key: str = "world_fixed") -> dict:
    """Drift between a release and the next grasp, in the fixed world frame.

    A put-down and a later pick-up of the same object should share a world
    point. The distance between those wrist positions is world-frame drift
    when no labelled object is available. `key` selects the joint field
    (world_fixed raw, or world_grounded after contact anchoring).
    """
    gaps = []
    for label in ("Left", "Right"):
        events = []
        for frame in frames:
            for hand in frame["hands"]:
                if hand["label"] != label or hand.get(key) is None:
                    continue
                action = hand.get("action")
                if action in {"release", "grasp"}:
                    wrist = np.asarray(hand[key], dtype=np.float64)[0, :3]
                    events.append((frame["frame"], action, wrist))
        for index, (frame_index, action, wrist) in enumerate(events):
            if action != "release":
                continue
            for later_frame, later_action, later_wrist in events[index + 1 :]:
                if later_frame - frame_index > max_gap:
                    break
                if later_action != "grasp":
                    continue
                gaps.append(float(np.linalg.norm(later_wrist - wrist)))
                break
    if not gaps:
        return {"pairs": 0, "mean_m": None, "max_m": None}
    return {
        "pairs": len(gaps),
        "mean_m": round(float(np.mean(gaps)), 5),
        "max_m": round(float(np.max(gaps)), 5),
    }
