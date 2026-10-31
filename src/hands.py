"""Hand track for PS4: 21 image + world landmarks, stable identity, re-entry, smoothing."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np

from src.mirrored_label import correct_detection_label

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "models" / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (5, 9),
    (9, 13),
    (13, 17),
)

FINGERTIPS = (4, 8, 12, 16, 20)
PALM_LANDMARKS = (0, 5, 9, 13, 17)


def download_model(dest: Path = DEFAULT_MODEL) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    urllib.request.urlretrieve(MODEL_URL, dest)
    return dest


def landmarks_bbox(landmarks, pad: float = 0.02) -> tuple[float, float, float, float]:
    xs = [point[0] for point in landmarks]
    ys = [point[1] for point in landmarks]
    return (
        max(0.0, min(xs) - pad),
        max(0.0, min(ys) - pad),
        min(1.0, max(xs) + pad),
        min(1.0, max(ys) + pad),
    )


def bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def center_distance(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ca, cb = bbox_center(a), bbox_center(b)
    return float(((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5)


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _other(label: str) -> str:
    return "Right" if label == "Left" else "Left"


def _fresh_identity(preferred: str, taken: set[str]) -> str | None:
    if preferred not in taken:
        return preferred
    alternate = _other(preferred)
    if alternate not in taken:
        return alternate
    return None


def assign_tracks(
    tracks: list[dict],
    detections: list[dict],
    *,
    max_gap: int = 10,
    iou_min: float = 0.25,
    center_dist_max: float = 0.18,
) -> tuple[list[dict], list[dict]]:
    """Match detections onto existing tracks by box overlap, not by the raw label."""
    pairs: list[tuple[float, int, int]] = []
    for track_index, track in enumerate(tracks):
        if track["missing"] > max_gap:
            continue
        for det_index, detection in enumerate(detections):
            overlap = bbox_iou(track["bbox"], detection["bbox"])
            if overlap >= iou_min:
                pairs.append((overlap, track_index, det_index))
    pairs.sort(reverse=True)

    used_tracks: set[int] = set()
    used_dets: set[int] = set()
    matches: dict[int, int] = {}
    for _overlap, track_index, det_index in pairs:
        if track_index in used_tracks or det_index in used_dets:
            continue
        used_tracks.add(track_index)
        used_dets.add(det_index)
        matches[track_index] = det_index

    dist_pairs: list[tuple[float, int, int]] = []
    for track_index, track in enumerate(tracks):
        if track_index in used_tracks or track["missing"] > max_gap:
            continue
        for det_index, detection in enumerate(detections):
            if det_index in used_dets:
                continue
            dist = center_distance(track["bbox"], detection["bbox"])
            if dist <= center_dist_max:
                dist_pairs.append((dist, track_index, det_index))
    dist_pairs.sort()
    for _dist, track_index, det_index in dist_pairs:
        if track_index in used_tracks or det_index in used_dets:
            continue
        used_tracks.add(track_index)
        used_dets.add(det_index)
        matches[track_index] = det_index

    updated: list[dict] = []
    taken: set[str] = set()
    for track_index, track in enumerate(tracks):
        if track_index in matches:
            detection = detections[matches[track_index]]
            updated.append(
                {
                    "identity": track["identity"],
                    "bbox": detection["bbox"],
                    "image": detection["image"],
                    "world": detection["world"],
                    "score": detection["score"],
                    "missing": 0,
                    "interpolated": False,
                    "raw_label": detection["label"],
                    "reset_filter": track["missing"] > 0,
                }
            )
            taken.add(track["identity"])
            continue
        missing = track["missing"] + 1
        if missing > max_gap:
            continue
        updated.append(
            {
                "identity": track["identity"],
                "bbox": track["bbox"],
                "image": track["image"],
                "world": track["world"],
                "score": track["score"],
                "missing": missing,
                "interpolated": True,
                "raw_label": track.get("raw_label", track["identity"]),
                "reset_filter": False,
            }
        )
        taken.add(track["identity"])

    for det_index, detection in enumerate(detections):
        if det_index in used_dets:
            continue
        identity = _fresh_identity(detection["label"], taken)
        if identity is None:
            continue
        updated.append(
            {
                "identity": identity,
                "bbox": detection["bbox"],
                "image": detection["image"],
                "world": detection["world"],
                "score": detection["score"],
                "missing": 0,
                "interpolated": False,
                "raw_label": detection["label"],
                "reset_filter": True,
            }
        )
        taken.add(identity)

    order = {"Left": 0, "Right": 1}
    updated.sort(key=lambda hand: order.get(hand["identity"], 9))
    return updated, updated


def _alpha(dt: float, cutoff) -> np.ndarray:
    cutoff = np.maximum(np.asarray(cutoff, dtype=np.float64), 1e-6)
    tau = 1.0 / (2.0 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Light smoother. beta is Hz per unit per second, and landmarks are normalized."""

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.8, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x: np.ndarray | None = None
        self._dx: np.ndarray | None = None
        self._t: float | None = None

    def __call__(self, values, t: float) -> np.ndarray:
        current = np.asarray(values, dtype=np.float64).reshape(-1)
        if self._x is None or self._dx is None or self._t is None:
            self._x = current.copy()
            self._dx = np.zeros_like(current)
            self._t = float(t)
            return self._x.copy()
        dt = max(float(t) - self._t, 1e-6)
        derivative = (current - self._x) / dt
        alpha_d = _alpha(dt, self.d_cutoff)
        derivative_hat = alpha_d * derivative + (1.0 - alpha_d) * self._dx
        cutoff = self.min_cutoff + self.beta * np.abs(derivative_hat)
        alpha = _alpha(dt, cutoff)
        hat = alpha * current + (1.0 - alpha) * self._x
        self._x = hat
        self._dx = derivative_hat
        self._t = float(t)
        return hat.copy()


def preprocess(bgr: np.ndarray, *, max_side: int = 1280, clahe: bool = True) -> tuple[np.ndarray, dict]:
    """Resize and lift the value channel. Normalized landmark coords stay valid."""
    height, width = bgr.shape[:2]
    long_side = max(height, width)
    scale = min(1.0, float(max_side) / float(long_side)) if max_side else 1.0
    proc_w = max(1, int(round(width * scale)))
    proc_h = max(1, int(round(height * scale)))
    image = bgr if (proc_w, proc_h) == (width, height) else cv2.resize(bgr, (proc_w, proc_h))
    if clahe:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue, sat, val = cv2.split(hsv)
        val = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(val)
        image = cv2.cvtColor(cv2.merge((hue, sat, val)), cv2.COLOR_HSV2BGR)
    return image, {"proc_w": image.shape[1], "proc_h": image.shape[0], "orig_w": width, "orig_h": height}


def to_public_hand(hand: dict) -> dict:
    return {
        "label": hand["identity"],
        "score": float(hand["score"]),
        "image": hand["image"],
        "world": hand["world"],
        "interpolated": bool(hand.get("interpolated", False)),
        "tracked": not bool(hand.get("interpolated", False)),
    }


class HandRuntime:
    """Video-mode landmarker plus identity, re-entry, and a small-hand retry."""

    def __init__(
        self,
        model_path: Path,
        *,
        max_side: int = 1280,
        clahe: bool = True,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        max_gap: int = 10,
        roi_area: float = 0.08,
        roi_expand: float = 1.3,
    ):
        if not model_path.exists():
            raise FileNotFoundError(
                f"model not found: {model_path}\nRun: bash scripts/download_models.sh"
            )
        self.max_side = max_side
        self.clahe = clahe
        self.max_gap = max_gap
        self.roi_area = roi_area
        self.roi_expand = roi_expand
        self.tracks: list[dict] = []
        self.filters: dict[str, OneEuroFilter] = {}
        self._world_filters: dict[str, OneEuroFilter] = {}
        self._last_ts = -1
        self.video, self.image = _open_landmarkers(
            model_path,
            min_detection_confidence,
            min_presence_confidence,
            min_tracking_confidence,
        )

    def close(self) -> None:
        self.video.close()
        self.image.close()

    def step(self, bgr: np.ndarray, t_sec: float) -> tuple[list[dict], float]:
        """Return identified hands and the seconds spent inside MediaPipe."""
        import time

        started = time.perf_counter()
        processed, meta = preprocess(bgr, max_side=self.max_side, clahe=self.clahe)
        timestamp = int(round(t_sec * 1000.0))
        if timestamp <= self._last_ts:
            timestamp = self._last_ts + 1
        self._last_ts = timestamp
        detections = _detect(self.video, processed, meta, timestamp, roi=None)
        detections = self._roi_retry(processed, meta, detections)
        elapsed = time.perf_counter() - started
        self.tracks, emitted = assign_tracks(self.tracks, detections, max_gap=self.max_gap)
        self._smooth(emitted, t_sec)
        return emitted, elapsed

    def _smooth(self, hands: list[dict], t_sec: float) -> None:
        live = {hand["identity"] for hand in hands}
        for hand in hands:
            if hand["interpolated"]:
                continue
            filt = self.filters.get(hand["identity"])
            if filt is None or hand.get("reset_filter"):
                filt = OneEuroFilter()
                self.filters[hand["identity"]] = filt
            image = filt(hand["image"], t_sec).reshape(-1, 3)
            hand["image"] = image.tolist()
            hand["bbox"] = landmarks_bbox(hand["image"])
            world_filt = self._world_filters.get(hand["identity"])
            if world_filt is None or hand.get("reset_filter"):
                world_filt = OneEuroFilter(min_cutoff=1.0, beta=1.0)
                self._world_filters[hand["identity"]] = world_filt
            hand["world"] = world_filt(hand["world"], t_sec).reshape(-1, 3).tolist()
        for identity in list(self.filters):
            if identity not in live:
                del self.filters[identity]
        for identity in list(self._world_filters):
            if identity not in live:
                del self._world_filters[identity]

    def _roi_retry(self, processed: np.ndarray, meta: dict, detections: list[dict]) -> list[dict]:
        if not self.tracks:
            return detections
        merged = list(detections)
        for track in self.tracks:
            if track["missing"] > self.max_gap:
                continue
            area = _area(track["bbox"])
            covered = any(bbox_iou(track["bbox"], det["bbox"]) >= 0.25 for det in merged)
            if area >= self.roi_area and covered:
                continue
            roi = _roi_window(track["bbox"], meta, self.roi_expand)
            if roi is None:
                continue
            crop = processed[roi["y"] : roi["y"] + roi["h"], roi["x"] : roi["x"] + roi["w"]]
            if crop.size == 0 or crop.shape[0] < 16 or crop.shape[1] < 16:
                continue
            extra = _detect(self.image, crop, meta, None, roi=roi)
            for candidate in extra:
                match = None
                for index, existing in enumerate(merged):
                    if bbox_iou(existing["bbox"], candidate["bbox"]) >= 0.5:
                        match = index
                        break
                if match is None:
                    merged.append(candidate)
                elif candidate["score"] > merged[match]["score"]:
                    merged[match] = candidate
        return merged


def _area(bbox) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _roi_window(bbox, meta: dict, expand: float) -> dict | None:
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw = max((x2 - x1) * expand, 0.12)
    bh = max((y2 - y1) * expand, 0.12)
    nx1, ny1 = max(0.0, cx - bw / 2.0), max(0.0, cy - bh / 2.0)
    nx2, ny2 = min(1.0, cx + bw / 2.0), min(1.0, cy + bh / 2.0)
    px1 = int(nx1 * meta["proc_w"])
    py1 = int(ny1 * meta["proc_h"])
    px2 = int(min(meta["proc_w"], max(px1 + 16, nx2 * meta["proc_w"])))
    py2 = int(min(meta["proc_h"], max(py1 + 16, ny2 * meta["proc_h"])))
    if px2 - px1 < 16 or py2 - py1 < 16:
        return None
    return {"x": px1, "y": py1, "w": px2 - px1, "h": py2 - py1}


def _open_landmarkers(model_path: Path, det: float, presence: float, tracking: float):
    import mediapipe as mp

    def build(mode):
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mode,
            num_hands=2,
            min_hand_detection_confidence=det,
            min_hand_presence_confidence=presence,
            min_tracking_confidence=tracking,
        )
        return mp.tasks.vision.HandLandmarker.create_from_options(options)

    return (
        build(mp.tasks.vision.RunningMode.VIDEO),
        build(mp.tasks.vision.RunningMode.IMAGE),
    )


def _detect(landmarker, image_bgr: np.ndarray, meta: dict, timestamp_ms: int | None, roi: dict | None):
    import mediapipe as mp

    rgb = np.ascontiguousarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    if timestamp_ms is None:
        result = landmarker.detect(mp_image)
    else:
        result = landmarker.detect_for_video(mp_image, int(timestamp_ms))
    detections = []
    image_sets = result.hand_landmarks or []
    world_sets = result.hand_world_landmarks or []
    handed = result.handedness or []
    for image_lms, world_lms, category in zip(image_sets, world_sets, handed):
        image_pts = []
        world_pts = []
        for image_lm, world_lm in zip(image_lms, world_lms):
            if roi is None:
                x, y = float(image_lm.x), float(image_lm.y)
            else:
                x = (roi["x"] + float(image_lm.x) * roi["w"]) / meta["proc_w"]
                y = (roi["y"] + float(image_lm.y) * roi["h"]) / meta["proc_h"]
            image_pts.append([x, y, float(image_lm.z)])
            world_pts.append([float(world_lm.x), float(world_lm.y), float(world_lm.z)])
        if len(image_pts) != 21:
            continue
        label = category[0].category_name
        label, _spec = correct_detection_label(label)
        detections.append(
            {
                "label": label,
                "score": float(category[0].score),
                "image": image_pts,
                "world": world_pts,
                "bbox": landmarks_bbox(image_pts),
            }
        )
    return detections
