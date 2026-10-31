"""Sim-friendly export from JSONL timeline."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# wrist landmark index = 0
FINGER_CHAINS = ((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20), (1, 2, 4))


def finger_curls(points: np.ndarray) -> dict[str, float]:
    names = ("index", "middle", "ring", "pinky", "thumb")
    out = {}
    for name, (a, b, c) in zip(names, FINGER_CHAINS):
        v1 = points[a] - points[b]
        v2 = points[c] - points[b]
        d = float(np.linalg.norm(v1) * np.linalg.norm(v2))
        if d < 1e-8:
            out[name] = 0.0
        else:
            cos = float(np.clip(np.dot(v1, v2) / d, -1, 1))
            out[name] = float(np.pi - np.arccos(cos))
    return out


def wrist_pose(points: np.ndarray) -> tuple[list[float], list[float] | None]:
    wrist = points[0]
    xi = points[5] - wrist
    xp = points[17] - wrist
    if np.linalg.norm(xi) < 1e-6 or np.linalg.norm(xp) < 1e-6:
        return wrist.tolist(), None
    x = xi / np.linalg.norm(xi)
    y = xp - np.dot(xp, x) * x
    if np.linalg.norm(y) < 1e-6:
        return wrist.tolist(), None
    y = y / np.linalg.norm(y)
    z = np.cross(x, y)
    rot = np.stack([x, y, z / np.linalg.norm(z)], axis=1)
    quat = Rotation.from_matrix(rot).as_quat().tolist()
    return wrist.tolist(), quat


def export_sim(frames: list[dict], *, fps: float, source: str) -> dict:
    exported = []
    for frame in frames:
        hands = []
        for hand in frame["hands"]:
            pts = np.asarray(hand.get("camera") or hand["world"], dtype=np.float64)
            w, quat = wrist_pose(pts)
            confidence = hand.get("joint_confidence")
            hands.append(
                {
                    "label": hand["label"],
                    "wrist_xyz": [round(v, 5) for v in w],
                    "wrist_quat_xyzw": None if quat is None else [round(v, 5) for v in quat],
                    "finger_curls_rad": {k: round(v, 5) for k, v in finger_curls(pts).items()},
                    "joint_confidence": None if not confidence else [round(float(v), 4) for v in confidence],
                    "action": hand.get("action"),
                    "grasp": hand.get("grasp"),
                    "inferred": bool(hand.get("inferred", False)),
                }
            )
        exported.append({"frame": frame["frame"], "t_sec": frame["t_sec"], "hands": hands})
    return {"source": source, "fps": fps, "frames": exported}


def export_world_sim(trajectories: dict[str, list[dict]], frames: list[dict], *, fps: float, source: str) -> dict:
    """Fixed-world wrist trajectories + finger curls in a simulator-friendly document."""
    source_hands = {(frame["frame"], hand["label"]): hand for frame in frames for hand in frame["hands"]}
    by_frame: dict[int, dict[str, dict]] = {}
    for label, samples in trajectories.items():
        for sample in samples:
            by_frame.setdefault(sample["frame"], {})[label] = sample
    exported = []
    for frame_index in sorted(by_frame):
        entry = by_frame[frame_index]
        hands = []
        for label in ("Left", "Right"):
            sample = entry.get(label)
            if sample is None:
                continue
            source_hand = source_hands.get((frame_index, label))
            curls = None
            if source_hand is not None:
                points = np.asarray(source_hand.get("camera") or source_hand["world"], dtype=np.float64)
                curls = {k: round(v, 5) for k, v in finger_curls(points).items()}
            quat = sample["quat_xyzw"]
            hands.append(
                {
                    "label": label,
                    "wrist_xyz": [round(v, 5) for v in sample["wrist_xyz"]],
                    "wrist_quat_xyzw": None if quat is None else [round(v, 5) for v in quat],
                    "finger_curls_rad": curls,
                    "tracked": sample["tracked"],
                    "interpolated": sample["interpolated"],
                    "smoothed": sample.get("smoothed", False),
                    "joint_confidence": None
                    if source_hand is None or not source_hand.get("joint_confidence")
                    else [round(float(v), 4) for v in source_hand["joint_confidence"]],
                    "action": None if source_hand is None else source_hand.get("action"),
                    "grasp": None if source_hand is None else source_hand.get("grasp"),
                    "inferred": False if source_hand is None else bool(source_hand.get("inferred", False)),
                }
            )
        t_sec = next(iter(entry.values()))["t_sec"]
        exported.append({"frame": frame_index, "t_sec": t_sec, "hands": hands})
    return {
        "source": source,
        "fps": fps,
        "coordinate_frame": {
            "name": "camera_at_frame_0",
            "axes": "+x right, +y down, +z forward (OpenCV)",
            "units": "meters",
            "scale": "rotation exact; translation up-to-scale (monocular camera-motion fallback)",
        },
        "frames": exported,
    }


def write_world_csv(path: Path, world_sim: dict) -> None:
    """Flat per-frame/per-hand CSV: one wrist pose + curl set per row."""
    columns = [
        "frame", "t_sec", "label", "x", "y", "z", "qx", "qy", "qz", "qw",
        "curl_index", "curl_middle", "curl_ring", "curl_pinky", "curl_thumb",
        "tracked", "interpolated", "inferred", "action", "grasp", "mean_joint_confidence",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for frame in world_sim["frames"]:
            for hand in frame["hands"]:
                quat = hand["wrist_quat_xyzw"] or [None] * 4
                curls = hand["finger_curls_rad"] or {}
                writer.writerow(
                    {
                        "frame": frame["frame"],
                        "t_sec": frame["t_sec"],
                        "label": hand["label"],
                        "x": hand["wrist_xyz"][0],
                        "y": hand["wrist_xyz"][1],
                        "z": hand["wrist_xyz"][2],
                        "qx": quat[0], "qy": quat[1], "qz": quat[2], "qw": quat[3],
                        "curl_index": curls.get("index"),
                        "curl_middle": curls.get("middle"),
                        "curl_ring": curls.get("ring"),
                        "curl_pinky": curls.get("pinky"),
                        "curl_thumb": curls.get("thumb"),
                        "tracked": int(hand["tracked"]),
                        "interpolated": int(hand["interpolated"]),
                        "inferred": int(hand.get("inferred", False)),
                        "action": hand.get("action"),
                        "grasp": hand.get("grasp"),
                        "mean_joint_confidence": None
                        if not hand.get("joint_confidence")
                        else round(float(np.mean(hand["joint_confidence"])), 4),
                    }
                )
