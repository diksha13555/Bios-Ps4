"""Sim-friendly export from JSONL timeline."""

from __future__ import annotations

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
            hands.append(
                {
                    "label": hand["label"],
                    "wrist_xyz": [round(v, 5) for v in w],
                    "wrist_quat_xyzw": None if quat is None else [round(v, 5) for v in quat],
                    "finger_curls_rad": {k: round(v, 5) for k, v in finger_curls(pts).items()},
                }
            )
        exported.append({"frame": frame["frame"], "t_sec": frame["t_sec"], "hands": hands})
    return {"source": source, "fps": fps, "frames": exported}
