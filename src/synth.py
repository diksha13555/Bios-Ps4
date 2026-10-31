"""Synthetic egocentric clip with exact 3D ground truth for benchmarking.

A capsule hand is drawn over a static textured background while a scripted
hand/camera motion plays out. Every frame's true 2D landmarks, world wrist
position, and action label are known, so the real pipeline can be scored
against exact ground truth instead of self-reported confidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.geometry import PARENT

WIDTH, HEIGHT, FPS = 640, 480, 30
FOCAL = 520.0
FINGERS = {
    "index": {"chain": (5, 6, 7, 8), "mcp": (0.068, 0.024, 0.0), "lengths": (0.040, 0.025, 0.022), "splay": 0.20},
    "middle": {"chain": (9, 10, 11, 12), "mcp": (0.072, 0.000, 0.0), "lengths": (0.045, 0.028, 0.024), "splay": 0.02},
    "ring": {"chain": (13, 14, 15, 16), "mcp": (0.066, -0.024, 0.0), "lengths": (0.041, 0.026, 0.022), "splay": -0.20},
    "pinky": {"chain": (17, 18, 19, 20), "mcp": (0.055, -0.046, 0.0), "lengths": (0.032, 0.018, 0.018), "splay": -0.42},
    "thumb": {"chain": (1, 2, 3, 4), "mcp": (0.022, 0.040, 0.006), "lengths": (0.032, 0.030, 0.024), "splay": -0.95},
}


def hand_local(curls: dict[str, float]) -> np.ndarray:
    """21 hand-local joints (meters) with per-finger curl in radians."""
    points = np.zeros((21, 3), dtype=np.float64)
    for name, spec in FINGERS.items():
        curl = float(np.clip(curls.get(name, 0.0), 0.0, 1.6))
        mcp = np.asarray(spec["mcp"], dtype=np.float64)
        chain = spec["chain"]
        points[chain[0]] = mcp
        angle = float(spec["splay"])
        current = mcp.copy()
        for joint, length, share in zip(chain[1:], spec["lengths"], (0.55, 0.65, 0.45)):
            angle += curl * share
            direction = np.array([np.cos(angle), np.sin(angle), 0.0])
            current = current + direction * length
            points[joint] = current
    return points


def project(points_camera: np.ndarray) -> np.ndarray:
    z = np.maximum(points_camera[:, 2], 1e-4)
    x = FOCAL * points_camera[:, 0] / z + WIDTH / 2
    y = FOCAL * points_camera[:, 1] / z + HEIGHT / 2
    return np.stack([x / WIDTH, y / HEIGHT], axis=1)


def script_frame(index: int) -> dict:
    """Ground-truth world state for one frame of the scripted clip."""
    curls = {name: 0.15 for name in FINGERS}
    world_wrist = np.array([0.0, 0.0, 0.45])
    action = "idle"
    visible = 30 <= index < 270
    if 30 <= index < 90:
        alpha = (index - 30) / 59.0
        world_wrist = np.array([0.22 * (1 - alpha), 0.05, 0.45])
        action = "reach"
    elif 90 <= index < 105:
        alpha = (index - 90) / 14.0
        curls = {name: 0.15 + 1.05 * alpha for name in FINGERS}
        action = "grasp"
    elif 105 <= index < 150:
        curls = {name: 1.2 for name in FINGERS}
        action = "hold"
    elif 150 <= index < 165:
        alpha = (index - 150) / 14.0
        curls = {name: 1.2 - 1.05 * alpha for name in FINGERS}
        action = "release"
    elif 165 <= index < 240:
        alpha = (index - 165) / 74.0
        world_wrist = np.array([-0.22 * alpha, 0.05, 0.45])
        action = "reach"
    elif 240 <= index < 270:
        action = "idle"
    camera_rot = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(0.0006 * index), -np.sin(0.0006 * index)],
        [0.0, np.sin(0.0006 * index), np.cos(0.0006 * index)],
    ])
    camera_t = np.array([0.0005 * index, 0.0, 0.0])
    return {
        "visible": visible,
        "action": action if visible else None,
        "curls": curls,
        "world_wrist": world_wrist,
        "camera_rot": camera_rot,
        "camera_t": camera_t,
    }


def render_background() -> np.ndarray:
    rng = np.random.default_rng(7)
    # Mid-gray base, close to real egocentric footage. A very dark base makes
    # CLAHE amplify the background and wash out the hand's local contrast.
    background = np.full((HEIGHT, WIDTH, 3), (96, 100, 108), dtype=np.uint8)
    for _ in range(140):
        x, y = int(rng.integers(0, WIDTH)), int(rng.integers(0, HEIGHT))
        w, h = int(rng.integers(4, 46)), int(rng.integers(4, 30))
        color = tuple(int(v) for v in rng.integers(60, 190, size=3))
        if rng.random() < 0.5:
            cv2.rectangle(background, (x, y), (min(x + w, WIDTH - 1), min(y + h, HEIGHT - 1)), color, -1)
        else:
            cv2.circle(background, (x, y), int(rng.integers(2, 12)), color, -1)
    cv2.rectangle(background, (0, int(HEIGHT * 0.72)), (WIDTH, HEIGHT), (120, 110, 100), -1)
    return cv2.GaussianBlur(background, (3, 3), 0)


def render_frame(background: np.ndarray, points_camera: np.ndarray | None, index: int) -> np.ndarray:
    frame = background.copy()
    rng = np.random.default_rng(index)
    noise = rng.normal(0, 2.0, frame.shape).astype(np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    if points_camera is None:
        return frame
    z = points_camera[:, 2]
    pixels = project(points_camera) * np.array([WIDTH, HEIGHT])
    skin = (150, 172, 205)
    palm = np.array([pixels[i] for i in (0, 5, 9, 13, 17)], dtype=np.int32)
    cv2.fillConvexPoly(frame, palm, (120, 140, 175), cv2.LINE_AA)
    order = sorted(PARENT.items(), key=lambda item: -z[item[0]])
    for child, parent in order:
        p1 = tuple(np.round(pixels[parent]).astype(int))
        p2 = tuple(np.round(pixels[child]).astype(int))
        depth = float(max(z[child], 0.15))
        thickness = int(np.clip(0.016 * FOCAL / depth, 3, 22))
        cv2.line(frame, p1, p2, (105, 125, 160), thickness + 3, cv2.LINE_AA)
        cv2.line(frame, p1, p2, skin, thickness, cv2.LINE_AA)
    return frame


def generate_clip(output_dir: Path, n_frames: int = 300) -> dict:
    """Render the clip plus eval-compatible labels and world ground truth."""
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "synthetic.mp4"
    labels_path = output_dir / "synthetic_labels.jsonl"
    gt_path = output_dir / "synthetic_gt.json"
    background = render_background()
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    labels = []
    gt_frames = []
    orientation = np.array([[0.25, 0.10, -1.05]]).T  # fingers angled downward, palm to camera
    rot_wrist = cv2.Rodrigues(orientation)[0]
    for index in range(n_frames):
        state = script_frame(index)
        frame_hands = []
        points_camera = None
        wrist_world = None
        if state["visible"]:
            local = hand_local(state["curls"])
            local = (rot_wrist @ local.T).T
            wrist_world = state["world_wrist"]
            world_points = local + wrist_world
            cam = state["camera_rot"].T @ (world_points - state["camera_t"]).T
            points_camera = cam.T
            normalized = project(points_camera)
            frame_hands.append({
                "label": "Right",
                "image": [[round(float(x), 5), round(float(y), 5), 0.0] for x, y in normalized],
            })
        writer.write(render_frame(background, points_camera, index))
        labels.append({"frame": index, "hands": frame_hands})
        gt_frames.append({
            "frame": index,
            "t_sec": round(index / FPS, 4),
            "action": state["action"],
            "wrist_world": None if wrist_world is None else [round(float(v), 5) for v in wrist_world],
        })
    writer.release()
    with labels_path.open("w", encoding="utf-8") as handle:
        for row in labels:
            handle.write(json.dumps(row) + "\n")
    gt_path.write_text(json.dumps({"fps": FPS, "frames": gt_frames}, indent=2), encoding="utf-8")
    return {"video": video_path, "labels": labels_path, "gt": gt_path}
