"""WiLoR mesh track in the same per-frame hand format as the MediaPipe JSONL.

Joints stay in the JSONL (`world`, `image`, `camera`). Full MANO vertices go to a
`.npz` sidecar; each hand carries `mesh_index` into that array. A merge step
attaches those fields onto an existing MediaPipe timeline without replacing it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MESH_SOURCE = "wilor"


def _as_points(array, width: int) -> np.ndarray:
    pts = np.asarray(array, dtype=np.float64)
    if pts.size == 0:
        return np.zeros((0, width), dtype=np.float64)
    pts = np.reshape(pts, (-1, pts.shape[-1]))
    if pts.shape[1] < width:
        pad = np.zeros((pts.shape[0], width - pts.shape[1]), dtype=np.float64)
        pts = np.concatenate([pts, pad], axis=1)
    return pts[:, :width]


def _fixed21(points: np.ndarray) -> np.ndarray:
    out = np.zeros((21, 3), dtype=np.float64)
    n = min(21, points.shape[0])
    out[:n] = points[:n, :3]
    return out


def _round6(points: np.ndarray) -> list:
    return np.round(points, 6).tolist()


def prediction_to_hand(out: dict, width: int, height: int, mesh_index: int) -> dict | None:
    """Map one WiLoR detection into the PS4 hand record."""
    preds = out.get("wilor_preds")
    if not preds:
        return None
    joints = _fixed21(_as_points(preds["pred_keypoints_3d"], 3))
    pixels = _as_points(preds["pred_keypoints_2d"], 2)
    image = np.zeros((21, 3), dtype=np.float64)
    n = min(21, pixels.shape[0])
    if width > 0 and height > 0 and n:
        image[:n, 0] = pixels[:n, 0] / float(width)
        image[:n, 1] = pixels[:n, 1] / float(height)
    cam_t = _as_points(preds["pred_cam_t_full"], 3).reshape(-1)[:3]
    if cam_t.shape[0] < 3:
        cam_t = np.zeros(3, dtype=np.float64)
    camera = joints + cam_t.reshape(1, 3)
    is_right = float(np.asarray(out.get("is_right", 1)).reshape(-1)[0]) >= 0.5
    verts = _as_points(preds["pred_vertices"], 3)
    return {
        "label": "Right" if is_right else "Left",
        "score": 1.0,
        "tracked": True,
        "interpolated": False,
        "source": MESH_SOURCE,
        "world": _round6(joints),
        "image": _round6(image),
        "camera": _round6(camera),
        "mesh_index": int(mesh_index),
        "vertices": verts.astype(np.float32),
        "cam_t": cam_t.astype(np.float32),
    }


def frame_record(frame: int, t_sec: float, hands: list[dict]) -> dict:
    public = []
    for hand in hands:
        item = {k: v for k, v in hand.items() if k not in {"vertices", "cam_t"}}
        public.append(item)
    return {"frame": frame, "t_sec": round(float(t_sec), 4), "hands": public}


def mesh_view(hand: dict) -> dict:
    return {
        "source": hand.get("source", MESH_SOURCE),
        "world": hand["world"],
        "image": hand["image"],
        "camera": hand.get("camera"),
        "mesh_index": hand.get("mesh_index"),
    }


def merge_frame(base: dict, mesh: dict | None) -> dict:
    """Attach mesh joints onto the matching MediaPipe hand. Extra mesh hands are appended."""
    merged = {
        "frame": base["frame"],
        "t_sec": base["t_sec"],
        "hands": [dict(hand) for hand in base.get("hands", [])],
    }
    if not mesh:
        return merged
    pending = [dict(hand) for hand in mesh.get("hands", [])]
    used: set[int] = set()
    for hand in merged["hands"]:
        for i, candidate in enumerate(pending):
            if i in used or candidate.get("label") != hand.get("label"):
                continue
            hand["mesh"] = mesh_view(candidate)
            used.add(i)
            break
    for i, candidate in enumerate(pending):
        if i not in used:
            merged["hands"].append(candidate)
    return merged


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def merge_jsonl(base_path: Path, mesh_path: Path, out_path: Path) -> int:
    mesh_by_frame = {int(row["frame"]): row for row in read_jsonl(mesh_path)}
    merged = [merge_frame(row, mesh_by_frame.get(int(row["frame"]))) for row in read_jsonl(base_path)]
    write_jsonl(out_path, merged)
    return sum(1 for row in merged for hand in row["hands"] if "mesh" in hand or hand.get("source") == MESH_SOURCE)


def save_mesh_npz(path: Path, chunks: list[dict], faces: np.ndarray | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not chunks:
        np.savez_compressed(path, vertices=np.zeros((0, 0, 3), np.float32), faces=np.zeros((0, 3), np.int32))
        return
    vertices = np.stack([chunk["vertices"] for chunk in chunks]).astype(np.float32)
    joints = np.stack([np.asarray(chunk["world"], dtype=np.float32) for chunk in chunks])
    cam_t = np.stack([chunk["cam_t"] for chunk in chunks]).astype(np.float32)
    frame_ids = np.asarray([chunk["frame"] for chunk in chunks], dtype=np.int32)
    labels = np.asarray([1 if chunk["label"] == "Right" else 0 for chunk in chunks], dtype=np.int8)
    face_arr = np.zeros((0, 3), np.int32) if faces is None else np.asarray(faces, dtype=np.int32)
    np.savez_compressed(
        path,
        vertices=vertices,
        joints=joints,
        cam_t=cam_t,
        frame_ids=frame_ids,
        labels=labels,
        faces=face_arr,
    )


def load_wilor(device: str | None = None, fp16: bool = True):
    """Load WiLoR-mini. Weights download from Hugging Face on first use."""
    try:
        import torch

        # PyTorch 2.6 defaults torch.load to weights_only=True. The WiLoR
        # detector checkpoint (official warmshao/WiLoR-mini weights) needs the
        # previous behavior, which ultralytics 8.1.34 still calls.
        _torch_load = torch.load

        def _load_checkpoint(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _torch_load(*args, **kwargs)

        torch.load = _load_checkpoint
        from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
            WiLorHandPose3dEstimationPipeline,
        )
    except ImportError as exc:
        raise RuntimeError(
            "WiLoR is not installed. On the cloud GPU run: bash scripts/setup_mesh_gpu.sh"
        ) from exc
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    dtype = torch.float16 if fp16 and dev.type == "cuda" else torch.float32
    pipe = WiLorHandPose3dEstimationPipeline(device=dev, dtype=dtype, verbose=False)
    faces = getattr(getattr(pipe, "wilor_model", None), "mano", None)
    face_arr = None if faces is None else np.asarray(faces.faces)
    return pipe, face_arr


def hands_from_predictions(outputs: list[dict], width: int, height: int, index_start: int) -> tuple[list[dict], int]:
    hands = []
    index = index_start
    for out in outputs:
        hand = prediction_to_hand(out, width, height, index)
        if hand is None:
            continue
        hands.append(hand)
        index += 1
    return hands, index
