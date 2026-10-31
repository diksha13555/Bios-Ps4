"""Load PS4 outputs for the local workbench. No Streamlit imports."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

COORD_KEYS = ("world", "camera", "world_fixed")


def safe_stem(filename: str) -> str:
    """Turn an uploaded filename into a single path segment."""
    stem = Path(Path(filename).name).stem
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem).strip("_")
    return (cleaned[:80] or "clip")


def save_upload(raw_dir: Path, filename: str, data: bytes) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / f"{safe_stem(filename)}.mp4"
    dest.write_bytes(data)
    return dest


def max_frames_for_seconds(fps: float, seconds: float) -> int | None:
    """None means the whole clip. A positive duration is converted with the file fps."""
    if seconds <= 0:
        return None
    rate = fps if fps and fps > 1 else 30.0
    return max(1, int(round(seconds * rate)))


def output_dirs(output_dir: Path) -> list[Path]:
    """The output root plus one level of result folders."""
    if not output_dir.is_dir():
        return []
    nested = [path for path in output_dir.iterdir() if path.is_dir() and not path.name.startswith(".")]
    return [output_dir, *sorted(nested)]


def discover_clips(output_dir: Path) -> list[str]:
    """Stems that have a side-by-side video, a report, a timeline, or a WiLoR mesh."""
    stems: set[str] = set()
    for folder in output_dirs(output_dir):
        for path in folder.iterdir():
            if not path.is_file():
                continue
            name = path.name
            if name.endswith("_sidebyside.mp4"):
                stems.add(name[: -len("_sidebyside.mp4")])
            elif name.endswith("_report.json"):
                stems.add(name[: -len("_report.json")])
            elif name.endswith("_mesh.npz"):
                stems.add(name[: -len("_mesh.npz")])
            elif name.endswith(".jsonl") and not name.endswith("_mesh.jsonl") and not name.endswith("_merged.jsonl"):
                stems.add(name[: -len(".jsonl")])
    return sorted(stem for stem in stems if not stem.startswith("_"))


def clip_paths(output_dir: Path, stem: str) -> dict[str, Path | None]:
    def pick(*names: str) -> Path | None:
        for folder in output_dirs(output_dir):
            for name in names:
                path = folder / name
                if path.is_file():
                    return path
        return None

    return {
        "video": pick(f"{stem}_sidebyside.mp4"),
        "jsonl": pick(f"{stem}.jsonl", f"{stem}_merged.jsonl", f"{stem}_mesh.jsonl"),
        "report": pick(f"{stem}_report.json"),
        "sim": pick(f"{stem}_sim.json"),
        "world_sim": pick(f"{stem}_world_sim.json"),
        "world_csv": pick(f"{stem}_world_sim.csv"),
        "merged": pick(f"{stem}_merged.jsonl"),
        "mesh_npz": pick(f"{stem}_mesh.npz"),
    }


def mesh_surfaces(bundle: dict, frame_id: int) -> list[dict]:
    """WiLoR vertices for one frame. labels are 0 left, 1 right."""
    frame_ids = np.asarray(bundle.get("frame_ids", []))
    if frame_ids.size == 0:
        return []
    vertices = np.asarray(bundle["vertices"])
    labels = np.asarray(bundle.get("labels", []))
    faces = np.asarray(bundle.get("faces", []))
    surfaces = []
    for index in np.where(frame_ids == int(frame_id))[0]:
        label = "Right" if labels.size and int(labels[index]) == 1 else "Left"
        surfaces.append({"label": label, "vertices": vertices[index], "faces": faces})
    return surfaces


def load_report(path: Path | None) -> dict | None:
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path | None) -> list[dict]:
    if path is None or not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def detection_codes(rows: list[dict]) -> list[int]:
    """0 none, 1 left, 2 right, 3 both. Interpolated hands still count as present."""
    codes = []
    for row in rows:
        labels = {hand.get("label") for hand in row.get("hands", [])}
        left = "Left" in labels
        right = "Right" in labels
        codes.append((1 if left else 0) + (2 if right else 0))
    return codes


def hand_points(hand: dict, key: str) -> np.ndarray | None:
    raw = hand.get(key)
    if raw is None and key == "world":
        raw = hand.get("mesh", {}).get("world") if isinstance(hand.get("mesh"), dict) else None
    if not raw:
        return None
    points = np.asarray(raw, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
        return None
    return points


def hand_rows(frame: dict | None, coord: str) -> list[dict]:
    if not frame:
        return []
    rows = []
    for hand in frame.get("hands", []):
        points = hand_points(hand, coord)
        trust = hand.get("joint_confidence")
        rows.append(
            {
                "label": hand.get("label", ""),
                "score": hand.get("score"),
                "action": hand.get("action"),
                "grasp": hand.get("grasp"),
                "inferred": hand.get("inferred"),
                "mean_trust": None if not trust else round(sum(trust) / len(trust), 3),
                "tracked": hand.get("tracked"),
                "interpolated": hand.get("interpolated"),
                "source": hand.get("source", "mediapipe"),
                "coord": coord if points is not None else "missing",
                "joints": 0 if points is None else int(points.shape[0]),
                "reproj_error_px": hand.get("reproj_error_px"),
                "mesh": "mesh" in hand,
            }
        )
    return rows
