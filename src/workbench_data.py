"""Load PS4 outputs for the local workbench. No Streamlit imports."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.hands import CONNECTIONS

COORD_KEYS = ("world", "camera", "world_fixed")


def discover_clips(output_dir: Path) -> list[str]:
    """Stems that have at least a side-by-side video, a report, or a timeline."""
    if not output_dir.is_dir():
        return []
    stems: set[str] = set()
    for path in output_dir.iterdir():
        if not path.is_file():
            continue
        name = path.name
        if name.endswith("_sidebyside.mp4"):
            stems.add(name[: -len("_sidebyside.mp4")])
        elif name.endswith("_report.json"):
            stems.add(name[: -len("_report.json")])
        elif name.endswith(".jsonl") and not name.endswith("_mesh.jsonl") and not name.endswith("_merged.jsonl"):
            stems.add(name[: -len(".jsonl")])
    return sorted(stem for stem in stems if not stem.startswith("_"))


def clip_paths(output_dir: Path, stem: str) -> dict[str, Path | None]:
    def pick(*names: str) -> Path | None:
        for name in names:
            path = output_dir / name
            if path.is_file():
                return path
        return None

    return {
        "video": pick(f"{stem}_sidebyside.mp4"),
        "jsonl": pick(f"{stem}.jsonl"),
        "report": pick(f"{stem}_report.json"),
        "sim": pick(f"{stem}_sim.json"),
        "world_sim": pick(f"{stem}_world_sim.json"),
        "world_csv": pick(f"{stem}_world_sim.csv"),
        "merged": pick(f"{stem}_merged.jsonl"),
        "mesh_npz": pick(f"{stem}_mesh.npz"),
    }


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
        rows.append(
            {
                "label": hand.get("label", ""),
                "score": hand.get("score"),
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
