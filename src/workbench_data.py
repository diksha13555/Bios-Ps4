"""Load PS4 outputs for the local workbench. No Streamlit imports."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

COORD_KEYS = ("world_grounded", "world", "camera", "world_fixed")


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


def _stems_in(folder: Path) -> list[str]:
    stems: set[str] = set()
    if not folder.is_dir():
        return []
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
    return sorted(stem for stem in stems if stem and not stem.startswith("_"))


def discover_runs(output_dir: Path) -> list[dict]:
    """One entry per result folder. Smoke runs stay out of the judge list."""
    runs = []
    for folder in output_dirs(output_dir):
        if "smoke" in folder.name or folder.name.startswith("."):
            continue
        for stem in _stems_in(folder):
            if "smoke" in stem or stem.startswith("ui_upload"):
                continue
            run_id = stem if folder == output_dir else f"{folder.name}/{stem}"
            report = folder / f"{stem}_report.json"
            body = report.read_text(encoding="utf-8") if report.is_file() else ""
            has_scorecard = '"scorecard"' in body or '"touchtrace"' in body
            has_video = (folder / f"{stem}_sidebyside.mp4").is_file()
            mesh = (folder / f"{stem}_mesh.npz").is_file() or related_file(output_dir, stem, "_mesh.npz") is not None
            runs.append(
                {
                    "id": run_id,
                    "stem": stem,
                    "folder": folder,
                    "scorecard": has_scorecard,
                    "video": has_video,
                    "mesh": mesh,
                }
            )
    runs.sort(
        key=lambda run: (
            0 if run["mesh"] else 1,
            0 if run["scorecard"] else 1,
            0 if run["video"] else 1,
            run["id"],
        )
    )
    return runs


def _same_take(left: str, right: str) -> int | None:
    """0 when the names match, 1 when one is the other plus a - or _ suffix."""
    if left == right:
        return 0
    if left.startswith(right + "-") or left.startswith(right + "_"):
        return 1
    if right.startswith(left + "-") or right.startswith(left + "_"):
        return 1
    return None


def related_file(output_dir: Path, stem: str, suffix: str) -> Path | None:
    """A file from the same take in another result folder. Scorecard reports win ties."""
    matches = []
    for folder in output_dirs(output_dir):
        if not folder.is_dir():
            continue
        for path in folder.iterdir():
            if not path.is_file() or not path.name.endswith(suffix):
                continue
            other = path.name[: -len(suffix)]
            rank = _same_take(stem, other)
            if rank is None:
                continue
            report = folder / f"{other}_report.json"
            scorecard = report.is_file() and '"scorecard"' in report.read_text(encoding="utf-8")
            matches.append((rank, 0 if scorecard else 1, -len(other), str(path)))
    if not matches:
        return None
    matches.sort()
    return Path(matches[0][3])


def clip_paths(output_dir: Path, stem: str, folder: Path | None = None) -> dict[str, Path | None]:
    def pick(*names: str) -> Path | None:
        folders = [folder] if folder is not None else output_dirs(output_dir)
        for candidate in folders:
            for name in names:
                path = candidate / name
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


def present_paths(output_dir: Path, folder: Path, stem: str) -> dict[str, Path | None]:
    """Files for this run, borrowing a mesh, video, or scorecard from the same take."""
    paths = clip_paths(output_dir, stem, folder=folder)
    fillers = {
        "video": "_sidebyside.mp4",
        "report": "_report.json",
        "mesh_npz": "_mesh.npz",
        "merged": "_merged.jsonl",
        "sim": "_sim.json",
        "world_sim": "_world_sim.json",
        "world_csv": "_world_sim.csv",
    }
    for key, suffix in fillers.items():
        if paths.get(key) is None:
            paths[key] = related_file(output_dir, stem, suffix)
    return paths


def summarize_hands(rows: list[dict]) -> dict:
    """Action counts and mean joint trust, read straight from the timeline."""
    inferred = 0
    hands = 0
    trust_total = 0.0
    trust_count = 0
    quality_total = 0.0
    quality_count = 0
    grounded = 0
    actions: dict[str, int] = {}
    grasps: dict[str, int] = {}
    for row in rows:
        for hand in row.get("hands", []):
            hands += 1
            if hand.get("inferred"):
                inferred += 1
            confidence = hand.get("joint_confidence")
            if confidence:
                trust_total += sum(confidence) / len(confidence)
                trust_count += 1
            if hand.get("world_grounded"):
                grounded += 1
            if hand.get("quality") is not None:
                quality_total += float(hand["quality"])
                quality_count += 1
            action = hand.get("action")
            if action:
                actions[action] = actions.get(action, 0) + 1
            grasp = hand.get("grasp")
            if grasp:
                grasps[grasp] = grasps.get(grasp, 0) + 1
    return {
        "hands": hands,
        "inferred": inferred,
        "mean_trust": None if trust_count == 0 else round(trust_total / trust_count, 3),
        "mean_quality": None if quality_count == 0 else round(quality_total / quality_count, 3),
        "grounded_hands": grounded,
        "actions": actions,
        "grasps": grasps,
    }


def quality_grade(score: float | None) -> str | None:
    if score is None:
        return None
    if score >= 0.85:
        return "A"
    if score >= 0.70:
        return "B"
    if score >= 0.55:
        return "C"
    return "D"


def quality_codes(rows: list[dict]) -> list[int]:
    """0 none, 1 D, 2 C, 3 B, 4 A (worst hand on the frame)."""
    codes = []
    for row in rows:
        scores = [float(h["quality"]) for h in row.get("hands", []) if h.get("quality") is not None]
        if not scores:
            codes.append(0)
            continue
        grade = quality_grade(min(scores))
        codes.append({"D": 1, "C": 2, "B": 3, "A": 4}.get(grade or "D", 1))
    return codes


def action_codes(rows: list[dict]) -> list[int]:
    """0 none, 1 idle, 2 reach, 3 grasp, 4 hold, 5 release."""
    order = {"idle": 1, "reach": 2, "grasp": 3, "hold": 4, "release": 5}
    codes = []
    for row in rows:
        names = [hand.get("action") for hand in row.get("hands", []) if hand.get("action")]
        codes.append(order.get(names[0], 0) if names else 0)
    return codes


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


def load_composite_eval(output_dir: Path) -> dict | None:
    """Composite detection benchmark from scripts/benchmark_composite.py."""
    path = output_dir / "benchmark" / "composite_eval.json"
    if not path.is_file():
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
                "quality": hand.get("quality"),
                "quality_grade": quality_grade(hand.get("quality")),
                "grounded": hand.get("world_grounded") is not None,
            }
        )
    return rows


COORD_PREFERENCE = ("mesh", "world_grounded", "world", "camera", "world_fixed", "image")


def present_coordinates(frame: dict | None, *, mesh: bool) -> list[str]:
    """Coordinate modes available on this frame, in dashboard display order."""
    if not frame:
        return ["world"]
    available = {"image"}
    if mesh:
        available.add("mesh")
    for key in COORD_KEYS:
        if any(hand_points(h, key) is not None for h in frame.get("hands", [])):
            available.add(key)
    return [key for key in COORD_PREFERENCE if key in available]
