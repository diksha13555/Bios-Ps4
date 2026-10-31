"""Local PS4 workbench: upload a clip, run the pipeline, then inspect it.

Run from the repo root:

    streamlit run app/workbench.py
"""

from __future__ import annotations

import sys
from html import escape
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hands import CONNECTIONS, DEFAULT_MODEL
from src.pipeline import run_video
from src.scorecard import build_scorecard
from src.workbench_data import (
    action_codes,
    detection_codes,
    discover_runs,
    hand_points,
    hand_rows,
    load_composite_eval,
    load_jsonl,
    load_report,
    max_frames_for_seconds,
    mesh_surfaces,
    present_coordinates,
    present_paths,
    quality_codes,
    quality_grade,
    save_upload,
    summarize_hands,
)

OUTPUT_DIR = ROOT / "outputs"
RAW_DIR = ROOT / "data" / "raw"
LEFT_COLOR = "#4aa3c7"
RIGHT_COLOR = "#d08a3a"


def _read_video_frame(path: str, index: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _clip_fps(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()
    return fps


def _skeleton_figure(frame: dict | None, coord: str, surfaces: list[dict] | None = None):
    fig = plt.figure(figsize=(7.5, 7.5), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("#000000")
    ax.set_facecolor("#000000")
    ax.tick_params(colors="#86868b", labelsize=7)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_color("#86868b")
        axis.pane.set_facecolor("#000000")
        axis.pane.set_edgecolor("#1d1d1f")
    surfaces = surfaces or []
    if (not frame or not frame.get("hands")) and not surfaces:
        ax.set_title("No hands in this frame", color="#86868b", fontsize=10)
        return fig
    clouds = []
    for surface in surfaces:
        verts = np.asarray(surface["vertices"], dtype=np.float64)
        faces = np.asarray(surface["faces"])
        if verts.ndim != 2 or verts.shape[0] < 3 or faces.size == 0:
            continue
        color = LEFT_COLOR if surface.get("label") == "Left" else RIGHT_COLOR
        ax.plot_trisurf(
            verts[:, 0],
            verts[:, 1],
            verts[:, 2],
            triangles=faces,
            color=color,
            alpha=0.55,
            linewidth=0,
            antialiased=False,
            shade=True,
        )
        clouds.append(verts)
    for hand in (frame or {}).get("hands", []):
        if coord == "mesh":
            points = hand_points(hand, "world")
            if points is None:
                continue
            color = LEFT_COLOR if hand.get("label") == "Left" else RIGHT_COLOR
            xs, ys, zs = points[:, 0], points[:, 1], points[:, 2] if points.shape[1] > 2 else np.zeros(len(points))
            clouds.append(points[:, :3])
            for a, b in CONNECTIONS:
                if a < len(points) and b < len(points):
                    ax.plot([xs[a], xs[b]], [ys[a], ys[b]], [zs[a], zs[b]], color=color, linewidth=1.0, alpha=0.45)
            ax.scatter(xs, ys, zs, color=color, s=12, alpha=0.55)
            continue
        points = hand_points(hand, "image" if coord == "image" else coord)
        if points is None:
            continue
        use = points
        if coord == "image":
            use = np.column_stack(
                [points[:, 0], 1.0 - points[:, 1], points[:, 2] if points.shape[1] > 2 else np.zeros(len(points))]
            )
        color = LEFT_COLOR if hand.get("label") == "Left" else RIGHT_COLOR
        xs, ys, zs = use[:, 0], use[:, 1], use[:, 2] if use.shape[1] > 2 else np.zeros(len(use))
        clouds.append(use[:, :3])
        for a, b in CONNECTIONS:
            if a < len(use) and b < len(use):
                ax.plot([xs[a], xs[b]], [ys[a], ys[b]], [zs[a], zs[b]], color=color, linewidth=2.2)
        ax.scatter(xs, ys, zs, color=color, s=28)
    if clouds:
        stacked = np.vstack(clouds)
        center = stacked.mean(axis=0)
        span = max(float(np.ptp(stacked, axis=0).max()), 1e-3) * 0.35
        ax.set_xlim(center[0] - span, center[0] + span)
        ax.set_ylim(center[1] - span, center[1] + span)
        ax.set_zlim(center[2] - span, center[2] + span)
        ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("WiLoR mesh" if coord == "mesh" else coord)
    ax.set_title("WiLoR mesh" if surfaces else coord, color="#f5f5f7", fontsize=10)
    fig.tight_layout()
    return fig


def _timeline_figure(codes: list[int], index: int):
    fig, ax = plt.subplots(figsize=(8.2, 0.55), dpi=120)
    fig.patch.set_facecolor("#000000")
    ax.set_facecolor("#000000")
    if codes:
        colors = {0: "#1d1d1f", 1: LEFT_COLOR, 2: RIGHT_COLOR, 3: "#7dcea0"}
        ax.bar(range(len(codes)), [1] * len(codes), color=[colors[c] for c in codes], width=1.0)
        ax.axvline(index, color="#f2f4f6", linewidth=0.8)
    ax.set_xlim(0, max(len(codes) - 1, 1))
    ax.set_yticks([])
    ax.tick_params(colors="#86868b", labelsize=7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0.2)
    return fig


@st.cache_resource
def _mesh_bundle(path: str) -> dict:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


COORD_HELP = {
    "mesh": "WiLoR MANO surface from the GPU run, same frame and Left/Right.",
    "image": "Where the hand sits in the video.",
    "world": "Hand shape in meters, origin on the wrist. This is not the factory floor.",
    "camera": "That hand placed in front of the camera.",
    "world_fixed": "The hand moving through the clip, camera fixed at frame 0. Translation is up to scale.",
    "world_grounded": "Hand-anchored path: bone scale plus contact pins during grasp/hold. Approximate meters.",
}


def _pct(value) -> str:
    if value is None:
        return "—"
    return f"{100 * float(value):.0f}%"


def _num(value, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}{suffix}"


def _trust_figure(hands: list[dict]):
    fig, ax = plt.subplots(figsize=(4.4, 2.6), dpi=120)
    fig.patch.set_facecolor("#000000")
    ax.set_facecolor("#000000")
    offset = 0
    labeled = False
    for hand in hands:
        confidence = hand.get("joint_confidence") or []
        if not confidence:
            continue
        labeled = True
        color = LEFT_COLOR if hand.get("label") == "Left" else RIGHT_COLOR
        xs = [offset + index for index in range(len(confidence))]
        ax.bar(xs, confidence, color=color, width=0.8)
        offset += len(confidence) + 1
    ax.set_ylim(0, 1)
    ax.set_ylabel("trust", color="#86868b")
    ax.tick_params(colors="#86868b", labelsize=7)
    ax.set_title("Per-joint trust" if labeled else "No per-joint trust on this frame", color="#f5f5f7", fontsize=10)
    for spine in ax.spines.values():
        spine.set_color("#1d1d1f")
    fig.tight_layout()
    return fig


def _quality_parts_figure(hands: list[dict]):
    fig, ax = plt.subplots(figsize=(4.4, 2.6), dpi=120)
    fig.patch.set_facecolor("#000000")
    ax.set_facecolor("#000000")
    labels = ["trust", "bone", "identity", "world", "plausibility"]
    offset = 0
    drawn = False
    for hand in hands:
        parts = hand.get("quality_parts") or {}
        if not parts:
            continue
        drawn = True
        color = LEFT_COLOR if hand.get("label") == "Left" else RIGHT_COLOR
        xs = [offset + i for i in range(len(labels))]
        ax.bar(xs, [parts.get(name, 0) for name in labels], color=color, width=0.75)
        offset += len(labels) + 1
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_ylabel("TouchTrace", color="#86868b")
    ax.set_title("Quality components" if drawn else "No TouchTrace score on this frame", color="#f5f5f7", fontsize=10)
    for spine in ax.spines.values():
        spine.set_color("#1d1d1f")
    fig.tight_layout()
    return fig


def _quality_timeline_figure(codes: list[int], index: int):
    fig, ax = plt.subplots(figsize=(8.2, 0.55), dpi=120)
    fig.patch.set_facecolor("#000000")
    ax.set_facecolor("#000000")
    colors = {0: "#1d1d1f", 1: "#e10600", 2: "#ff6b6b", 3: "#ffb4a8", 4: "#fff5f5"}
    if codes:
        ax.bar(range(len(codes)), [1] * len(codes), color=[colors.get(c, "#1d1d1f") for c in codes], width=1.0)
        ax.axvline(index, color="#f2f4f6", linewidth=0.8)
    ax.set_xlim(0, max(len(codes) - 1, 1))
    ax.set_yticks([])
    ax.tick_params(colors="#86868b", labelsize=7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0.2)
    return fig


def _action_figure(codes: list[int], index: int):
    fig, ax = plt.subplots(figsize=(8.2, 0.55), dpi=120)
    fig.patch.set_facecolor("#000000")
    ax.set_facecolor("#000000")
    colors = {0: "#1d1d1f", 1: "#5c6b73", 2: "#4aa3c7", 3: "#d08a3a", 4: "#7dcea0", 5: "#c76b8a"}
    if codes:
        ax.bar(range(len(codes)), [1] * len(codes), color=[colors.get(code, "#1d1d1f") for code in codes], width=1.0)
        ax.axvline(index, color="#f2f4f6", linewidth=0.8)
    ax.set_xlim(0, max(len(codes) - 1, 1))
    ax.set_yticks([])
    ax.tick_params(colors="#86868b", labelsize=7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0.2)
    return fig


@st.cache_data(show_spinner=False)
def _merged_wilor_gap(path: str) -> dict:
    return build_scorecard(load_jsonl(Path(path)), {})["mediapipe_wilor"]


def _wilor_labels(wilor: dict) -> tuple[str, str | None]:
    aligned = wilor.get("mean_joint_m")
    if aligned is None:
        return "—", None
    label = f"{float(aligned) * 1000:.0f} mm aligned"
    raw = wilor.get("mean_joint_raw_m")
    if raw is not None:
        return label, f"Raw pre-alignment {float(raw) * 1000:.0f} mm · Umeyama scale {_num(wilor.get('mean_scale'), 3)}"
    return f"{float(aligned) * 1000:.0f} mm", None


def _spec_strip(report: dict | None, summary: dict, wilor: dict) -> None:
    report = report or {}
    share = report.get("detection_share") or {}
    accuracy = report.get("accuracy") or {}
    factory = report.get("factory") or {}
    scorecard = report.get("scorecard") or {}
    balance = factory.get("hand_balance") or {}
    if not wilor.get("frames"):
        wilor = (scorecard.get("mediapipe_wilor") or {}) | wilor
    wilor_text, wilor_note = _wilor_labels(wilor)
    swaps = scorecard.get("identity_disagreements")
    touch = report.get("touchtrace") or {}
    grounding = report.get("grounding") or {}
    touch_label = "—"
    if touch.get("overall") is not None:
        touch_label = f"{100 * float(touch['overall']):.0f}% · {touch.get('grade', '?')}"
    raw_drift = (scorecard.get("world_drift") or {}).get("mean_m")
    grounded_drift = (scorecard.get("world_drift_grounded") or {}).get("mean_m")
    items = [
        ("TouchTrace", touch_label),
        ("Either hand", _pct(share.get("either"))),
        ("Both hands", _pct(share.get("both"))),
        ("Confidence", _pct(accuracy.get("mean_detection_confidence"))),
        ("Hands FPS", _num(report.get("fps_hands"), 1)),
        ("Joint trust", _num(summary.get("mean_trust"), 2)),
        ("Bone-length CV", _num(scorecard.get("bone_length_cv"), 3)),
        ("Identity lock", "—" if swaps is None else str(int(swaps))),
        ("WiLoR agreement", wilor_text),
        ("Left active", _pct(balance.get("left_active_share"))),
        ("Right active", _pct(balance.get("right_active_share"))),
        ("Risky wrist", _pct((factory.get("risky_wrist_share") or {}).get("Left"))),
        ("Cycles", "—" if "cycle_count" not in factory else str(int(factory.get("cycle_count") or 0))),
    ]
    if grounding.get("applied"):
        drift_pair = "—" if raw_drift is None and grounded_drift is None else (
            f"{_num(raw_drift)} → {_num(grounded_drift)} m"
        )
        items.append(("Drift after grounding", drift_pair))
        anchors = grounding.get("anchors") or []
        items.append(("Contact anchors", str(len(anchors))))
    cells = "".join(
        f'<div class="xdr-spec"><span>{escape(label)}</span><strong>{escape(value)}</strong></div>'
        for label, value in items
    )
    st.markdown(f'<div class="xdr-specs">{cells}</div>', unsafe_allow_html=True)
    components = touch.get("components") or {}
    if components:
        order = ("trust", "bone", "identity", "world", "plausibility")
        chips = " · ".join(f"{name} {components[name]:.0%}" for name in order if name in components)
        st.caption(f"TouchTrace components: {chips}. Heuristic, not ground-truth accuracy.")
    if summary.get("mean_quality") is not None and touch.get("overall") is None:
        st.caption(
            f"Mean TouchTrace from timeline: {_num(summary['mean_quality'], 2)} "
            f"({quality_grade(summary['mean_quality']) or '—'}). Re-run analysis to refresh the report block."
        )
    if wilor_note:
        st.caption(wilor_note + ". Fair mesh-vs-MediaPipe compare after rigid alignment.")


def _composite_benchmark_panel(output_dir: Path) -> None:
    evaluation = load_composite_eval(output_dir)
    if not evaluation:
        return
    bench = evaluation.get("benchmark") or {}
    detection = evaluation.get("detection") or {}
    _section("Composite detection benchmark", "Real hand on synthetic clutter")
    cols = st.columns(4)
    cols[0].metric("Precision", _pct(detection.get("precision")))
    cols[1].metric("Recall", _pct(detection.get("recall")))
    cols[2].metric("Handedness", _pct((evaluation.get("handedness") or {}).get("accuracy")))
    cols[3].metric("Identity switches", evaluation.get("tracking", {}).get("identity_label_switches", "—"))
    wrist = evaluation.get("wrist_localization") or {}
    if wrist.get("mean_error_norm") is not None:
        st.caption(
            f"Wrist localization {_num(wrist.get('mean_error_norm'), 3)} normalized "
            f"({bench.get('type', 'composite')}). Labels are exact from compositing, not model output."
        )
    if bench.get("note"):
        st.caption(bench["note"])


def _hero_badge(report: dict | None, summary: dict | None) -> str:
    touch = (report or {}).get("touchtrace") or {}
    if touch.get("overall") is not None:
        return f"{touch.get('grade', '?')} · {100 * float(touch['overall']):.0f}%"
    trust = (summary or {}).get("mean_trust")
    if trust is not None:
        return f"{round(100 * float(trust))}% trust"
    return "—"


def _touchtrace_panel(report: dict | None, rows: list[dict]) -> None:
    touch = (report or {}).get("touchtrace") or {}
    summary = summarize_hands(rows)
    _section("TouchTrace quality", "Composite honesty score")
    cols = st.columns(4)
    cols[0].metric("Clip grade", touch.get("grade") or quality_grade(summary.get("mean_quality")) or "—")
    cols[1].metric("Overall", _num(touch.get("overall") or summary.get("mean_quality"), 2))
    cols[2].metric("Segments scored", len(touch.get("segments") or []))
    cols[3].metric("Hands with quality", summary.get("hands") if summary.get("mean_quality") else 0)
    components = touch.get("components") or {}
    if components:
        chips = st.columns(5)
        for column, name in zip(chips, ("trust", "bone", "identity", "world", "plausibility")):
            column.metric(name.capitalize(), _num(components.get(name), 2))
    if touch.get("notes"):
        st.caption(touch["notes"])
    segments = touch.get("segments") or []
    if segments:
        worst = sorted(segments, key=lambda item: float(item.get("quality") or 0))[:12]
        st.dataframe(
            [
                {
                    "label": item.get("label"),
                    "action": item.get("action"),
                    "frames": f"{item.get('start_frame')}–{item.get('end_frame')}",
                    "quality": item.get("quality"),
                    "grade": item.get("grade"),
                }
                for item in worst
            ],
            hide_index=True,
            use_container_width=True,
        )
        st.caption("Lowest-scoring action segments first. Re-film or review these spans.")
    elif summary.get("mean_quality") is None:
        st.info("This run has no TouchTrace scores. Run analysis again with the latest pipeline.")


def _grounding_panel(report: dict | None) -> None:
    grounding = (report or {}).get("grounding") or {}
    scorecard = (report or {}).get("scorecard") or {}
    _section("Contact grounding", "Hand-anchored scale")
    if not grounding.get("applied") and grounding.get("hand_scale") is None:
        st.info(
            grounding.get("note")
            or "Grounding did not run on this clip (no contacts, missing camera path, or --no-ground). "
            "world_grounded stays empty."
        )
        return
    if not grounding.get("applied"):
        st.caption(
            "Bone-based hand scale was estimated, but no grasp/hold anchors pinned the path — "
            "world_grounded is not written for this take."
        )
    cols = st.columns(4)
    cols[0].metric("Hand scale", _num(grounding.get("hand_scale"), 3))
    cols[1].metric("VO scale", _num(grounding.get("vo_scale"), 3))
    cols[2].metric("Anchors", len(grounding.get("anchors") or []))
    raw = (scorecard.get("world_drift") or {}).get("mean_m")
    grounded = (scorecard.get("world_drift_grounded") or {}).get("mean_m")
    cols[3].metric("Drift raw → grounded", "—" if raw is None and grounded is None else f"{_num(raw)} → {_num(grounded)} m")
    per_label = grounding.get("hand_scale_per_label") or {}
    if per_label:
        st.caption("Hand scale per label: " + " · ".join(f"{label} {value}" for label, value in per_label.items()))
    anchors = grounding.get("anchors") or []
    if anchors:
        st.dataframe(
            [
                {
                    "label": item.get("label"),
                    "action": item.get("action"),
                    "frames": f"{item.get('start_frame')}–{item.get('end_frame')}",
                    "duration_s": item.get("duration_sec"),
                }
                for item in anchors
            ],
            hide_index=True,
            use_container_width=True,
        )
    if grounding.get("units"):
        st.caption(grounding["units"] + ". Heuristic, not ground truth.")


def _pitch(report: dict | None, summary: dict, wilor: dict) -> None:
    report = report or {}
    share = report.get("detection_share") or {}
    accuracy = report.get("accuracy") or {}
    factory = report.get("factory") or {}
    scorecard = report.get("scorecard") or {}
    balance = factory.get("hand_balance") or {}
    risky = factory.get("risky_wrist_share") or {}
    smooth = factory.get("smoothness") or {}
    drift = (scorecard.get("world_drift") or {}).get("mean_m")
    bone = scorecard.get("bone_length_cv")
    swaps = scorecard.get("identity_disagreements")
    if not wilor.get("frames"):
        wilor = (scorecard.get("mediapipe_wilor") or {}) | wilor
    wilor_text, wilor_note = _wilor_labels(wilor)

    tracking = st.columns(4)
    tracking[0].metric("Either hand", _pct(share.get("either")))
    tracking[1].metric("Both hands", _pct(share.get("both")))
    tracking[2].metric("Mean confidence", _pct(accuracy.get("mean_detection_confidence")))
    tracking[3].metric("Hands FPS", _num(report.get("fps_hands"), 1))

    trust = st.columns(4)
    trust[0].metric("Bone-length CV", _num(bone, 3))
    trust[1].metric("Identity corrections", "—" if swaps is None else str(int(swaps)))
    aligned_mm = wilor.get("mean_joint_m")
    trust[2].metric(
        "WiLoR aligned",
        "—" if aligned_mm is None else f"{float(aligned_mm) * 1000:.0f} mm",
    )
    trust[3].metric("Mean joint trust", _num(summary.get("mean_trust"), 2))
    if wilor_note:
        st.caption(wilor_note)

    factory_row = st.columns(4)
    factory_row[0].metric("Left of active time", _pct(balance.get("left_active_share")))
    factory_row[1].metric("Right of active time", _pct(balance.get("right_active_share")))
    left_risk, right_risk = risky.get("Left"), risky.get("Right")
    factory_row[2].metric(
        "Risky wrist",
        "—" if left_risk is None and right_risk is None else f"{_pct(left_risk)} / {_pct(right_risk)}",
    )
    factory_row[3].metric("Cycles", "—" if "cycle_count" not in factory else str(int(factory.get("cycle_count") or 0)))
    trusted = factory.get("trusted_filter") or {}
    if trusted:
        st.caption(
            f"Factory stats exclude inferred/low-trust frames "
            f"({trusted.get('excluded_hands', 0)} hands, "
            f"{_pct(trusted.get('excluded_share'))} of hand-frames). "
            f"{trusted.get('note', '')}"
        )

    if accuracy.get("pitch_lead"):
        st.caption(accuracy["pitch_lead"])
    raw = (accuracy.get("coverage_raw") or {}).get("either")
    filled = (accuracy.get("coverage_with_hold_and_fill") or {}).get("either")
    notes = []
    if raw is not None and filled is not None:
        notes.append(f"Coverage {_pct(raw)} before gap fill, {_pct(filled)} after. Gaps longer than 5 frames stay empty.")
    if summary.get("inferred"):
        notes.append(f"{summary['inferred']} filled hands are marked inferred and capped at trust 0.35.")
    if drift is not None:
        notes.append(f"Put-down drift averages {_num(drift)} m. Monocular translation is up to scale.")
    if smooth:
        notes.append(
            f"Smoothness {_num(smooth.get('Left'), 2)} left / {_num(smooth.get('Right'), 2)} right. "
            "Wrist jerk in the up-to-scale world frame drives this toward zero."
        )
    if factory.get("rula"):
        notes.append("Risky wrist is the RULA wrist row only, from the metacarpal direction, not a full-body score.")
    if notes:
        st.caption(" ".join(notes))

    actions = summary.get("actions") or {}
    if actions:
        order = ["idle", "reach", "grasp", "hold", "release"]
        chips = [f"{name} {actions[name]}" for name in order if actions.get(name)]
        grasps = summary.get("grasps") or {}
        grasp_chips = [f"{name} {grasps[name]}" for name in ("power", "precision", "lateral", "other") if grasps.get(name)]
        st.caption("Actions: " + " · ".join(chips) + (". Grasps: " + " · ".join(grasp_chips) if grasp_chips else ""))

    with st.expander("What the judges are looking at", expanded=False):
        st.markdown(
            """
**Tracking.** 21 MediaPipe points. Left and Right stay locked by overlap, so a raw label flip does not swap the hands.

**Trust.** Each joint is 0–1 from the detection score, reprojection error, jitter, and disagreement with WiLoR. Low-trust joints are smoothed harder and pulled back onto a stable bone length. That is the blur and glove cleanup.

**Actions.** Idle, reach, grasp, hold, release. Grasp/release only fire on continuously tracked frames — not after gaps or on inferred fill.

**Factory.** Cycle time is reach, then grasp, then hold, then release. Active share, risky wrist, and smoothness use trusted (non-inferred) frames only. The wrist number is one RULA row.

**WiLoR agreement.** Procrustes-aligned joint error (raw pre-alignment error shown separately when available).

**Coordinates.** Image is the video. World is the hand shape in meters. Camera is that hand in front of the lens. Fixed world is the path through the clip. Grounded world is the contact-anchored path in approximate meters.

**TouchTrace.** One 0–1 quality score and A–D grade per hand-frame and segment.

**Mesh.** The shaded hand is the WiLoR surface from the GPU run.

**Sim export.** The JSON and CSV carry joint trust and the action, so a robot can ignore shaky frames.

**Scorecard.** No ground truth. Bone-length stability, identity corrections, MediaPipe versus WiLoR, and the gap between a release and the next grasp.
            """
        )


def _run_upload(uploaded, seconds: float) -> str:
    if not DEFAULT_MODEL.is_file():
        raise FileNotFoundError("MediaPipe model is missing. Run bash scripts/download_models.sh first.")
    dest = save_upload(RAW_DIR, uploaded.name, uploaded.getvalue())
    fps = _clip_fps(dest)
    paths = run_video(dest, OUTPUT_DIR, max_frames=max_frames_for_seconds(fps, seconds))
    return Path(paths["jsonl"]).stem if paths.get("jsonl") else dest.stem


def _section(title: str, detail: str = "") -> None:
    st.markdown(
        f'<div class="tt-section"><h2>{escape(title)}</h2><span>{escape(detail)}</span></div>',
        unsafe_allow_html=True,
    )


def _show_figure(fig) -> None:
    try:
        st.pyplot(fig, clear_figure=True, use_container_width=True)
    finally:
        plt.close(fig)


def _sync_coordinate_choice(run_id: str, present: list[str], *, has_mesh: bool) -> None:
    """Prefer WiLoR mesh when available; reset when the processed run changes."""
    coord_key = f"coordinates-{run_id}"
    if st.session_state.get("inspect_run_id") != run_id:
        st.session_state["inspect_run_id"] = run_id
        if has_mesh and "mesh" in present:
            st.session_state[coord_key] = "mesh"
    if st.session_state.get(coord_key) not in present:
        st.session_state[coord_key] = present[0]


def _inspect(paths: dict, rows: list[dict], run_id: str, report: dict | None = None) -> None:
    _section("The MotionTruth studio", "21 landmarks")
    if not rows:
        st.info("No landmark timeline is available for this run. Run analysis to create the JSONL, or choose another processed run.")
        return
    start = next((i for i, row in enumerate(rows) if row.get("hands")), 0)
    frame_key = f"frame-{run_id}"
    if frame_key in st.session_state and st.session_state[frame_key] >= len(rows):
        st.session_state[frame_key] = start
    index = st.slider("Frame", 0, len(rows) - 1, start, key=frame_key) if len(rows) > 1 else 0
    frame = rows[index]
    frame_id = int(frame.get("frame", index))
    st.caption(f"Frame {frame_id} · {_num(frame.get('t_sec'), 3)} s · {len(frame.get('hands', []))} hand(s)")
    has_mesh = paths.get("mesh_npz") is not None
    present = present_coordinates(frame, mesh=has_mesh)
    _sync_coordinate_choice(run_id, present, has_mesh=has_mesh)
    coord_key = f"coordinates-{run_id}"
    coord = st.radio("Coordinate space", present, horizontal=True, key=coord_key)
    if has_mesh and coord != "mesh":
        st.caption("WiLoR mesh is available for this take — select **mesh** to view the MANO surface.")
    st.caption(COORD_HELP.get(coord, ""))
    surfaces = []
    if coord == "mesh":
        try:
            surfaces = mesh_surfaces(_mesh_bundle(str(paths["mesh_npz"])), frame_id)
        except (OSError, ValueError, KeyError) as exc:
            st.warning(f"Could not read the mesh for this run: {exc}")
    stage, context = st.columns([2, 1], gap="large")
    with stage:
        _show_figure(_skeleton_figure(frame, coord, surfaces))
        st.caption("Left · teal    /    Right · amber")
        if coord == "mesh":
            if surfaces:
                st.caption(f"MANO surface · {len(surfaces)} hand(s) · {len(surfaces[0]['vertices'])} vertices per first hand")
            else:
                st.info("No mesh at this frame. Move the frame slider to inspect another moment.")
    with context:
        st.markdown("### Frame context")
        if paths.get("video") is not None:
            image = _read_video_frame(str(paths["video"]), frame_id)
            if image is not None:
                st.image(image, caption="Side-by-side video at this frame", use_container_width=True)
            else:
                st.caption("The preview frame could not be decoded.")
        else:
            st.caption("No video preview is available for this run.")
        if has_mesh:
            mesh_here = bool(surfaces) if coord == "mesh" else None
            if coord == "mesh":
                st.caption(
                    "MANO mesh at this frame."
                    if mesh_here
                    else "No mesh vertices at this frame — scrub to a frame with hands tracked in the WiLoR pass."
                )
        for hand in frame.get("hands", []):
            bits = [
                hand.get("label", "Hand"),
                hand.get("action") or "Unlabelled",
            ]
            if hand.get("grasp"):
                bits.append(hand["grasp"])
            if hand.get("inferred"):
                bits.append("inferred · actions frozen")
            if hand.get("quality") is not None:
                bits.append(f"TouchTrace {hand['quality']:.2f}")
            st.text(" · ".join(bits))
        if not frame.get("hands"):
            st.caption("No hand measurements at this frame.")
        _show_figure(_trust_figure(frame.get("hands", [])))
        st.caption("Joint trust is a heuristic, not measured accuracy. Inferred joints are capped at 0.35.")
        if any(h.get("quality_parts") for h in frame.get("hands", [])):
            with st.expander("TouchTrace on this frame", expanded=False):
                _show_figure(_quality_parts_figure(frame.get("hands", [])))
    with st.expander("Detection and action timelines", expanded=False):
        _show_figure(_timeline_figure(detection_codes(rows), index))
        st.caption("Hands: teal left · amber right · green both · gray none. White marks the selected frame.")
        codes = action_codes(rows)
        if any(codes):
            _show_figure(_action_figure(codes, index))
            st.caption("Actions: gray idle · teal reach · amber grasp · green hold · pink release.")
        qcodes = quality_codes(rows)
        if any(qcodes):
            _show_figure(_quality_timeline_figure(qcodes, index))
            st.caption("TouchTrace grades: dark D · red C · salmon B · pale A.")
    segments = (report or {}).get("touchtrace", {}).get("segments") or []
    if segments and not st.session_state.demo:
        options = {
            f"{item['label']} {item['action']} f{item['start_frame']}–{item['end_frame']} ({item.get('grade', '?')})": int(
                item["start_frame"]
            )
            for item in sorted(segments, key=lambda row: float(row.get("quality") or 0))[:8]
        }
        pick, go = st.columns([4, 1])
        with pick:
            jump = st.selectbox("Low-scoring segment", ["—", *options.keys()], key=f"jump-{run_id}")
        with go:
            st.write("")
            if st.button("Go", key=f"jump-go-{run_id}") and jump in options:
                target = options[jump]
                st.session_state[frame_key] = next(
                    (i for i, row in enumerate(rows) if int(row.get("frame", i)) >= target),
                    index,
                )
                st.rerun()
    if not st.session_state.demo:
        with st.expander("Landmark measurements", expanded=False):
            table_coord = coord if coord != "mesh" else "world_grounded"
            if table_coord not in present:
                table_coord = "world" if "world" in present else coord
            st.dataframe(hand_rows(frame, table_coord), hide_index=True, use_container_width=True)


def _exports(paths: dict, run_id: str) -> None:
    _section("Take the next step.", "Local files · Original data")
    st.caption(
        "Downloads include joint trust, TouchTrace quality, and grounded wrist columns when the run used the latest pipeline."
    )
    files = [
        ("Report", "report", "Operational metrics and scorecard", "application/json"),
        ("JSONL", "jsonl", "Per-frame hand measurements", "application/x-ndjson"),
        ("World CSV", "world_csv", "Tabular wrist trajectories", "text/csv"),
        ("World JSON", "world_sim", "Structured simulator export", "application/json"),
    ]
    for label, key, description, mime in files:
        text, action = st.columns([3, 1])
        with text:
            st.markdown(f"### {label}")
            st.caption(description)
        with action:
            path = paths.get(key)
            if path is not None:
                st.download_button(f"Download {label}", path.read_bytes(), file_name=path.name,
                                   mime=mime, key=f"dl-{label}-{run_id}", use_container_width=True)
            else:
                st.caption("Not available for this run")
        st.divider()


def _dot_hand(pattern: str, transform: str) -> str:
    """A pointing hand drawn as dots. Original geometry, not a copied image."""
    return f'''
    <g transform="{transform}" fill="url(#{pattern})">
      <rect x="8" y="96" width="54" height="36" rx="16"/>
      <ellipse cx="92" cy="118" rx="62" ry="42"/>
      <rect x="138" y="86" width="200" height="20" rx="10"/>
      <rect x="128" y="36" width="20" height="72" rx="10"/>
      <rect x="100" y="46" width="18" height="64" rx="9"/>
      <rect x="74" y="58" width="16" height="52" rx="8"/>
      <ellipse cx="62" cy="156" rx="26" ry="14" transform="rotate(-36 62 156)"/>
    </g>'''


def _scene(trust: str) -> str:
    cubes = "".join(f'<i style="--i:{index}"></i>' for index in range(9))
    return f'''
    <section class="mt-scene">
      <header class="mt-top">
        <a class="mt-brand" href="#studio">TouchTrace</a>
        <nav class="mt-pills">
          <a href="#studio">MotionTruth</a>
          <a href="#studio">Studio</a>
          <a href="#studio">GhostHand</a>
          <a href="#report">Report</a>
        </nav>
        <a class="mt-open" href="#studio">Open studio</a>
      </header>
      <h1>When intelligence reaches out to instinct, the motion comes with proof</h1>
      <p class="mt-sub">an unlikely alliance: human movement and machine learning, with every frame accountable</p>
      <div class="mt-stage">
        <svg class="mt-hand mt-hand-left" viewBox="0 0 360 210" aria-hidden="true">
          <defs>
            <pattern id="ash" width="8" height="8" patternUnits="userSpaceOnUse">
              <circle cx="2.2" cy="2.2" r="1.45" fill="#f4f4f5"/>
            </pattern>
            <linearGradient id="wash" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0" stop-color="white" stop-opacity="0"/>
              <stop offset="0.18" stop-color="white"/>
            </linearGradient>
            <mask id="in"><rect width="360" height="210" fill="url(#wash)"/></mask>
          </defs>
          <g mask="url(#in)">{_dot_hand("ash", "")}</g>
        </svg>
        <div class="mt-core">
          <div class="mt-glow"></div>
          <div class="mt-cubes">{cubes}</div>
          <a class="mt-cta" href="#studio">See it in action →</a>
        </div>
        <svg class="mt-hand mt-hand-right" viewBox="0 0 360 210" aria-hidden="true">
          <defs>
            <pattern id="mix" width="9" height="9" patternUnits="userSpaceOnUse">
              <circle cx="2" cy="2" r="1.35" fill="#f4f4f5"/>
              <circle cx="6.5" cy="5.5" r="1.25" fill="#34d399"/>
              <circle cx="3" cy="7" r="1.15" fill="#f59e0b"/>
            </pattern>
          </defs>
          {_dot_hand("mix", "translate(360 210) scale(-0.72)")}
        </svg>
        <aside class="mt-flag">
          <strong>{escape(trust)}</strong>
          <span><i></i>index tip uncertain · flagged</span>
        </aside>
      </div>
      <p class="mt-scroll">Scroll to bring them together</p>
    </section>'''


def main() -> None:
    st.set_page_config(page_title="TouchTrace", layout="wide", initial_sidebar_state="collapsed")
    st.markdown(f"<style>{Path(__file__).with_suffix('.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    st.session_state.setdefault("demo", False)
    runs = discover_runs(OUTPUT_DIR)
    if not runs:
        run = None
    else:
        ids = [item["id"] for item in runs]
        current = st.session_state.get("run")
        if current not in ids:
            current = ids[0]
        run = runs[ids.index(current)]
    paths = report = rows = summary = None
    if run is not None:
        try:
            paths = present_paths(OUTPUT_DIR, run["folder"], run["stem"])
            report = load_report(paths["report"])
            rows = load_jsonl(paths["jsonl"])
            summary = summarize_hands(rows)
        except (OSError, ValueError) as exc:
            st.error(f"This run could not be loaded: {exc}. Choose another run or regenerate its results.")
            return
    st.markdown(_scene(_hero_badge(report, summary)), unsafe_allow_html=True)
    with st.expander("Open studio", expanded=False):
        uploaded = st.file_uploader("Source video", type=["mp4", "mov", "avi", "mkv"])
        seconds = st.number_input("Seconds to process", min_value=0, max_value=600, value=50, step=10,
                                  help="Set to 0 to process the entire clip.")
        if st.button("Run analysis", type="primary"):
            if uploaded is None:
                st.session_state.run_error = "Choose a source video before running analysis."
            else:
                with st.spinner("Tracking hands and preparing exports. Keep this tab open."):
                    try:
                        st.session_state.run = _run_upload(uploaded, float(seconds))
                        st.session_state.run_error = None
                    except Exception as exc:
                        st.session_state.run_error = str(exc)
                st.rerun()
        if st.session_state.get("run_error"):
            st.error(st.session_state.run_error)
        st.caption("Processed locally. Your footage stays on this machine.")
    if not runs:
        st.caption("Your processed clips will appear here after a run.")
    else:
        selected = st.selectbox("Processed run", ids, index=ids.index(run["id"]))
        st.session_state.run = selected
        if selected != run["id"]:
            st.session_state.pop("inspect_run_id", None)
            for key in list(st.session_state):
                if key.startswith("coordinates-"):
                    del st.session_state[key]
            st.rerun()
    if run is None:
        st.markdown('<div class="tt-empty"><h2>Your first take starts here.</h2>'
                    '<p>Open studio, choose a video, and run analysis.</p></div>', unsafe_allow_html=True)
        return
    mesh_runs = [item for item in runs if item.get("mesh")]
    if paths.get("mesh_npz") is None:
        hint = (
            f"WiLoR mesh is on **{mesh_runs[0]['id']}**. Pick that run and set Coordinate space to **mesh**."
            if mesh_runs
            else "No WiLoR `.npz` in `outputs/`. Run `scripts/run_mesh.py` on this clip, then refresh."
        )
        st.warning(hint)
    wilor = (report or {}).get("scorecard", {}).get("mediapipe_wilor") or {}
    if not wilor.get("frames") and paths.get("merged") is not None:
        try:
            wilor = _merged_wilor_gap(str(paths["merged"]))
        except (OSError, ValueError, KeyError):
            st.warning("The merged mesh timeline could not be read. Other run results remain available.")
    st.markdown(f'<div class="tt-context"><strong>{escape(run["id"])}</strong>'
                f'<span>{len(rows):,} timeline frames · Local workspace</span></div>', unsafe_allow_html=True)
    if paths["video"] is not None:
        st.markdown('<div class="xdr-stage">', unsafe_allow_html=True)
        st.video(str(paths["video"]))
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No video preview for this run. Available measurements can still be inspected below.")
    _spec_strip(report, summary, wilor)
    st.markdown(
        '<p class="xdr-note">Coverage counts a hand only when it is visible. Gaps longer than five frames stay empty. '
        "Fixed-world translation is up to scale. Risky wrist is the RULA wrist row, not a full-body score.</p>",
        unsafe_allow_html=True,
    )
    if st.session_state.demo:
        _inspect(paths, rows, run["id"], report)
    else:
        st.markdown('<div id="studio"></div>', unsafe_allow_html=True)
        inspect, quality, exports = st.tabs(["Inspect", "Quality", "Export"])
        with inspect:
            _inspect(paths, rows, run["id"], report)
        with quality:
            _touchtrace_panel(report, rows)
            _grounding_panel(report)
            _composite_benchmark_panel(OUTPUT_DIR)
            st.divider()
            _section("Operational diagnostics", "Tracking and factory")
            st.caption("These indicators describe model output, not ground-truth accuracy. Missing measurements are shown as —.")
            _pitch(report, summary, wilor)
        st.markdown('<div id="report"></div>', unsafe_allow_html=True)
        with exports:
            _exports(paths, run["id"])
    st.markdown('<div class="tt-footer"><span>TouchTrace</span><span>Every motion has proof.</span></div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
