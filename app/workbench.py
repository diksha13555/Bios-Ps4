"""Local PS4 workbench: upload a clip, run the pipeline, then inspect it.

Run from the repo root:

    streamlit run app/workbench.py
"""

from __future__ import annotations

import sys
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
from src.workbench_data import (
    COORD_KEYS,
    clip_paths,
    detection_codes,
    discover_clips,
    hand_points,
    hand_rows,
    load_jsonl,
    load_report,
    max_frames_for_seconds,
    save_upload,
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


def _skeleton_figure(frame: dict | None, coord: str):
    fig = plt.figure(figsize=(4.2, 4.2), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("#121418")
    ax.set_facecolor("#121418")
    ax.tick_params(colors="#9aa3ad", labelsize=7)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_color("#9aa3ad")
        axis.pane.set_facecolor("#121418")
        axis.pane.set_edgecolor("#2a3038")
    if not frame or not frame.get("hands"):
        ax.set_title("No hands in this frame", color="#9aa3ad", fontsize=10)
        return fig
    clouds = []
    for hand in frame["hands"]:
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
                ax.plot([xs[a], xs[b]], [ys[a], ys[b]], [zs[a], zs[b]], color=color, linewidth=1.4)
        ax.scatter(xs, ys, zs, color=color, s=12)
    if clouds:
        stacked = np.vstack(clouds)
        center = stacked.mean(axis=0)
        span = max(float(np.ptp(stacked, axis=0).max()), 1e-3) * 0.6
        ax.set_xlim(center[0] - span, center[0] + span)
        ax.set_ylim(center[1] - span, center[1] + span)
        ax.set_zlim(center[2] - span, center[2] + span)
    ax.set_xlabel(coord)
    ax.set_title(coord, color="#d7dde3", fontsize=10)
    fig.tight_layout()
    return fig


def _timeline_figure(codes: list[int], index: int):
    fig, ax = plt.subplots(figsize=(8.2, 0.55), dpi=120)
    fig.patch.set_facecolor("#121418")
    ax.set_facecolor("#121418")
    if codes:
        colors = {0: "#2a3038", 1: LEFT_COLOR, 2: RIGHT_COLOR, 3: "#7dcea0"}
        ax.bar(range(len(codes)), [1] * len(codes), color=[colors[c] for c in codes], width=1.0)
        ax.axvline(index, color="#f2f4f6", linewidth=0.8)
    ax.set_xlim(0, max(len(codes) - 1, 1))
    ax.set_yticks([])
    ax.tick_params(colors="#9aa3ad", labelsize=7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0.2)
    return fig


def _pct(value) -> str:
    if value is None:
        return "—"
    return f"{100 * float(value):.0f}%"


def _metrics(report: dict | None) -> None:
    share = (report or {}).get("detection_share") or {}
    cols = st.columns(4)
    cols[0].metric("Either hand", _pct(share.get("either")))
    cols[1].metric("Both hands", _pct(share.get("both")))
    cols[2].metric("Hands FPS", f"{float((report or {}).get('fps_hands') or 0):.1f}")
    cols[3].metric("End-to-end FPS", f"{float((report or {}).get('fps_end_to_end') or 0):.1f}")
    factory = (report or {}).get("factory") or {}
    scorecard = (report or {}).get("scorecard") or {}
    if not factory and not scorecard:
        return
    extra = st.columns(4)
    balance = factory.get("hand_balance") or {}
    extra[0].metric("Left share of active frames", _pct(balance.get("left_active_share")))
    drift = (scorecard.get("world_drift") or {}).get("mean_m")
    extra[1].metric("World drift (m)", "—" if drift is None else f"{float(drift):.2f}")
    bone = scorecard.get("bone_length_cv")
    extra[2].metric("Bone-length variation", "—" if bone is None else f"{float(bone):.3f}")
    extra[3].metric("Cycles", int(factory.get("cycle_count") or 0))


def _run_upload(uploaded, seconds: float) -> str:
    if not DEFAULT_MODEL.is_file():
        raise FileNotFoundError("MediaPipe model is missing. Run bash scripts/download_models.sh first.")
    dest = save_upload(RAW_DIR, uploaded.name, uploaded.getvalue())
    fps = _clip_fps(dest)
    paths = run_video(dest, OUTPUT_DIR, max_frames=max_frames_for_seconds(fps, seconds))
    return Path(paths["jsonl"]).stem if paths.get("jsonl") else dest.stem


def main() -> None:
    st.set_page_config(page_title="PS4 Hand Workbench", layout="wide", initial_sidebar_state="expanded")
    if "demo" not in st.session_state:
        st.session_state.demo = False

    clips = discover_clips(OUTPUT_DIR)
    with st.sidebar:
        st.header("Upload a clip")
        uploaded = st.file_uploader("MP4", type=["mp4", "mov", "avi", "mkv"], label_visibility="collapsed")
        seconds = st.number_input(
            "Seconds to process",
            min_value=0,
            max_value=600,
            value=50,
            step=10,
            help="0 processes the whole file. One minute of video takes a few minutes on CPU.",
        )
        if st.button("Run", type="primary", disabled=uploaded is None, use_container_width=True):
            with st.spinner("Tracking hands. Keep this tab open."):
                try:
                    st.session_state.stem = _run_upload(uploaded, float(seconds))
                    st.session_state.run_error = None
                except Exception as exc:
                    st.session_state.run_error = str(exc)
            st.rerun()
        if st.session_state.get("run_error"):
            st.error(st.session_state.run_error)
        st.caption("The clip stays in data/raw and the results stay in outputs. Neither is committed.")
        st.divider()
        st.header("Processed clips")
        if not clips:
            st.info("No results yet. Upload a clip and press Run.")
            stem = None
        else:
            current = st.session_state.get("stem")
            index = clips.index(current) if current in clips else 0
            stem = st.radio("Processed clip", clips, index=index, label_visibility="collapsed")
            st.session_state.stem = stem
        st.divider()
        st.session_state.demo = st.toggle("Demo mode", value=st.session_state.demo)
        st.caption("Demo mode keeps the video and the headline numbers.")

    if st.session_state.demo:
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"] {display: none;}
            [data-testid="collapsedControl"] {display: none;}
            </style>
            """,
            unsafe_allow_html=True,
        )

    st.title("PS4 hand motion")
    if stem is None:
        st.info("Upload a clip in the sidebar and press Run.")
        st.stop()

    paths = clip_paths(OUTPUT_DIR, stem)
    report = load_report(paths["report"])
    rows = load_jsonl(paths["jsonl"])
    st.caption(stem)
    _metrics(report)

    if paths["video"] is not None:
        st.video(str(paths["video"]))
    else:
        st.warning("No side-by-side video for this clip.")

    if st.session_state.demo:
        if st.button("Leave demo mode"):
            st.session_state.demo = False
            st.rerun()
        st.stop()

    st.subheader("Frame inspector")
    if not rows:
        st.info("This clip has a video and no per-frame JSONL yet.")
    else:
        start = next((i for i, row in enumerate(rows) if row.get("hands")), 0)
        index = st.slider("Frame", 0, len(rows) - 1, start)
        frame = rows[index]
        st.pyplot(_timeline_figure(detection_codes(rows), index), clear_figure=True)
        st.caption("Timeline: teal left, amber right, green both, gray none. White line is the selected frame.")
        present = []
        for key in ("image", *COORD_KEYS):
            if key == "image" or any(hand_points(hand, key) is not None for hand in frame.get("hands", [])):
                present.append(key)
        coord = st.radio("Coordinates", present or ["world"], horizontal=True)
        left, right = st.columns([1.3, 1])
        with left:
            if paths["video"] is not None:
                image = _read_video_frame(str(paths["video"]), int(frame.get("frame", index)))
                if image is not None:
                    st.image(image, caption=f"t = {frame.get('t_sec', 0)} s")
            st.dataframe(hand_rows(frame, "image" if coord == "image" else coord), hide_index=True)
        with right:
            fig = _skeleton_figure(frame, coord)
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)

    downloads = st.columns(4)
    files = [
        ("Report", paths["report"]),
        ("JSONL", paths["jsonl"]),
        ("World CSV", paths["world_csv"]),
        ("World JSON", paths["world_sim"]),
    ]
    for column, (label, path) in zip(downloads, files):
        with column:
            if path is not None:
                st.download_button(label, path.read_bytes(), file_name=path.name, key=f"dl-{label}-{stem}")


if __name__ == "__main__":
    main()
