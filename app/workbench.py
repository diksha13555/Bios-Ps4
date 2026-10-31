"""Local PS4 workbench: demo playback plus a frame inspector.

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

from src.hands import CONNECTIONS
from src.workbench_data import (
    COORD_KEYS,
    clip_paths,
    detection_codes,
    discover_clips,
    hand_points,
    hand_rows,
    load_jsonl,
    load_report,
)

OUTPUT_DIR = ROOT / "outputs"
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


def _metrics(report: dict | None) -> None:
    share = (report or {}).get("detection_share") or {}
    cols = st.columns(4)
    cols[0].metric("Either hand", f"{100 * float(share.get('either', 0)):.0f}%")
    cols[1].metric("Both hands", f"{100 * float(share.get('both', 0)):.0f}%")
    cols[2].metric("Hands FPS", f"{float((report or {}).get('fps_hands', 0)):.1f}")
    cols[3].metric("End-to-end FPS", f"{float((report or {}).get('fps_end_to_end', 0)):.1f}")


def main() -> None:
    st.set_page_config(page_title="PS4 Hand Workbench", layout="wide", initial_sidebar_state="expanded")
    if "demo" not in st.session_state:
        st.session_state.demo = False

    clips = discover_clips(OUTPUT_DIR)
    with st.sidebar:
        st.header("Clips")
        if not clips:
            st.info("No outputs yet. Run scripts/run_ps4.py first.")
            stem = None
        else:
            stem = st.radio("Processed clip", clips, label_visibility="collapsed")
        st.divider()
        st.session_state.demo = st.toggle("Demo mode", value=st.session_state.demo)
        st.caption("Demo mode keeps the video and the four headline numbers.")

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
        coord_options = [key for key in ("image", *COORD_KEYS) if any(hand_points(h, key if key != "image" else "world") is not None or key == "image" for h in frame.get("hands", []))]
        if not coord_options:
            coord_options = ["world"]
        coord = st.radio("Coordinates", coord_options, horizontal=True)
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
                st.download_button(label, path.read_bytes(), file_name=path.name)


if __name__ == "__main__":
    main()
