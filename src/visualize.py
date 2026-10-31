"""Side-by-side: source with 2D skeleton, 3D panel from world (or camera) landmarks."""

from __future__ import annotations

import cv2
import numpy as np

from src.hands import CONNECTIONS

COLORS = {"Left": (255, 180, 60), "Right": (80, 140, 255)}


class SkeletonPanel:
    def __init__(self, width: int, height: int, *, zoom: float = 1.25):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self._plt = plt
        self.zoom = float(np.clip(zoom, 1.0, 2.0))
        self._center: np.ndarray | None = None
        self._radius: float | None = None
        dpi = 100
        self.fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        self.fig.patch.set_facecolor("#141414")
        self.ax = self.fig.add_axes([0.0, 0.01, 1.0, 0.94], projection="3d")
        self.ax.set_box_aspect((1, 1, 1), zoom=self.zoom)
        self.w, self.h = width, height

    def close(self):
        self._plt.close(self.fig)

    def render(self, hands: list[dict]) -> np.ndarray:
        ax = self.ax
        ax.cla()
        ax.set_facecolor("#141414")
        ax.set_box_aspect((1, 1, 1), zoom=self.zoom)
        clouds = []
        title = "MediaPipe world landmarks (m) — auto zoom"
        for hand in hands:
            if hand.get("camera") is not None:
                pts = np.asarray(hand["camera"], dtype=np.float64)
                plot = np.column_stack([pts[:, 0], -pts[:, 1], pts[:, 2]])
                title = "Approximate camera landmarks — auto zoom"
            else:
                off = -0.1 if hand["label"] == "Left" else 0.1
                w = np.asarray(hand["world"], dtype=np.float64)
                plot = np.column_stack([w[:, 0] + off, -w[:, 1], w[:, 2]])
            clouds.append(plot)
            color = tuple(c / 255 for c in COLORS.get(hand["label"], (200, 200, 200))[::-1])
            for a, b in CONNECTIONS:
                seg = plot[[a, b]]
                ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=color, linewidth=3)
            ax.scatter(plot[:, 0], plot[:, 1], plot[:, 2], s=32, color=color, depthshade=False)
            ax.text(plot[0, 0], plot[0, 1], plot[0, 2], hand["label"], color=color, fontsize=11, weight="bold")
        ax.set_title(title, color="#eee", fontsize=11, pad=2)
        self._limits(ax, clouds)
        ax.view_init(elev=18, azim=-70)
        self.fig.canvas.draw()
        buf = np.asarray(self.fig.canvas.buffer_rgba()).reshape(self.fig.canvas.get_width_height()[1], -1, 4)
        bgr = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
        if bgr.shape[1] != self.w or bgr.shape[0] != self.h:
            bgr = cv2.resize(bgr, (self.w, self.h))
        return bgr

    def _limits(self, ax, clouds):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        if not clouds:
            if self._center is None:
                ax.set_xlim(-0.15, 0.15)
                ax.set_ylim(-0.15, 0.15)
                ax.set_zlim(-0.15, 0.15)
            return
        stack = np.vstack(clouds)
        low = stack.min(axis=0)
        high = stack.max(axis=0)
        target_center = (low + high) / 2.0
        # Display-only fitting: never scale or mutate exported metric landmarks.
        target_radius = max(float((high - low).max()) * 0.56, 0.035)
        if self._center is None or self._radius is None:
            self._center = target_center
            self._radius = target_radius
        else:
            self._center = 0.65 * self._center + 0.35 * target_center
            self._radius = 0.65 * self._radius + 0.35 * target_radius
        c, r = self._center, self._radius
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)


def draw_2d(frame: np.ndarray, hands: list[dict]) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    for hand in hands:
        color = COLORS.get(hand["label"], (200, 200, 200))
        pts = []
        for p in hand["image"]:
            pts.append((int(np.clip(p[0] * w, 0, w - 1)), int(np.clip(p[1] * h, 0, h - 1))))
        dashed = hand.get("interpolated")
        for a, b in CONNECTIONS:
            if dashed:
                cv2.line(out, pts[a], pts[b], color, 1, cv2.LINE_AA)
            else:
                cv2.line(out, pts[a], pts[b], color, 2, cv2.LINE_AA)
        for pt in pts:
            cv2.circle(out, pt, 3, color, -1, cv2.LINE_AA)
        tag = hand["label"] + (" ~" if dashed else "")
        cv2.putText(out, tag, pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return out


def compose(source: np.ndarray, hands: list[dict], panel: SkeletonPanel, hud: str) -> np.ndarray:
    left = draw_2d(source, hands)
    right = panel.render(hands)
    if right.shape[0] != left.shape[0]:
        right = cv2.resize(right, (left.shape[1], left.shape[0]))
    both = np.hstack([left, right])
    cv2.putText(both, hud, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 3, cv2.LINE_AA)
    cv2.putText(both, hud, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    eh, ew = both.shape[:2]
    return both[: eh - eh % 2, : ew - ew % 2]


def resize_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    m = max(h, w)
    if max_side <= 0 or m <= max_side:
        return img
    s = max_side / m
    return cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
