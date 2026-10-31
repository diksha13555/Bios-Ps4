from __future__ import annotations

import copy

import numpy as np

from src.visualize import SkeletonPanel


def test_limits_fit_actual_cloud_bounds_tightly():
    panel = SkeletonPanel(320, 240)
    try:
        cloud = np.array([[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]])
        panel._limits(panel.ax, [cloud])
        assert np.allclose(panel.ax.get_xlim(), (-0.112, 0.112))
        assert np.allclose(panel.ax.get_ylim(), (-0.112, 0.112))
        assert np.allclose(panel.ax.get_zlim(), (-0.112, 0.112))
    finally:
        panel.close()


def test_zoom_is_clamped_and_render_preserves_landmarks():
    points = np.linspace([-0.08, -0.04, -0.02], [0.08, 0.04, 0.02], 21)
    hand = {"label": "Left", "world": points.tolist(), "image": points.tolist()}
    before = copy.deepcopy(hand)
    panel = SkeletonPanel(320, 240, zoom=5.0)
    try:
        assert panel.zoom == 2.0
        image = panel.render([hand])
        assert image.shape == (240, 320, 3)
        assert hand == before
    finally:
        panel.close()


def test_view_limits_are_temporally_smoothed():
    panel = SkeletonPanel(320, 240)
    try:
        first = np.array([[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]])
        second = np.array([[-0.2, 0.0, 0.0], [0.2, 0.0, 0.0]])
        panel._limits(panel.ax, [first])
        panel._limits(panel.ax, [second])
        expected_radius = 0.65 * 0.112 + 0.35 * 0.224
        assert np.allclose(panel.ax.get_xlim(), (-expected_radius, expected_radius))
    finally:
        panel.close()
