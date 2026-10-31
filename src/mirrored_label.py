"""Mirrored-input handedness correction (spec pending).

MediaPipe Tasks Hand Landmarker labels Left/Right as the subject's hands. Whether
the pipeline must flip labels depends on how egocentric frames are oriented in
the TouchTrace / BIOS V2 Problem Statement 2 specification. That PDF is not in
this repository yet, so no automatic swap is applied here.

When the hackathon rule is confirmed, implement it in ``correct_detection_label``
without changing IoU identity tracking in ``assign_tracks``.
"""

from __future__ import annotations

from typing import Literal

SpecStatus = Literal["pending_spec"]


def correct_detection_label(
    label: str,
    *,
    frame_mirrored: bool | None = None,
) -> tuple[str, SpecStatus]:
    """Return the handedness label to use for a single detection.

    Currently returns the model label unchanged. ``frame_mirrored`` is reserved
    for a future spec-driven rule (e.g. selfie mirror vs rear camera).
    """
    _ = frame_mirrored
    if label not in ("Left", "Right"):
        return "Right", "pending_spec"
    return label, "pending_spec"
