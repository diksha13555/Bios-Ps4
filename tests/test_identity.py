from src.hands import assign_tracks, center_distance, landmarks_bbox
from src.metrics import detection_share


def _det(label, box, score=0.9):
    c = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, 0.0]
    img = [list(c) for _ in range(21)]
    return {"label": label, "score": score, "image": img, "world": img, "bbox": box}


def test_side_swap_keeps_identity():
    box = (0.2, 0.2, 0.5, 0.6)
    tracks, out = assign_tracks([], [_det("Left", box)])
    tracks, out = assign_tracks(tracks, [_det("Right", (0.22, 0.22, 0.52, 0.62))])
    assert out[0]["identity"] == "Left"


def test_two_hand_label_flip_preserves_identities():
    left_box = (0.05, 0.2, 0.35, 0.6)
    right_box = (0.55, 0.2, 0.85, 0.6)
    tracks, out = assign_tracks(
        [],
        [_det("Left", left_box), _det("Right", right_box)],
    )
    assert {h["identity"] for h in out} == {"Left", "Right"}
    tracks, out = assign_tracks(
        tracks,
        [_det("Right", left_box), _det("Left", right_box)],
    )
    assert out[0]["identity"] == "Left"
    assert out[1]["identity"] == "Right"


def test_detection_share():
    frames = [
        {"hands": [{"label": "Left", "interpolated": False}]},
        {"hands": [{"label": "Left", "interpolated": True}]},
    ]
    s = detection_share(frames, detected_only=True)
    assert abs(s["left"] - 0.5) < 1e-9


def test_reentry_within_ten_frames():
    box = (0.2, 0.2, 0.5, 0.6)
    tracks, _out = assign_tracks([], [_det("Left", box)])
    for _ in range(10):
        tracks, out = assign_tracks(tracks, [])
    assert out[0]["identity"] == "Left"
    assert out[0]["interpolated"] is True
    _tracks, out = assign_tracks(tracks, [_det("Right", (0.22, 0.22, 0.52, 0.62))])
    assert out[0]["identity"] == "Left"
    assert out[0]["interpolated"] is False


def test_eleven_empty_frames_drop_the_track():
    box = (0.2, 0.2, 0.5, 0.6)
    tracks, _out = assign_tracks([], [_det("Left", box)])
    out = []
    for _ in range(11):
        tracks, out = assign_tracks(tracks, [])
    assert out == []


def test_center_distance_fallback_when_iou_low():
    box_a = (0.2, 0.2, 0.35, 0.45)
    tracks, _ = assign_tracks([], [_det("Left", box_a)])
    shifted = (0.38, 0.2, 0.53, 0.45)
    assert center_distance(box_a, shifted) > 0.05
    tracks, out = assign_tracks(tracks, [_det("Right", shifted)], iou_min=0.25, center_dist_max=0.2)
    assert len(out) == 1
    assert out[0]["identity"] == "Left"
    assert out[0]["interpolated"] is False


def test_no_duplicate_identity_when_both_slots_taken():
    left_box = (0.05, 0.2, 0.35, 0.6)
    right_box = (0.55, 0.2, 0.85, 0.6)
    tracks, _ = assign_tracks([], [_det("Left", left_box), _det("Right", right_box)])
    extra = (0.1, 0.65, 0.3, 0.9)
    tracks, out = assign_tracks(tracks, [_det("Left", extra)], max_gap=0)
    identities = [h["identity"] for h in out]
    assert identities.count("Left") <= 1
    assert identities.count("Right") <= 1


def test_expired_track_allows_new_identity():
    box = (0.2, 0.2, 0.5, 0.6)
    tracks, _ = assign_tracks([], [_det("Left", box)])
    for _ in range(11):
        tracks, _ = assign_tracks(tracks, [])
    tracks, out = assign_tracks(tracks, [_det("Right", (0.22, 0.22, 0.52, 0.62))])
    assert len(out) == 1
    assert out[0]["identity"] == "Right"
