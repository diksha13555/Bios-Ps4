import json

from src.eval_gt import evaluate_ground_truth, load_jsonl, match_hands_spatial


def _hand(label, x, y):
    pt = [x, y, 0.0]
    return {"label": label, "score": 0.9, "image": [pt] * 21, "world": [pt] * 21}


def test_spatial_match_ignores_label():
    preds = [_hand("Right", 0.3, 0.4)]
    gt = [_hand("Left", 0.31, 0.41)]
    matches, up, ug = match_hands_spatial(preds, gt, match_dist=0.05)
    assert matches == [(0, 0)]
    assert up == []
    assert ug == []


def test_empty_labelled_frame_counts_false_positives():
    preds = {0: {"frame": 0, "hands": [_hand("Left", 0.2, 0.3)]}}
    labels = {0: {"frame": 0, "hands": []}}
    r = evaluate_ground_truth(preds, labels)
    assert r["status"] == "ok"
    assert r["detection"]["false_positives"] == 1
    assert r["detection"]["false_negatives"] == 0


def test_handedness_separate_from_detection():
    preds = {0: {"frame": 0, "hands": [_hand("Right", 0.3, 0.4)]}}
    labels = {0: {"frame": 0, "hands": [_hand("Left", 0.31, 0.41)]}}
    r = evaluate_ground_truth(preds, labels, match_dist=0.05)
    assert r["detection"]["true_positives"] == 1
    assert r["handedness"]["accuracy"] == 0.0


def test_landmark_error_and_pck():
    preds = {0: {"frame": 0, "hands": [_hand("Left", 0.30, 0.40)]}}
    labels = {0: {"frame": 0, "hands": [_hand("Left", 0.32, 0.42)]}}
    r = evaluate_ground_truth(preds, labels, match_dist=0.05, pck_threshold=0.05)
    assert r["landmarks"]["mean_error_norm"] is not None
    assert r["landmarks"]["pck"] == 1.0


def test_unlabelled_frames_excluded():
    preds = {0: {"frame": 0, "hands": [_hand("Left", 0.3, 0.4)]}, 99: {"frame": 99, "hands": []}}
    labels = {0: {"frame": 0, "hands": [_hand("Left", 0.3, 0.4)]}}
    r = evaluate_ground_truth(preds, labels, match_dist=0.02)
    assert r["labelled_frames"] == 1
    assert r["detection"]["true_positives"] == 1


def test_no_labels_status():
    r = evaluate_ground_truth({}, {})
    assert r["status"] == "no_labelled_frames"


def test_load_jsonl(tmp_path):
    path = tmp_path / "a.jsonl"
    path.write_text(json.dumps({"frame": 3, "hands": []}) + "\n", encoding="utf-8")
    data = load_jsonl(path)
    assert 3 in data
