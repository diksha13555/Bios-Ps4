from src.mirrored_label import correct_detection_label


def test_no_swap_until_spec():
    assert correct_detection_label("Left")[0] == "Left"
    assert correct_detection_label("Right", frame_mirrored=True)[0] == "Right"
