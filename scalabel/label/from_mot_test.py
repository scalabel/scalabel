"""Test cases for from_mot.py."""
from ..unittest.util import get_test_file
from .from_mot import parse_annotations


def test_parse_annotations() -> None:
    """Test the parse annotations function."""
    filepath = get_test_file("motchallenge_labels.txt")

    name_map = {"1": "keep", "2": "discard", "3": "ignore"}
    result = parse_annotations(
        filepath,
        name_mapping=name_map,
        discard_classes=["discard"],
        ignore_classes=["ignore"],
    )

    assert list(result.keys()) == [0, 1, 3]

    for frame_idx, labels in result.items():
        for label in labels:
            assert label.attributes is not None
            if label.attributes["ignore"]:
                assert label.category == "ignore"
                assert frame_idx == 1
            else:
                assert label.category == "keep"
                assert label.id == "1"
                assert label.attributes["visibility"] == 1.0
                assert label.box2d is not None
                assert label.box2d.x1 in [458, 460]
