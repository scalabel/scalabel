"""Test cases for from_mot.py."""
from ..unittest.util import get_test_file
from .from_mot import parse_annotations


def test_parse_annotations() -> None:
    """Test the parse annotations function."""
    filepath = get_test_file("motchallenge_labels.txt")
    result = parse_annotations(filepath)

    assert list(result.keys()) == [0, 1, 3]

    for _, labels in result.items():
        for label in labels:
            assert label.attributes is not None
            assert label.category == "pedestrian"
            assert label.id == "1"
            assert label.attributes["visibility"] == 1.0
            assert label.box2d is not None
            assert label.box2d.x1 in [458, 460]
            assert label.box2d.x2 in [587, 589]
