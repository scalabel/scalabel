"""Test cases for from_mot.py."""
from ..unittest.util import get_test_file
from .from_mot import parse_annotations
from .typing import Category, Config


def test_parse_annotations() -> None:
    """Test the parse annotations function."""
    filepath = get_test_file("motchallenge_labels.txt")
    metadata_cfg = Config(
        categories=[
            Category(name="pedestrian"),
            Category(name="static person"),
        ]
    )
    result = parse_annotations(filepath, metadata_cfg)

    assert list(result.keys()) == [0, 1, 3]

    for frame_idx, labels in result.items():
        for label in labels:
            assert label.attributes is not None
            if label.attributes["crowd"]:
                assert label.category == "static person"
                assert frame_idx == 1
            else:
                assert label.category == "pedestrian"
                assert label.id == "1"
                assert label.attributes["visibility"] == 1.0
                assert label.box2d is not None
                assert label.box2d.x1 in [458, 460]
                assert label.box2d.x2 in [589, 587]
