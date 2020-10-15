"""Test label type definition."""

from scalabel.label.typing import Frame


def test_frame_load() -> None:
    """Test loading frame."""
    json = (
        '{"name": 1, "videoName": "a", "size": [10, 20], '
        '"labels":[{"id": 1, "box2d": {"x1": 1, "y1": 2, "x2": 3, "y2": 4}}]}'
    )
    # pylint and mypy doesn't recognize from_json method
    frame = Frame.from_json(json)  # type: ignore # pylint: disable=no-member
    assert frame.name == "1"
    assert frame.video_name == "a"
    assert frame.labels[0].id == "1"
