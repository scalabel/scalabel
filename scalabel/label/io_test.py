"""Test cases for detection engine config."""
from ..unittest.util import get_test_file
from .io import dump, load, parse


def test_parse() -> None:
    """Test parse label string."""
    json = (
        '{"name": 1, "videoName": "a", "size": [10, 20], '
        '"labels":[{"id": 1, "box2d": '
        '{"x1": 1, "y1": 2, "x2": 3, "y2": 4}}]}'
    )
    frames = parse(json)
    frame = frames[0]
    assert frame.name == "1"
    assert frame.video_name == "a"
    assert frame.labels[0].id == "1"
    b = frame.labels[0].box_2d
    assert b is not None
    assert b.y2 == 4


def test_load() -> None:
    """Test loading labels."""
    filepath = get_test_file("image_list_with_auto_labels.json")
    labels = load(filepath)
    assert len(labels) == 10
    assert labels[0].labels[0].id == "0"
    assert labels[0].labels[0].box_2d is not None
    assert labels[-1].labels[-1].box_2d is not None
    box = labels[-1].labels[-1].box_2d
    assert box.x1 == 218.7211456298828
    assert box.x2 == 383.5201416015625
    assert box.y1 == 362.24761962890625
    assert box.y2 == 482.4760437011719


def test_dump() -> None:
    """Test dump labels."""
    filepath = get_test_file("image_list_with_auto_labels.json")
    labels = load(filepath)
    labels_dict = dump(labels)
    assert "box3d" not in labels_dict[0]["labels"][0]
    assert "box2d" in labels_dict[0]["labels"][0]
    assert labels[0].labels[0].box_2d is not None
    assert (
        labels_dict[0]["labels"][0]["box2d"]["x1"]
        == labels[0].labels[0].box_2d.x1
    )
