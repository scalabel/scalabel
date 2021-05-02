"""Test cases for io.py."""
import json

from ..unittest.util import get_test_file
from .io import dump, group_and_sort, load, parse
from .typing import Frame


def test_parse() -> None:
    """Test parse label string."""
    raw = json.loads(
        '{"name": 1, "videoName": "a", "size": [10, 20], '
        '"labels":[{"id": 1, "box2d": '
        '{"x1": 1, "y1": 2, "x2": 3, "y2": 4}, "attributes":'
        '{"crowd": false, "trafficLightColor": "G", "speed": 10}}]}'
    )
    frame = parse(raw)
    assert frame.name == "1"
    assert frame.video_name == "a"
    assert isinstance(frame.labels, list)
    assert len(frame.labels) == 1
    assert frame.labels[0].id == "1"
    assert frame.labels[0].attributes is not None
    assert frame.labels[0].attributes["crowd"] is False
    assert frame.labels[0].attributes["traffic_light_color"] == "G"
    assert frame.labels[0].attributes["speed"] == 10.0
    b = frame.labels[0].box_2d
    assert b is not None
    assert b.y2 == 4


def test_load() -> None:
    """Test loading labels."""
    filepath = get_test_file("image_list_with_auto_labels.json")

    def assert_correctness(inputs: str, nprocs: int) -> None:
        frames = load(inputs, nprocs)
        assert len(frames) == 10
        assert (
            frames[0].url == "https://s3-us-west-2.amazonaws.com/bdd-label/"
            "bdd100k/frames-20000/val/c1ba5ee6-b2cb1e51.jpg"
        )
        assert frames[0].frame_index == 0
        assert frames[-1].frame_index == 9
        assert frames[0].labels is not None
        assert frames[-1].labels is not None
        assert frames[0].labels[0].id == "0"
        assert frames[0].labels[0].box_2d is not None
        assert frames[-1].labels[-1].box_2d is not None
        box = frames[-1].labels[-1].box_2d
        assert box.x1 == 218.7211456298828
        assert box.x2 == 383.5201416015625
        assert box.y1 == 362.24761962890625
        assert box.y2 == 482.4760437011719
        assert frames[0].labels[0].poly_2d is not None
        polys = frames[0].labels[0].poly_2d
        assert isinstance(polys, list)
        poly = polys[0]
        assert len(poly.vertices) == len(poly.types)
        assert len(poly.vertices[0]) == 2
        for char in poly.types:
            assert char in ["C", "L"]

    assert_correctness(filepath, nprocs=0)
    assert_correctness(filepath, nprocs=2)


def test_group_and_sort() -> None:
    """Check the group and sort results."""
    frames = [
        Frame(name="bbb-1", video_name="bbb", frame_index=1, labels=[]),
        Frame(name="aaa-2", video_name="aaa", frame_index=2, labels=[]),
        Frame(name="aaa-2", video_name="aaa", frame_index=1, labels=[]),
    ]
    frames_list = group_and_sort(frames)

    assert len(frames_list) == 2
    assert len(frames_list[0]) == 2
    assert len(frames_list[1]) == 1

    assert str(frames_list[0][0].video_name) == "aaa"
    assert frames_list[0][1].name == "aaa-2"
    assert frames_list[0][1].frame_index == 2


def test_dump() -> None:
    """Test dump labels."""
    filepath = get_test_file("image_list_with_auto_labels.json")
    labels = load(filepath)
    labels_dict = [dump(label) for label in labels]
    assert labels_dict[0]["frameIndex"] == labels[0].frame_index
    assert labels_dict[-1]["frameIndex"] == labels[-1].frame_index
    assert "box3d" not in labels_dict[0]["labels"][0]
    assert "box2d" in labels_dict[0]["labels"][0]
    assert labels[0].labels is not None
    assert labels[0].labels[0].box_2d is not None
    assert (
        labels_dict[0]["labels"][0]["box2d"]["x1"]
        == labels[0].labels[0].box_2d.x1
    )
