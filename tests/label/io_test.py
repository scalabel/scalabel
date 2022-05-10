"""Test cases for io.py."""
import json

from ..unittest.util import get_test_file
from .io import dump, group_and_sort, load, parse
from .typing import Frame


def test_parse() -> None:
    """Test parse label string."""
    raw = json.loads(
        '{"name": 1, "videoName": "a", "size": {"width": 10, "height": 20}, '
        '"labels":[{"id": 1, "box2d": '
        '{"x1": 1, "y1": 2, "x2": 3, "y2": 4}, "ignored": true, '
        '"attributes": {"trafficLightColor": "G", "speed": 10}}]}'
    )
    frame = parse(raw)
    assert frame.name == "1"
    assert frame.videoName == "a"
    assert isinstance(frame.labels, list)
    labels = frame.labels
    assert isinstance(labels, list)
    assert len(labels) == 1
    label = labels[0]  # pylint: disable=unsubscriptable-object
    assert label.id == "1"
    assert label.attributes is not None
    assert label.attributes["trafficLightColor"] == "G"
    assert label.attributes["speed"] == 10.0
    b = label.box2d
    assert b is not None
    assert b.y2 == 4


def test_load() -> None:
    """Test loading labels."""
    filepath = get_test_file("image_list_with_auto_labels.json")

    def assert_correctness(inputs: str, nprocs: int) -> None:
        frames = load(inputs, nprocs).frames
        assert len(frames) == 10
        assert (
            frames[0].url == "https://s3-us-west-2.amazonaws.com/bdd-label/"
            "bdd100k/frames-20000/val/c1ba5ee6-b2cb1e51.jpg"
        )
        assert frames[0].frameIndex == 0
        assert frames[-1].frameIndex == 9
        assert frames[0].labels is not None
        assert frames[-1].labels is not None
        assert frames[0].labels[0].id == "0"
        assert frames[0].labels[0].box2d is not None
        assert frames[-1].labels[-1].box2d is not None
        box = frames[-1].labels[-1].box2d
        assert box.x1 == 218.7211456298828
        assert box.x2 == 383.5201416015625
        assert box.y1 == 362.24761962890625
        assert box.y2 == 482.4760437011719
        assert frames[0].labels[0].poly2d is not None
        polys = frames[0].labels[0].poly2d
        assert isinstance(polys, list)
        poly = polys[0]
        assert len(poly.vertices) == len(poly.types)
        assert len(poly.vertices[0]) == 2
        for char in poly.types:
            assert char in ["C", "L"]

    assert_correctness(filepath, nprocs=0)
    assert_correctness(filepath, nprocs=2)


def test_load_graph() -> None:
    """Test loading labels."""
    filepath = get_test_file("image_list_with_auto_labels_graph.json")

    def assert_correctness(inputs: str, nprocs: int) -> None:
        frames = load(inputs, nprocs).frames
        assert len(frames) == 10
        assert (
            frames[0].url == "https://s3-us-west-2.amazonaws.com/bdd-label/"
            "bdd100k/frames-20000/val/c1ba5ee6-b2cb1e51.jpg"
        )
        assert frames[2].frameIndex == 2
        assert frames[4].frameIndex == 4
        assert frames[2].labels is not None
        assert frames[4].labels is not None
        assert frames[2].labels[0].id == "0"
        assert frames[2].labels[0].box2d is not None
        assert frames[4].labels[1].box2d is not None
        box = frames[4].labels[1].box2d
        assert box.x1 == 1181.4259033203125
        assert box.x2 == 1241.681396484375
        assert box.y1 == 101.82328796386719
        assert box.y2 == 155.20513916015625
        assert frames[0].labels is not None
        assert frames[0].labels[0].poly2d is not None
        polys = frames[0].labels[0].poly2d
        assert isinstance(polys, list)
        poly = polys[0]
        assert len(poly.vertices) == len(poly.types)
        assert len(poly.vertices[0]) == 2
        for char in poly.types:
            assert char in ["C", "L"]
        assert frames[0].labels[0].graph is not None
        assert frames[0].labels[0].graph.nodes is not None
        assert frames[0].labels[0].graph.edges is not None
        nodes = frames[0].labels[0].graph.nodes
        edges = frames[0].labels[0].graph.edges
        assert isinstance(nodes, list)
        assert isinstance(edges, list)
        assert len(nodes) == 9
        assert len(edges) == 9
        assert nodes[1].location[0] == 205.20687963549207
        assert nodes[1].location[1] == 278.4950509338821
        assert nodes[1].category == "polygon"
        assert edges[2].source == "5vowGRmRHjolm1-G"
        assert edges[2].target == "MqOQsu8Tqn6sLoLM"

        _ = load(inputs, nprocs, False)

    assert_correctness(filepath, nprocs=0)
    assert_correctness(filepath, nprocs=2)


def test_group_and_sort() -> None:
    """Check the group and sort results."""
    # frames = [
    #     Frame(name="bbb-1", videoName="bbb", frameIndex=1, labels=[]),
    #     Frame(name="aaa-2", videoName="aaa", frameIndex=2, labels=[]),
    #     Frame(name="aaa-2", videoName="aaa", frameIndex=1, labels=[]),
    # ]
    frames = [
        Frame(name="bbb-1", videoName="bbb", frameIndex=1, labels=[]),
        Frame(name="aaa-2", videoName="aaa", frameIndex=2, labels=[]),
        Frame(name="bbb-2", videoName="bbb", frameIndex=2, labels=[]),
        Frame(name="aaa-2", videoName="aaa", frameIndex=1, labels=[]),
        Frame(name="bbb-3", videoName="bbb", frameIndex=3, labels=[]),
    ]
    frames_list = group_and_sort(frames)

    assert len(frames_list) == 2
    assert len(frames_list[0]) == 2
    assert len(frames_list[1]) == 3

    assert str(frames_list[0][0].videoName) == "aaa"
    assert frames_list[0][1].name == "aaa-2"
    assert frames_list[0][1].frameIndex == 2

    assert str(frames_list[1][0].videoName) == "bbb"
    assert frames_list[1][1].frameIndex == 2
    assert frames_list[1][1].name == "bbb-2"


def test_dump() -> None:
    """Test dump labels."""
    filepath = get_test_file("image_list_with_auto_labels.json")
    labels = load(filepath).frames
    labels_dict = [dump(label.dict()) for label in labels]
    assert labels_dict[0]["frameIndex"] == labels[0].frameIndex
    assert labels_dict[-1]["frameIndex"] == labels[-1].frameIndex
    assert "box3d" not in labels_dict[0]["labels"][0]
    assert "box2d" in labels_dict[0]["labels"][0]
    assert labels[0].labels is not None
    assert labels[0].labels[0].box2d is not None
    assert (
        labels_dict[0]["labels"][0]["box2d"]["x1"]
        == labels[0].labels[0].box2d.x1
    )
