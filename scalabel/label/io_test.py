"""Test cases for detection engine config."""
import unittest

from ..unittest.util import get_test_file
from .io import dump, load, parse


class TestIO(unittest.TestCase):
    """Test the IO."""

    def test_parse(self) -> None:
        """Test parse label string."""
        json = (
            '{"name": 1, "videoName": "a", "size": [10, 20], '
            '"labels":[{"id": 1, "box2d": '
            '{"x1": 1, "y1": 2, "x2": 3, "y2": 4}}]}'
        )
        frames = parse(json)
        frame = frames[0]
        self.assertTrue(frame.name == "1")
        self.assertTrue(frame.video_name == "a")
        self.assertTrue(frame.labels[0].id == "1")
        b = frame.labels[0].box_2d
        self.assertTrue(b is not None)
        if b is not None:
            self.assertTrue(b.y2 == 4)

    def test_load(self) -> None:
        """Test loading labels."""
        filepath = get_test_file("image_list_with_auto_labels.json")
        labels = load(filepath)
        print(labels[0].labels)
        self.assertTrue(len(labels) == 10)
        self.assertTrue(
            (
                labels[0].url
                == "https://s3-us-west-2.amazonaws.com/bdd-label/"
                "bdd100k/frames-20000/val/c1ba5ee6-b2cb1e51.jpg"
            )
        )
        self.assertTrue(labels[0].labels[0].id == "0")
        self.assertTrue(labels[0].labels[0].box_2d is not None)
        self.assertTrue(labels[-1].labels[-1].box_2d is not None)
        box = labels[-1].labels[-1].box_2d
        self.assertTrue(box is not None)
        if box is not None:
            self.assertTrue(box.x1 == 218.7211456298828)
            self.assertTrue(box.x2 == 383.5201416015625)
            self.assertTrue(box.y1 == 362.24761962890625)
            self.assertTrue(box.y2 == 482.4760437011719)

    def test_dump(self) -> None:
        """Test dump labels."""
        filepath = get_test_file("image_list_with_auto_labels.json")
        labels = load(filepath)
        labels_dict = dump(labels)
        self.assertTrue("box3d" not in labels_dict[0]["labels"][0])
        self.assertTrue("box2d" in labels_dict[0]["labels"][0])
        self.assertTrue(labels[0].labels[0].box_2d is not None)
        if labels[0].labels[0].box_2d is not None:
            self.assertTrue(
                (
                    labels_dict[0]["labels"][0]["box2d"]["x1"]
                    == labels[0].labels[0].box_2d.x1
                )
            )


if __name__ == "__main__":
    unittest.main()
