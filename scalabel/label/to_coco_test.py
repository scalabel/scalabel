"""Test cases for bdd100k2coco.py."""
import os
import unittest

from .typing import Frame
from .to_coco import (
    group_and_sort,
    load_default_cfgs,
    process_category,
    read,
    scalabel2coco_detection,
)


SHAPE = (720, 1280)


class TestGroupAndSort(unittest.TestCase):
    """Check the group and sort results' order."""

    frames = [
        Frame(name="bbb-2", video_name="bbb", frame_index=1, labels=[]),
        Frame(name="aaa-1", video_name="aaa", frame_index=2, labels=[]),
        Frame(name="aaa-2", video_name="aaa", frame_index=1, labels=[]),
    ]
    frames_list = group_and_sort(frames)

    def test_num(self) -> None:
        """Check the number of frames in the results."""
        self.assertEqual(len(self.frames_list), 2)
        self.assertEqual(len(self.frames_list[0]), 2)
        self.assertEqual(len(self.frames_list[1]), 2)

    def test_names(self) -> None:
        """Check `name` and `video_name` in the results."""
        self.assertEqual(self.frames_list[0][0].video_name, "aaa")
        self.assertEqual(self.frames_list[0][1].name, "aaa-2")


def test_process_category() -> None:
    """Check the category after processing."""
    pass


def test_poly2ds2coco() -> None:
    """Check the converted RLE from poly2d."""
    pass


class TestBDD100K2COCO(unittest.TestCase):
    """Test cases for converting BDD100K labels to COCO format."""

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    val_bdd = read("{}/testcases/unitest_val_bdd.json".format(cur_dir))
    categories, name_mapping, ignore_mapping = load_default_cfgs("det")
    val_coco = scalabel2coco_detection(
        SHAPE, val_bdd, categories, name_mapping, ignore_mapping
    )

    def test_type(self) -> None:
        """Check coco format type."""
        self.assertTrue(isinstance(self.val_coco, dict))
        self.assertEqual(len(self.val_coco), 4)

    def test_num_images(self) -> None:
        """Check the number of images is unchanged."""
        self.assertEqual(len(self.val_bdd), len(self.val_coco["images"]))

    def test_num_anns(self) -> None:
        """Check the number of annotations is unchanged."""
        len_bdd = sum([len(item.labels) for item in self.val_bdd])
        len_coco = len(self.val_coco["annotations"])
        self.assertEqual(len_coco, len_bdd)


if __name__ == "__main__":
    unittest.main()
