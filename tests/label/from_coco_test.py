"""Test cases for from_coco.py."""
import unittest

from .coco_typing import GtType
from .from_coco import coco_to_scalabel


class TestCOCOToScalabel(unittest.TestCase):
    """Test cases for converting COCO to Scalabel format."""

    coco_obj = {
        "type": "detection",
        "categories": [
            {"id": 0, "name": "a"},
            {"id": 5, "name": "b"},
            {"id": 10, "name": "c"},
        ],
        "annotations": [
            {
                "id": 0,
                "image_id": 0,
                "category_id": 0,
                "iscrowd": 0,
                "ignore": 0,
            },
            {
                "id": 1,
                "image_id": 1,
                "category_id": 5,
                "iscrowd": 0,
                "ignore": 0,
            },
            {
                "id": 2,
                "image_id": 0,
                "category_id": 10,
                "iscrowd": 0,
                "ignore": 0,
            },
        ],
        "images": [
            {"id": 0, "file_name": "test.png", "height": 10, "width": 10},
            {"id": 1, "file_name": "test.png", "height": 10, "width": 10},
        ],
    }

    def test_from_coco(self) -> None:
        """Test conversion."""
        frames, cfg = coco_to_scalabel(GtType(**self.coco_obj))  # type: ignore
        self.assertEqual(len(frames), 2)
        assert frames[0].labels is not None
        assert frames[1].labels is not None
        self.assertEqual(len(frames[0].labels), 2)
        self.assertEqual(len(frames[1].labels), 1)

        cats = cfg.categories
        self.assertEqual(len(cats), 3)
        self.assertEqual(cats[0].name, "a")
        self.assertEqual(cats[1].name, "b")
        self.assertEqual(cats[2].name, "c")


if __name__ == "__main__":
    unittest.main()
