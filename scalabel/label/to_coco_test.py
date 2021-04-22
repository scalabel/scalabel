"""Test cases for to_coco_test.py."""
import unittest
from functools import partial

import numpy as np

from ..unittest.util import get_test_file
from .coco_typing import AnnType
from .io import load
from .to_coco import (
    DEFAULT_COCO_CONFIG,
    load_coco_config,
    process_category,
    scalabel2coco_detection,
    set_seg_object_geometry,
)

SHAPE = (720, 1280)


class TestProcessCategory(unittest.TestCase):
    """Check the category after processing."""

    def test_ignore_as_class(self) -> None:
        """Check the case ignore_as_class as True."""
        categories, name_mapping, ignore_mapping = load_coco_config(
            "ins_seg",
            DEFAULT_COCO_CONFIG,
            ignore_as_class=True,
        )
        ignored, cat_id = process_category(
            "random",
            categories,
            name_mapping,
            ignore_mapping,
            ignore_as_class=True,
        )
        self.assertFalse(ignored)
        self.assertEqual(cat_id, 9)

    def test_not_ignore(self) -> None:
        """Check the case ignore_as_class as False."""
        categories, name_mapping, ignore_mapping = load_coco_config(
            "det",
            DEFAULT_COCO_CONFIG,
            ignore_as_class=False,
        )
        process_category_ = partial(
            process_category,
            categories=categories,
            name_mapping=name_mapping,
            ignore_mapping=ignore_mapping,
            ignore_as_class=False,
        )
        ignored, cat_id = process_category_("trailer")
        self.assertTrue(ignored)
        self.assertEqual(cat_id, 4)
        ignored, cat_id = process_category_("person")
        self.assertFalse(ignored)
        self.assertEqual(cat_id, 1)


class TestMaskToCoco(unittest.TestCase):
    """Test cases for conversion from Mask to COCO RLE."""

    def test_set_seg_object_geometry(self) -> None:
        """Check the mask to RLE conversion."""
        npy_file = get_test_file("mask.npy")
        rle_file = get_test_file("rle.npy")
        ann = AnnType(id=1, image_id=1, category_id=1, iscrowd=0, ignore=0)

        mask = np.load(npy_file)
        gt_rle = np.load(rle_file, allow_pickle=True).tolist()

        ann = set_seg_object_geometry(ann, mask, mask_mode="rle")
        self.assertListEqual(
            ann["bbox"], [199.0, 192.0, 403.0, 332.0]  # type: ignore
        )
        self.assertAlmostEqual(ann["area"], 108081.0)  # type: ignore
        self.assertDictEqual(ann["segmentation"], gt_rle)  # type: ignore


class TestScalabelToCOCODetection(unittest.TestCase):
    """Test cases for converting Scalabel detections to COCO format."""

    scalabel = load(get_test_file("scalabel_det.json"))
    categories, name_mapping, ignore_mapping = load_coco_config(
        "det", DEFAULT_COCO_CONFIG
    )
    coco = scalabel2coco_detection(
        SHAPE, scalabel, categories, name_mapping, ignore_mapping
    )

    def test_type(self) -> None:
        """Check coco format type."""
        self.assertTrue(isinstance(self.coco, dict))
        self.assertEqual(len(self.coco), 4)

    def test_num_images(self) -> None:
        """Check the number of images is unchanged."""
        self.assertEqual(len(self.scalabel), len(self.coco["images"]))

    def test_num_anns(self) -> None:
        """Check the number of annotations is unchanged."""
        len_scalabel = sum(
            [
                len(item.labels)
                for item in self.scalabel
                if item.labels is not None
            ]
        )
        len_coco = len(self.coco["annotations"])
        self.assertEqual(len_scalabel, len_coco)


if __name__ == "__main__":
    unittest.main()
