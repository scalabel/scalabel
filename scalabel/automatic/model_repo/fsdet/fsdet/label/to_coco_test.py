"""Test cases for to_coco_test.py."""
import unittest

import numpy as np

from ..unittest.util import get_test_file
from .coco_typing import AnnType
from .io import load, load_label_config
from .to_coco import (
    scalabel2coco_detection,
    scalabel2coco_ins_seg,
    scalabel2coco_pose,
    set_seg_object_geometry,
)


class TestMaskToCoco(unittest.TestCase):
    """Test cases for conversion from Mask to COCO RLE."""

    def test_set_seg_object_geometry(self) -> None:
        """Check the mask to RLE conversion."""
        npy_file = get_test_file("mask.npy")
        rle_file = get_test_file("rle.npy")
        ann = AnnType(id=1, image_id=1, category_id=1, iscrowd=0, ignore=0)

        mask = np.load(npy_file)
        gt_rle = np.load(rle_file, allow_pickle=True).tolist()

        ann = set_seg_object_geometry(ann, mask)
        self.assertListEqual(
            ann["bbox"], [199.0, 192.0, 403.0, 332.0]  # type: ignore
        )
        self.assertAlmostEqual(ann["area"], 108081.0)  # type: ignore
        self.assertDictEqual(ann["segmentation"], gt_rle)  # type: ignore


class TestScalabelToCOCODetection(unittest.TestCase):
    """Test cases for converting Scalabel detections to COCO format."""

    scalabel = load(get_test_file("scalabel_det.json")).frames
    config = load_label_config(get_test_file("configs.toml"))
    coco = scalabel2coco_detection(scalabel, config)

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


class TestScalabelToCOCOInsSeg(unittest.TestCase):
    """Test cases for converting Scalabel segmentations to COCO format."""

    scalabel = load(get_test_file("scalabel_ins_seg.json")).frames
    config = load_label_config(get_test_file("configs.toml"))
    coco = scalabel2coco_ins_seg(scalabel, config)

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


class TestScalabelToCOCOPose(unittest.TestCase):
    """Test cases for converting Scalabel pose to COCO format."""

    scalabel = load(get_test_file("scalabel_pose.json"))
    config = scalabel.config
    if not config:
        config = load_label_config(get_test_file("pose_configs.toml"))
    coco = scalabel2coco_pose(scalabel.frames, config)

    def test_type(self) -> None:
        """Check coco format type."""
        self.assertTrue(isinstance(self.coco, dict))
        self.assertEqual(len(self.coco), 3)

    def test_num_images(self) -> None:
        """Check the number of images is unchanged."""
        self.assertEqual(len(self.scalabel.frames), len(self.coco["images"]))

    def test_num_anns(self) -> None:
        """Check the number of annotations is unchanged."""
        len_scalabel = sum(
            [
                len(item.labels)
                for item in self.scalabel.frames
                if item.labels is not None
            ]
        )
        len_coco = len(self.coco["annotations"])
        self.assertEqual(len_scalabel, len_coco)


if __name__ == "__main__":
    unittest.main()
