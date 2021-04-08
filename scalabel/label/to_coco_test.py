"""Test cases for bdd100k2coco.py."""
import json
import unittest
from functools import partial

import numpy as np

from ..unittest.util import get_test_file
from .coco_typing import AnnType
from .to_coco import (
    box2d_to_bbox,
    group_and_sort,
    load_default_cfgs,
    mask_to_polygon,
    poly2ds_to_mask,
    process_category,
    read,
    scalabel2coco_detection,
    set_seg_object_geometry,
)
from .typing import Box2D, Frame, Poly2D

SHAPE = (720, 1280)


class TestGroupAndSort(unittest.TestCase):
    """Check the group and sort results' order."""

    frames = [
        Frame(name="bbb-1", video_name="bbb", frame_index=1, labels=[]),
        Frame(name="aaa-2", video_name="aaa", frame_index=2, labels=[]),
        Frame(name="aaa-2", video_name="aaa", frame_index=1, labels=[]),
    ]
    frames_list = group_and_sort(frames)

    def test_num(self) -> None:
        """Check the number of frames in the results."""
        self.assertEqual(len(self.frames_list), 2)
        self.assertEqual(len(self.frames_list[0]), 2)
        self.assertEqual(len(self.frames_list[1]), 1)

    def test_names(self) -> None:
        """Check `name` and `video_name` in the results."""
        self.assertSequenceEqual(str(self.frames_list[0][0].video_name), "aaa")
        self.assertSequenceEqual(self.frames_list[0][1].name, "aaa-2")
        self.assertEqual(self.frames_list[0][1].frame_index, 2)


class TestProcessCategory(unittest.TestCase):
    """Check the category after processing."""

    def test_ignore_as_class(self) -> None:
        """Check the case ignore_as_class as True."""
        categories, name_mapping, ignore_mapping = load_default_cfgs(
            "ins_seg",
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
        categories, name_mapping, ignore_mapping = load_default_cfgs(
            "det",
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


class TestBox2DToCoco(unittest.TestCase):
    """Test cases for conversion from Box2D to COCO bbox."""

    def test_box2d_to_coco(self) -> None:
        """Check the Box2D to bbox conversion."""
        box_2d = Box2D(x1=10, x2=19, y1=10, y2=19)
        bbox = box2d_to_bbox(box_2d)
        self.assertListEqual(bbox, [10.0, 10.0, 10.0, 10.0])


class TestPoly2DToCoco(unittest.TestCase):
    """Test cases for conversion from Poly2D to COCO RLE/polygons."""

    def test_poly2ds_to_mask(self) -> None:
        """Check the Poly2D to mask conversion."""
        json_file = get_test_file("poly2ds.json")
        npy_file = get_test_file("mask.npy")

        with open(json_file) as fp:
            polys = json.load(fp)
        gt_mask = np.load(npy_file).tolist()

        poly2ds = [Poly2D(**poly) for poly in polys]
        mask = poly2ds_to_mask(SHAPE, poly2ds).tolist()
        self.assertListEqual(mask, gt_mask)

    def test_mask_to_polygon(self) -> None:
        """Check the mask to polygon conversion."""
        npy_file = get_test_file("mask.npy")
        poly_file = get_test_file("polygon.npy")

        mask = np.load(npy_file).tolist()
        gt_polygon = np.load(poly_file).tolist()

        polygon = mask_to_polygon(mask, 0, 0)
        self.assertEqual(polygon, gt_polygon)

    def test_mask_to_rle(self) -> None:
        """Check the mask to polygon conversion."""
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

    scalabel = read(get_test_file("scalabel_det.json"))
    categories, name_mapping, ignore_mapping = load_default_cfgs("det")
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
        len_scalabel = sum([len(item.labels) for item in self.scalabel])
        len_coco = len(self.coco["annotations"])
        self.assertEqual(len_scalabel, len_coco)


if __name__ == "__main__":
    unittest.main()
