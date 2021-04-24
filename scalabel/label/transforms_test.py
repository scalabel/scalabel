"""Test cases for transforms.py."""
import json
import unittest

import numpy as np

from ..unittest.util import get_test_file
from .transforms import (
    bbox_to_box2d,
    box2d_to_bbox,
    mask_to_box2d,
    mask_to_polygon,
    poly2ds_to_mask,
    polygon_to_poly2ds,
)
from .typing import Box2D, Poly2D

SHAPE = (720, 1280)


class TestCOCO2ScalabelFuncs(unittest.TestCase):
    """Test cases for conversion functions from COCO to Scalabel."""

    def test_bbox_to_box2d(self) -> None:
        """Check the function for bbox to Box2D."""
        bbox = [10.0, 10.0, 10.0, 10.0]
        box_2d = bbox_to_box2d(bbox)
        gt_box_2d = Box2D(x1=10, x2=19, y1=10, y2=19)
        self.assertEqual(box_2d, gt_box_2d)

    def test_mask_to_box2d(self) -> None:
        """Check the function for mask to Box2D."""
        mask = np.zeros((10, 10))
        mask[4:6, 2:8] = 1
        mask[2:8, 4:6] = 1
        box_2d = mask_to_box2d(mask)
        gt_box_2d = Box2D(x1=2, x2=7, y1=2, y2=7)
        self.assertEqual(box_2d, gt_box_2d)

    def test_polygon_to_poly2ds(self) -> None:
        """Check the function for bbox to Box2D."""
        poly_file = get_test_file("polygon.npy")
        polygon = np.load(poly_file).tolist()

        poly_2d = polygon_to_poly2ds(polygon)[0]
        vertices = poly_2d.vertices
        types = poly_2d.types

        self.assertTrue(poly_2d.closed)
        self.assertEqual(len(vertices), len(types))
        for i, vertice in enumerate(vertices):
            self.assertAlmostEqual(vertice[0], polygon[0][2 * i])
            self.assertAlmostEqual(vertice[1], polygon[0][2 * i + 1])
        for c in types:
            self.assertEqual(c, "L")


class TestScalabel2COCOFuncs(unittest.TestCase):
    """Test cases for conversion functions from Scalabel to COCO."""

    def test_box2d_to_bbox(self) -> None:
        """Check the Box2D to bbox conversion."""
        box_2d = Box2D(x1=10, x2=29, y1=10, y2=19)
        bbox = box2d_to_bbox(box_2d)
        self.assertListEqual(bbox, [10.0, 10.0, 20.0, 10.0])

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


if __name__ == "__main__":
    unittest.main()
