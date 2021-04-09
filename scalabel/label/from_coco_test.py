"""Test cases for bdd100k2coco.py."""
import unittest

import numpy as np

from ..unittest.util import get_test_file
from .from_coco import bbox_to_box2d, polygon_to_poly2ds
from .typing import Box2D


class TestBoxAndPolyConversions(unittest.TestCase):
    """Check the conversion functions for Box2D and Poly2D."""

    def test_bbox_to_box2d(self) -> None:
        """Check the function for bbox to Box2D."""
        bbox = [10.0, 10.0, 10.0, 10.0]
        box_2d = bbox_to_box2d(bbox)
        gt_box_2d = Box2D(x1=10, x2=19, y1=10, y2=19)
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
