"""Test cases for transforms.py."""
import json
import unittest

import numpy as np

from ..common.io import open_read_text
from ..common.typing import NDArrayU8
from ..unittest.util import get_test_file
from .io import load
from .transforms import (
    bbox_to_box2d,
    box2d_to_bbox,
    coco_rle_to_rle,
    frame_to_masks,
    frame_to_rles,
    keypoints_to_nodes,
    mask_to_box2d,
    mask_to_rle,
    nodes_to_edges,
    poly2ds_to_mask,
    polygon_to_poly2ds,
    rle_to_box2d,
)
from .typing import Box2D, ImageSize, Poly2D

SHAPE = ImageSize(height=720, width=1280)


class TestCOCO2ScalabelFuncs(unittest.TestCase):
    """Test cases for conversion functions from COCO to Scalabel."""

    def test_bbox_to_box2d(self) -> None:
        """Check the function for bbox to Box2D."""
        bbox = [10.0, 10.0, 10.0, 10.0]
        box2d = bbox_to_box2d(bbox)
        gt_box2d = Box2D(x1=10, x2=19, y1=10, y2=19)
        self.assertEqual(box2d, gt_box2d)

    def test_mask_to_box2d(self) -> None:
        """Check the function for mask to Box2D."""
        mask: NDArrayU8 = np.zeros((10, 10), dtype=np.uint8)
        mask[4:6, 2:8] = 1
        mask[2:8, 4:6] = 1
        box2d = mask_to_box2d(mask)
        gt_box2d = Box2D(x1=2, x2=7, y1=2, y2=7)
        self.assertEqual(box2d, gt_box2d)

    def test_polygon_to_poly2ds(self) -> None:
        """Check the function for bbox to Box2D."""
        poly_file = get_test_file("polygon.npy")
        polygon = np.load(poly_file).tolist()

        poly2d = polygon_to_poly2ds(polygon)[0]
        vertices = poly2d.vertices
        types = poly2d.types

        self.assertTrue(poly2d.closed)
        self.assertEqual(len(vertices), len(types))
        for i, vertice in enumerate(vertices):
            self.assertAlmostEqual(vertice[0], polygon[0][2 * i])
            self.assertAlmostEqual(vertice[1], polygon[0][2 * i + 1])
        for c in types:
            self.assertEqual(c, "L")

    def test_coco_rle_to_rle(self) -> None:
        """Check the function for COCO RLE to Scalabel RLE."""
        json_file = get_test_file("coco_rle.json")
        with open(json_file, "r", encoding="utf-8") as fp:
            mask = json.load(fp)
        rle = coco_rle_to_rle(mask)
        gt_file = get_test_file("scalabel_rle.json")
        with open(gt_file, "r", encoding="utf-8") as fp:
            gt_rle = json.load(fp)
        self.assertEqual(rle.counts, gt_rle["counts"])
        self.assertEqual(rle.size, tuple(gt_rle["size"]))

    def test_keypoints_to_nodes(self) -> None:
        """Check the function for keypoints to Nodes."""
        keypoints = []
        for i in range(14):
            keypoints.extend([i, i + 50, i / 42.0])
        nodes = keypoints_to_nodes(keypoints)
        self.assertEqual(len(nodes), 14)
        for i, node in enumerate(nodes):
            self.assertEqual(len(node.id), 16)
            self.assertEqual(node.category, "coco_kpt")
            self.assertEqual(node.location, (float(i), float(i + 50)))
            self.assertEqual(node.score, i / 42.0)
        nodes = keypoints_to_nodes(keypoints, [str(i) for i in range(14)])
        for i, node in enumerate(nodes):
            self.assertEqual(node.category, str(i))

    def test_nodes_to_edges(self) -> None:
        """Check the function for Nodes to Edges."""
        keypoints = []
        for i in range(14):
            keypoints.extend([i, i + 50, i / 42.0])
        nodes = keypoints_to_nodes(keypoints)
        edge_map = {i: ([i + 1], str(i)) for i in range(13)}
        edges = nodes_to_edges(nodes, edge_map)
        self.assertEqual(len(edges), 13)
        for i, edge in enumerate(edges):
            self.assertEqual(edge.source, nodes[i].id)
            self.assertEqual(edge.target, nodes[i + 1].id)
            self.assertEqual(edge.type, str(i))


class TestScalabel2COCOFuncs(unittest.TestCase):
    """Test cases for conversion functions from Scalabel to COCO."""

    def test_box2d_to_bbox(self) -> None:
        """Check the Box2D to bbox conversion."""
        box2d = Box2D(x1=10, x2=29, y1=10, y2=19)
        bbox = box2d_to_bbox(box2d)
        self.assertListEqual(bbox, [10.0, 10.0, 20.0, 10.0])

    def test_poly2ds_to_mask(self) -> None:
        """Check the Poly2D to mask conversion."""
        json_file = get_test_file("poly2ds.json")
        npy_file = get_test_file("mask.npy")

        with open_read_text(json_file) as fp:
            polys = json.load(fp)
        gt_mask = np.load(npy_file).tolist()

        poly2ds = [Poly2D(**poly) for poly in polys]
        mask = poly2ds_to_mask(SHAPE, poly2ds).tolist()
        self.assertListEqual(mask, gt_mask)


class TestScalabelPoly2D2RLEFuncs(unittest.TestCase):
    """Test cases for conversion functions from Poly2Ds to RLE."""

    def test_frame_to_rles(self) -> None:
        """Check the Frame to RLE conversion."""
        json_file = get_test_file("scalabel_ins_seg.json")
        frames = load(json_file).frames
        for frame in frames:
            if frame.labels is None:
                continue
            poly2ds = [
                label.poly2d
                for label in frame.labels
                if label.poly2d is not None
            ]
            masks = frame_to_masks(SHAPE, poly2ds)
            rles_dt = [mask_to_rle(mask) for mask in masks]
            rles_gt = [
                label.rle for label in frame.labels if label.rle is not None
            ]
            for dt, gt in zip(rles_dt, rles_gt):
                self.assertEqual(dt, gt)
            rles_dt = frame_to_rles(SHAPE, poly2ds)
            for dt, gt in zip(rles_dt, rles_gt):
                self.assertEqual(dt, gt)

    def test_rle_to_box2d(self) -> None:
        """Check the RLE to Box2D conversion."""
        json_file = get_test_file("scalabel_ins_seg.json")
        frames = load(json_file).frames
        for frame in frames:
            if frame.labels is None:
                continue
            for label in frame.labels:
                if label.rle is None:
                    continue
                box2d = rle_to_box2d(label.rle)
                self.assertEqual(box2d, label.box2d)
