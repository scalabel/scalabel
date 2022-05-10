"""Test cases for to_rle.py."""
import copy
import os
import shutil
import unittest

from .io import load, load_label_config
from .to_rle import seg_to_rles


class TestToRLEs(unittest.TestCase):
    """Test cases for converting scalabel labels to rles."""

    test_out = "./test_rles"

    def task_specific_test(self, task_name: str, file_name: str) -> None:
        """General test function for different tasks."""
        cur_dir = os.path.dirname(os.path.abspath(__file__))

        json_path = f"{cur_dir}/testcases/rles/pred.json"
        frames = load(json_path).frames
        assert frames[0].labels is not None
        scalabel_config = load_label_config(
            f"{cur_dir}/../eval/testcases/{task_name}/{task_name}_configs.toml"
        )
        dt_rle = seg_to_rles(copy.deepcopy(frames), scalabel_config, 1)
        gt_rle = load(f"{cur_dir}/testcases/rles/{file_name}").frames
        frames[0].labels = [
            label for label in frames[0].labels if label.poly2d is not None
        ]
        for gt, dt, og in zip(gt_rle, dt_rle, frames):
            self.assertEqual(gt.name, dt.name)
            self.assertEqual(gt.videoName, dt.videoName)
            self.assertEqual(gt.labels is None, dt.labels is None)
            if gt.labels is None or dt.labels is None or og.labels is None:
                continue
            for gt_label, dt_label, og_label in zip(
                gt.labels, dt.labels, og.labels
            ):
                self.assertEqual(gt_label.rle is None, dt_label.rle is None)
                self.assertTrue(dt_label.poly2d is None)
                self.assertEqual(og_label.poly2d is None, dt_label.rle is None)
                if gt_label.rle is None or dt_label.rle is None:
                    continue
                self.assertEqual(gt_label.rle.counts, dt_label.rle.counts)
                self.assertEqual(gt_label.rle.size, dt_label.rle.size)

    def test_semseg_to_rles(self) -> None:
        """Test case for semantic segmentation to rles."""
        self.task_specific_test("sem_seg", "semseg_rle.json")

    def test_insseg_to_rles(self) -> None:
        """Test case for instance segmentation to rles."""
        self.task_specific_test("ins_seg", "insseg_rle.json")

    def test_panseg_to_rles(self) -> None:
        """Test case for panoptic segmentation to rles."""
        self.task_specific_test("ins_seg", "panseg_rle.json")

    def test_segtrack_to_rles(self) -> None:
        """Test case for segmentation tracking to rles."""
        self.task_specific_test("seg_track", "segtrack")

    @classmethod
    def tearDownClass(cls) -> None:
        """Class teardown for bitmask tests."""
        if os.path.exists(cls.test_out):
            shutil.rmtree(cls.test_out)
