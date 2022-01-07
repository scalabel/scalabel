"""Test cases for evaluation scripts."""
import os
import unittest

import numpy as np

from ..common.typing import NDArrayF64
from ..label.io import load, load_label_config
from ..unittest.util import get_test_file
from .sem_seg import evaluate_sem_seg


class TestScalabelSemSegEval(unittest.TestCase):
    """Test cases for Scalabel semantic segmentation evaluation."""

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    gts_path = f"{cur_dir}/testcases/sem_seg/sem_seg_sample.json"
    preds_path = f"{cur_dir}/testcases/sem_seg/sem_seg_preds.json"
    gts = load(gts_path).frames
    preds = load(preds_path).frames
    config = load_label_config(get_test_file("sem_seg/sem_seg_configs.toml"))
    result = evaluate_sem_seg(gts, preds, config, nproc=1)

    def test_frame(self) -> None:
        """Test case for the function frame()."""
        data_frame = self.result.pd_frame()
        categories = set(
            [
                "road",
                "sidewalk",
                "building",
                "wall",
                "fence",
                "pole",
                "traffic light",
                "traffic sign",
                "vegetation",
                "terrain",
                "sky",
                "person",
                "rider",
                "car",
                "bicycle",
                "bus",
                "motorcycle",
                "train",
                "truck",
                "AVERAGE",
            ]
        )
        self.assertSetEqual(categories, set(data_frame.index.values))

        data_arr = data_frame.to_numpy()
        aps: NDArrayF64 = np.array(
            [
                99.54405124,
                87.67803909,
                95.88390161,
                0.0,
                22.64488433,
                64.17477739,
                0.0,
                55.16129032,
                80.93376685,
                0.0,
                98.23104973,
                0.0,
                0.0,
                94.99133267,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                77.69367703,
            ],
            dtype=np.float64,
        )
        self.assertTrue(
            np.isclose(np.nan_to_num(data_arr[:, 0], nan=-1.0), aps).all()
        )

        overall_scores: NDArrayF64 = np.array(
            [77.69367703, 85.08073851], dtype=np.float64
        )
        self.assertTrue(
            np.isclose(
                np.nan_to_num(data_arr[-1], nan=-1.0), overall_scores
            ).all()
        )

    def test_summary(self) -> None:
        """Check evaluation scores' correctness."""
        summary = self.result.summary()
        overall_reference = {
            "mIoU": 77.69367702513847,
            "mAcc": 85.08073850792114,
            "fIoU": 95.96580841723501,
            "pAcc": 97.59265266143856,
        }
        self.assertSetEqual(set(summary.keys()), set(overall_reference.keys()))
        for name, score in summary.items():
            if np.isnan(score):
                score = np.nan_to_num(score, nan=-1.0)
            self.assertAlmostEqual(score, overall_reference[name])


class TestScalabelSemSegEvalEmpty(unittest.TestCase):
    """Test cases for Scalabel instance segmentation on empty test cases."""

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    gts_path = f"{cur_dir}/testcases/sem_seg/sem_seg_sample.json"
    preds_path = f"{cur_dir}/testcases/sem_seg/sem_seg_preds_empty.json"
    gts = load(gts_path).frames
    preds = load(preds_path).frames
    config = load_label_config(get_test_file("sem_seg/sem_seg_configs.toml"))
    result = evaluate_sem_seg(gts, preds, config, nproc=1)

    def test_frame(self) -> None:
        """Test case for the function frame()."""
        data_frame = self.result.pd_frame()
        categories = set(
            [
                "road",
                "sidewalk",
                "building",
                "fence",
                "wall",
                "pole",
                "traffic light",
                "traffic sign",
                "terrain",
                "vegetation",
                "sky",
                "person",
                "rider",
                "bicycle",
                "bus",
                "car",
                "motorcycle",
                "train",
                "truck",
                "AVERAGE",
            ]
        )
        self.assertSetEqual(categories, set(data_frame.index.values))

        data_arr = data_frame.to_numpy()
        aps: NDArrayF64 = np.array([0.0] * 20, dtype=np.float64)
        self.assertTrue(
            np.isclose(np.nan_to_num(data_arr[:, 0], nan=-1.0), aps).all()
        )

        overall_scores: NDArrayF64 = np.array([0.0] * 2, dtype=np.float64)
        self.assertTrue(
            np.isclose(
                np.nan_to_num(data_arr[-1], nan=-1.0), overall_scores
            ).all()
        )

    def test_summary(self) -> None:
        """Check evaluation scores' correctness."""
        summary = self.result.summary()
        overall_reference = {
            "mIoU": 0.0,
            "mAcc": 0.0,
            "fIoU": 0.0,
            "pAcc": 0.0,
        }
        self.assertSetEqual(set(summary.keys()), set(overall_reference.keys()))
        for name, score in summary.items():
            if np.isnan(score):
                score = np.nan_to_num(score, nan=-1.0)
            self.assertAlmostEqual(score, overall_reference[name])
