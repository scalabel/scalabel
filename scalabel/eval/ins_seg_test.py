"""Test cases for evaluation scripts."""
import os
import unittest

import numpy as np

from ..common.typing import NDArrayF64
from ..label.io import load, load_label_config
from ..unittest.util import get_test_file
from .ins_seg import evaluate_ins_seg


class TestScalabelInsSegEval(unittest.TestCase):
    """Test cases for Scalabel instance segmentation evaluation."""

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    gts_path = f"{cur_dir}/testcases/ins_seg/ins_seg_rle_sample.json"
    preds_path = f"{cur_dir}/testcases/ins_seg/ins_seg_preds.json"
    gts = load(gts_path).frames
    preds = load(preds_path).frames
    config = load_label_config(get_test_file("ins_seg/ins_seg_configs.toml"))
    result = evaluate_ins_seg(gts, preds, config, nproc=1)

    def test_frame(self) -> None:
        """Test case for the function frame()."""
        data_frame = self.result.pd_frame()
        categories = set(
            [
                "pedestrian",
                "rider",
                "car",
                "truck",
                "bus",
                "train",
                "motorcycle",
                "bicycle",
                "OVERALL",
            ]
        )
        self.assertSetEqual(categories, set(data_frame.index.values))

        data_arr = data_frame.to_numpy()
        aps: NDArrayF64 = np.array(
            [
                60.198019801980195,
                49.99999999999999,
                33.66336633663366,
                69.99999999999999,
                -1.0,
                -1.0,
                -1.0,
                29.999999999999993,
                48.77227722772277,
            ],
            dtype=np.float64,
        )
        self.assertTrue(
            np.isclose(np.nan_to_num(data_arr[:, 0], nan=-1.0), aps).all()
        )

        overall_scores: NDArrayF64 = np.array(
            [
                48.77227722772277,
                86.73267326732673,
                36.83168316831682,
                0.0,
                53.399339933993396,
                50.247524752475236,
                44.66666666666667,
                48.66666666666667,
                48.66666666666667,
                0.0,
                53.333333333333336,
                50.0,
            ],
            dtype=np.float64,
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
            "AP/pedestrian": 60.198019801980195,
            "AP/rider": 49.99999999999999,
            "AP/car": 33.66336633663366,
            "AP/truck": 69.99999999999999,
            "AP/bus": -1.0,
            "AP/train": -1.0,
            "AP/motorcycle": -1.0,
            "AP/bicycle": 29.999999999999993,
            "AP": 48.77227722772277,
            "AP50": 86.73267326732673,
            "AP75": 36.83168316831682,
            "APs": 0.0,
            "APm": 53.399339933993396,
            "APl": 50.247524752475236,
            "AR1": 44.66666666666667,
            "AR10": 48.66666666666667,
            "AR100": 48.66666666666667,
            "ARs": 0.0,
            "ARm": 53.333333333333336,
            "ARl": 50.0,
        }
        self.assertSetEqual(set(summary.keys()), set(overall_reference.keys()))
        for name, score in summary.items():
            if np.isnan(score):
                score = np.nan_to_num(score, nan=-1.0)
            self.assertAlmostEqual(score, overall_reference[name])


class TestScalabelInsSegEvalEmpty(unittest.TestCase):
    """Test cases for Scalabel instance segmentation on empty test cases."""

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    gts_path = f"{cur_dir}/testcases/ins_seg/ins_seg_rle_sample.json"
    preds_path = f"{cur_dir}/testcases/ins_seg/ins_seg_preds_empty.json"
    gts = load(gts_path).frames
    preds = load(preds_path).frames
    config = load_label_config(get_test_file("ins_seg/ins_seg_configs.toml"))
    result = evaluate_ins_seg(gts, preds, config, nproc=1)

    def test_frame(self) -> None:
        """Test case for the function frame()."""
        data_frame = self.result.pd_frame()
        categories = set(
            [
                "pedestrian",
                "rider",
                "car",
                "truck",
                "bus",
                "train",
                "motorcycle",
                "bicycle",
                "OVERALL",
            ]
        )
        self.assertSetEqual(categories, set(data_frame.index.values))

        data_arr = data_frame.to_numpy()
        aps: NDArrayF64 = np.array(
            [0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, 0.0, 0.0], dtype=np.float64
        )
        self.assertTrue(
            np.isclose(np.nan_to_num(data_arr[:, 0], nan=-1.0), aps).all()
        )

        overall_scores: NDArrayF64 = np.array([0.0] * 12, dtype=np.float64)
        self.assertTrue(
            np.isclose(
                np.nan_to_num(data_arr[-1], nan=-1.0), overall_scores
            ).all()
        )

    def test_summary(self) -> None:
        """Check evaluation scores' correctness."""
        summary = self.result.summary()
        overall_reference = {
            "AP/pedestrian": 0.0,
            "AP/rider": 0.0,
            "AP/car": 0.0,
            "AP/truck": 0.0,
            "AP/bus": -1.0,
            "AP/train": -1.0,
            "AP/motorcycle": -1.0,
            "AP/bicycle": 0.0,
            "AP": 0.0,
            "AP50": 0.0,
            "AP75": 0.0,
            "APs": 0.0,
            "APm": 0.0,
            "APl": 0.0,
            "AR1": 0.0,
            "AR10": 0.0,
            "AR100": 0.0,
            "ARs": 0.0,
            "ARm": 0.0,
            "ARl": 0.0,
        }
        self.assertSetEqual(set(summary.keys()), set(overall_reference.keys()))
        for name, score in summary.items():
            if np.isnan(score):
                score = np.nan_to_num(score, nan=-1.0)
            self.assertAlmostEqual(score, overall_reference[name])
