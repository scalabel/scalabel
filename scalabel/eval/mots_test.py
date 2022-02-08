"""Test cases for mots.py."""
import os
import unittest

import numpy as np

from ..common.typing import NDArrayF64
from ..label.io import group_and_sort, load, load_label_config
from ..unittest.util import get_test_file
from .mots import acc_single_video_mots, evaluate_seg_track


class TestScalabelMotsEval(unittest.TestCase):
    """Test cases for Scalabel MOTS evaluation."""

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    gts = group_and_sort(
        load(f"{cur_dir}/testcases/seg_track/seg_track_sample.json").frames
    )
    preds = group_and_sort(
        load(f"{cur_dir}/testcases/seg_track/seg_track_preds.json").frames
    )
    config = load_label_config(
        get_test_file("seg_track/seg_track_configs.toml")
    )
    result = evaluate_seg_track(
        acc_single_video_mots, gts, preds, config, nproc=1
    )

    def test_frame(self) -> None:
        """Test case for the function frame()."""
        data_frame = self.result.pd_frame()
        categories = set(
            [
                "human",
                "vehicle",
                "bike",
                "pedestrian",
                "rider",
                "car",
                "truck",
                "bus",
                "train",
                "motorcycle",
                "bicycle",
                "AVERAGE",
                "OVERALL",
            ]
        )
        self.assertSetEqual(categories, set(data_frame.index.values))

        data_arr = data_frame.to_numpy()
        motas: NDArrayF64 = np.array(
            [
                -1.0,
                -6.06060606,
                77.73224044,
                39.24050633,
                -1.0,
                -1.0,
                14.14141414,
                -1.0,
                -44.03669725,
                75.76150356,
                5.05050505,
                -36.86830564,
                64.30611079,
            ],
            dtype=np.float64,
        )
        data_arr_mota = data_arr[:, 0]
        data_arr_mota[np.abs(data_arr_mota) > 100] = np.nan
        self.assertTrue(
            np.isclose(np.nan_to_num(data_arr_mota, nan=-1.0), motas).all()
        )

        overall_scores: NDArrayF64 = np.array(
            [
                64.30611079,
                83.57249803,
                74.5157385,
                201.0,
                399.0,
                25.0,
                56.0,
                19.0,
                11.0,
                42.0,
            ],
            dtype=np.float64,
        )
        self.assertTrue(np.isclose(data_arr[-1], overall_scores).all())

    def test_summary(self) -> None:
        """Check evaluation scores' correctness."""
        summary = self.result.summary()
        overall_reference = {
            "IDF1": 74.51573849878935,
            "MOTA": 64.30611079383209,
            "MOTP": 83.57249803157146,
            "FP": 201,
            "FN": 399,
            "IDSw": 25,
            "MT": 56,
            "PT": 19,
            "ML": 11,
            "FM": 42,
            "mIDF1": 26.458539514581545,
            "mMOTA": -36.86830564411494,
            "mMOTP": 45.791906065134405,
        }
        self.assertSetEqual(set(summary.keys()), set(overall_reference.keys()))
        for name, score in overall_reference.items():
            self.assertAlmostEqual(score, summary[name])


class TestScalabelMotsEvalEmpty(unittest.TestCase):
    """Test cases for Scalabel MOTS evaluation on empty predictions."""

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    gts = group_and_sort(
        load(f"{cur_dir}/testcases/seg_track/seg_track_sample.json").frames
    )
    preds = group_and_sort(
        load(
            f"{cur_dir}/testcases/seg_track/seg_track_preds_empty.json"
        ).frames
    )
    config = load_label_config(
        get_test_file("seg_track/seg_track_configs.toml")
    )
    result = evaluate_seg_track(
        acc_single_video_mots, gts, preds, config, nproc=1
    )

    def test_frame(self) -> None:
        """Test case for the function frame()."""
        data_frame = self.result.pd_frame()
        categories = set(
            [
                "human",
                "vehicle",
                "bike",
                "pedestrian",
                "rider",
                "car",
                "truck",
                "bus",
                "train",
                "motorcycle",
                "bicycle",
                "AVERAGE",
                "OVERALL",
            ]
        )
        self.assertSetEqual(categories, set(data_frame.index.values))

        data_arr = data_frame.to_numpy()
        motas: NDArrayF64 = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                -1.0,
                -1.0,
                0.0,
                -1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )
        data_arr_mota = data_arr[:, 0]
        data_arr_mota[np.abs(data_arr_mota) > 100] = np.nan
        self.assertTrue(
            np.isclose(np.nan_to_num(data_arr_mota, nan=-1.0), motas).all()
        )

        overall_scores: NDArrayF64 = np.array(
            [
                0.0,
                -1.0,
                0.0,
                0.0,
                1751.0,
                0.0,
                0.0,
                0.0,
                86.0,
                0.0,
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
            "IDF1": 0.0,
            "MOTA": 0.0,
            "MOTP": -1.0,
            "FP": 0,
            "FN": 1751,
            "IDSw": 0,
            "MT": 0,
            "PT": 0,
            "ML": 86,
            "FM": 0,
            "mIDF1": 0.0,
            "mMOTA": 0.0,
            "mMOTP": 0.0,
        }
        self.assertSetEqual(set(summary.keys()), set(overall_reference.keys()))
        for name, score in overall_reference.items():
            self.assertAlmostEqual(
                score, np.nan_to_num(summary[name], nan=-1.0)
            )
