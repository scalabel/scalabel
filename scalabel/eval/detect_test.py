"""Test cases for evaluation scripts."""
import os
import unittest

import numpy as np

from ..common.typing import NDArrayF64
from ..label.io import load, load_label_config
from ..unittest.util import get_test_file
from .detect import evaluate_det


class TestScalabelDetectEval(unittest.TestCase):
    """Test cases for Scalabel detection evaluation."""

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    gts_path = f"{cur_dir}/testcases/box_track/track_sample_anns.json"
    preds_path = f"{cur_dir}/testcases/det/bbox_predictions.json"
    gts = load(gts_path).frames
    preds = load(preds_path).frames
    config = load_label_config(get_test_file("det/det_configs.toml"))
    result = evaluate_det(gts, preds, config, nproc=1)

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
                "traffic light",
                "traffic sign",
                "OVERALL",
            ]
        )
        self.assertSetEqual(categories, set(data_frame.index.values))

        data_arr = data_frame.to_numpy()
        aps: NDArrayF64 = np.array(
            [
                41.37670388,
                41.70985554,
                66.14185519,
                50.07078219,
                0.30253025,
                -1.0,
                4.57462895,
                -1.0,
                -1.0,
                -1.0,
                34.02939267,
            ],
            dtype=np.float64,
        )
        self.assertTrue(
            np.isclose(np.nan_to_num(data_arr[:, 0], nan=-1.0), aps).all()
        )

        overall_scores: NDArrayF64 = np.array(
            [
                34.02939267,
                55.32390041,
                34.9387336,
                20.91624814,
                48.43646525,
                64.28530467,
                23.8773384,
                38.05003678,
                40.9516045,
                25.55304933,
                58.38594872,
                66.04261954,
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
            "AP/pedestrian": 41.37670388442971,
            "AP/rider": 41.70985553789942,
            "AP/car": 66.14185518719762,
            "AP/truck": 50.07078219289907,
            "AP/bus": 0.3025302530253025,
            "AP/train": -1.0,
            "AP/motorcycle": 4.574628954954978,
            "AP/bicycle": -1.0,
            "AP/traffic light": -1.0,
            "AP/traffic sign": -1.0,
            "AP": 34.029392668401016,
            "AP50": 55.323900409039695,
            "AP75": 34.93873359997877,
            "APs": 20.91624813537171,
            "APm": 48.4364652499885,
            "APl": 64.28530466767323,
            "AR1": 23.877338403079264,
            "AR10": 38.050036781867405,
            "AR100": 40.95160450224777,
            "ARs": 25.5530493279957,
            "ARm": 58.38594871794871,
            "ARl": 66.04261954261955,
        }
        self.assertSetEqual(set(summary.keys()), set(overall_reference.keys()))
        for name, score in summary.items():
            if np.isnan(score):
                score = np.nan_to_num(score, nan=-1.0)
            self.assertAlmostEqual(score, overall_reference[name])


class TestScalabelDetectEvalEmpty(unittest.TestCase):
    """Test cases for Scalabel detection evaluation on empty test cases."""

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    gts_path = f"{cur_dir}/testcases/box_track/track_sample_anns.json"
    preds_path = f"{cur_dir}/testcases/det/bbox_predictions_empty.json"
    gts = load(gts_path).frames
    preds = load(preds_path).frames
    config = load_label_config(get_test_file("det/det_configs.toml"))
    result = evaluate_det(gts, preds, config, nproc=1)

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
                "traffic light",
                "traffic sign",
                "OVERALL",
            ]
        )
        self.assertSetEqual(categories, set(data_frame.index.values))

        data_arr = data_frame.to_numpy()
        aps: NDArrayF64 = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, -1.0, -1.0, 0.0],
            dtype=np.float64,
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
            "AP/bus": 0.0,
            "AP/train": -1.0,
            "AP/motorcycle": 0.0,
            "AP/bicycle": -1.0,
            "AP/traffic light": -1.0,
            "AP/traffic sign": -1.0,
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
