"""Test cases for evaluation scripts."""
import os
import unittest

import numpy as np

from ..label.io import load, load_label_config
from ..unittest.util import get_test_file
from .pose import evaluate_pose


class TestBDD100KPoseEval(unittest.TestCase):
    """Test cases for BDD100K pose estimation evaluation."""

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    gts_path = f"{cur_dir}/testcases/pose_sample.json"
    preds_path = f"{cur_dir}/testcases/pose_preds.json"
    gts = load(gts_path).frames
    preds = load(preds_path).frames
    config = load_label_config(get_test_file("pose_configs.toml"))
    result = evaluate_pose(gts, preds, config, nproc=1)

    def test_frame(self) -> None:
        """Test case for the function frame()."""
        data_frame = self.result.pd_frame()
        categories = set(["OVERALL"])
        self.assertSetEqual(categories, set(data_frame.index.values))

        data_arr = data_frame.to_numpy()
        APs = np.array([14.059405940594061])  # pylint: disable=invalid-name
        self.assertTrue(
            np.isclose(np.nan_to_num(data_arr[:, 0], nan=-1.0), APs).all()
        )

        overall_scores = np.array(
            [
                14.059405940594061,
                25.247524752475247,
                12.871287128712872,
                14.059405940594061,
                -1.0,
                27.500000000000004,
                50.0,
                25.0,
                27.500000000000004,
                -1.0,
            ]
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
            "AP": 14.059405940594061,
            "AP50": 25.247524752475247,
            "AP75": 12.871287128712872,
            "APm": 14.059405940594061,
            "APl": -1.0,
            "AR": 27.500000000000004,
            "AR50": 50.0,
            "AR75": 25.0,
            "ARm": 27.500000000000004,
            "ARl": -1.0,
        }
        self.assertSetEqual(set(summary.keys()), set(overall_reference.keys()))
        for name, score in summary.items():
            if np.isnan(score):
                score = np.nan_to_num(score, nan=-1.0)
            self.assertAlmostEqual(score, overall_reference[name])
