"""Test cases for evaluation scripts."""
import os
import unittest

import numpy as np

from ..common.typing import NDArrayF64
from ..label.io import load, load_label_config
from ..unittest.util import get_test_file
from .tagging import evaluate_tagging


class TestScalabelTaggingEval(unittest.TestCase):
    """Test cases for Scalabel tagging evaluation."""

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    gts_path = f"{cur_dir}/testcases/tagging/tag_gts.json"
    preds_path = f"{cur_dir}/testcases/tagging/tag_preds.json"
    gts = load(gts_path).frames
    preds = load(preds_path).frames
    config = load_label_config(get_test_file("tagging/tag_configs.toml"))
    result = evaluate_tagging(gts, preds, config, "weather", nproc=1)

    def test_frame(self) -> None:
        """Test case for the function frame()."""
        data_frame = self.result.pd_frame()
        categories = set(
            [
                "rainy",
                "snowy",
                "clear",
                "overcast",
                "undefined",
                "partly cloudy",
                "foggy",
                "AVERAGE",
            ]
        )
        self.assertSetEqual(categories, set(data_frame.index.values))

        data_arr = data_frame.to_numpy()
        aps: NDArrayF64 = np.array(
            [100.0, -1.0, 100, 100, 100, 0.0, 0.0, 66.66666667],
            dtype=np.float64,
        )
        self.assertTrue(
            np.isclose(np.nan_to_num(data_arr[:, 0], nan=-1.0), aps).all()
        )

        overall_scores: NDArrayF64 = np.array(
            [66.66666667, 61.11111111, 63.33333333, 80.0], dtype=np.float64
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
            "precision": 66.66666666666666,
            "recall": 61.11111111111111,
            "f1_score": 63.33333333333333,
            "accuracy": 80.0,
        }
        self.assertSetEqual(set(summary.keys()), set(overall_reference.keys()))
        for name, score in summary.items():
            if np.isnan(score):
                score = np.nan_to_num(score, nan=-1.0)
            self.assertAlmostEqual(score, overall_reference[name])
