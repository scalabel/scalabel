"""Test cases for evaluation scripts."""
import os
import unittest

import numpy as np

from ..common.typing import NDArrayF64
from ..label.io import load, load_label_config
from ..unittest.util import get_test_file
from .boundary import evaluate_boundary


class TestScalabelBoundaryEval(unittest.TestCase):
    """Test cases for Scalabel boundary evaluation."""

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    gts_path = f"{cur_dir}/testcases/boundary/boundary_gts.json"
    preds_path = f"{cur_dir}/testcases/boundary/boundary_preds.json"
    gts = load(gts_path).frames
    preds = load(preds_path).frames
    config = load_label_config(get_test_file("boundary/boundary_configs.toml"))
    result = evaluate_boundary(gts, preds, config, nproc=1)

    def test_frame(self) -> None:
        """Test case for the function frame()."""
        data_frame = self.result.pd_frame()
        categories = set(
            [
                "double other",
                "single yellow",
                "single other",
                "CATEGORY",
                "double yellow",
                "crosswalk",
                "AVERAGE",
                "solid",
                "DIRECTION",
                "dashed",
                "road curb",
                "STYLE",
                "vertical",
                "single white",
                "double white",
                "parallel",
            ]
        )
        self.assertSetEqual(categories, set(data_frame.index.values))

        data_arr = data_frame.to_numpy()
        aps: NDArrayF64 = np.array(
            [
                63.87878283,
                50.0,
                66.76055675,
                54.33678958,
                50.0,
                100.0,
                100.0,
                69.34481942,
                57.82339711,
                100.0,
                57.03808825,
                33.07663973,
                56.93939142,
                60.54867316,
                70.91036806,
                62.79947755,
            ],
            dtype=np.float64,
        )
        self.assertTrue(
            np.isclose(
                np.nan_to_num(data_arr[:, 0], nan=-1.0), aps, atol=1e-2
            ).all()
        )

        overall_scores: NDArrayF64 = np.array(
            [62.79947755, 70.62433628, 81.32172097], dtype=np.float64
        )
        self.assertTrue(
            np.isclose(
                np.nan_to_num(data_arr[-1], nan=-1.0),
                overall_scores,
                atol=1e-2,
            ).all()
        )

    def test_summary(self) -> None:
        """Check evaluation scores' correctness."""
        summary = self.result.summary()
        overall_reference = {
            "F1_pix1/DIRECTION": 56.93939141632752,
            "F1_pix1/STYLE": 60.548673163879585,
            "F1_pix1/CATEGORY": 70.91036806350797,
            "F1_pix1": 62.79947754790502,
            "F1_pix2/DIRECTION": 63.129709105365755,
            "F1_pix2/STYLE": 72.2857742015133,
            "F1_pix2/CATEGORY": 76.45752552306385,
            "F1_pix2": 70.62433627664764,
            "F1_pix5/DIRECTION": 71.03185350668716,
            "F1_pix5/STYLE": 88.64525818744838,
            "F1_pix5/CATEGORY": 84.28805121632493,
            "F1_pix5": 81.32172097015349,
        }
        self.assertSetEqual(set(summary.keys()), set(overall_reference.keys()))
        for name, score in summary.items():
            if np.isnan(score):
                score = np.nan_to_num(score, nan=-1.0)
            self.assertAlmostEqual(score, overall_reference[name])
