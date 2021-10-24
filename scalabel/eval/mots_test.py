"""Test cases for mots.py."""
import os
import unittest

import numpy as np

from ..label.io import group_and_sort, load, load_label_config
from ..unittest.util import get_test_file
from .mots import acc_single_video_mots, evaluate_seg_track


class TestBDD100KMotsEval(unittest.TestCase):
    """Test cases for BDD100K MOTS evaluation."""

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
    result = evaluate_seg_track(acc_single_video_mots, gts, preds, config)

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
        MOTAs = np.array(  # pylint: disable=invalid-name
            [
                15.6462585,
                -3.38983051,
                75.4674978,
                39.2405063,
                0.0,
                -1.0,
                11.8644068,
                -1.0,
                7.16981132,
                73.9502999,
                4.23728814,
                17.3536049,
                64.4092749,
            ]
        )
        data_arr_mota = data_arr[:, 0]
        data_arr_mota[np.abs(data_arr_mota) > 100] = np.nan
        self.assertTrue(
            np.isclose(np.nan_to_num(data_arr_mota, nan=-1.0), MOTAs).all()
        )

        overall_scores = np.array(
            [
                64.40927494,
                83.48730786,
                75.74533564,
                352.0,
                587.0,
                28.0,
                82.0,
                23.0,
                22.0,
                73.0,
            ]
        )
        self.assertTrue(np.isclose(data_arr[-1], overall_scores).all())

    def test_summary(self) -> None:
        """Check evaluation scores' correctness."""
        summary = self.result.summary()
        overall_reference = {
            "IDF1": 75.74533564146951,
            "MOTA": 64.40927493559072,
            "MOTP": 83.48730785942826,
            "FP": 352,
            "FN": 587,
            "IDSw": 28,
            "MT": 82,
            "PT": 23,
            "ML": 22,
            "FM": 73,
            "mIDF1": 32.189803601228505,
            "mMOTA": 17.35360485969023,
            "mMOTP": 46.1394286831572,
        }
        self.assertSetEqual(set(summary.keys()), set(overall_reference.keys()))
        for name, score in overall_reference.items():
            self.assertAlmostEqual(score, summary[name])
