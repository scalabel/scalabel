"""Test cases for mot.py."""
import os
import unittest

from ..label.io import group_and_sort, load, load_label_config
from ..unittest.util import get_test_file
from .box_track import METRIC_MAPS, acc_single_video_mot, evaluate_track


class TestBDD100KMotEval(unittest.TestCase):
    """Test cases for BDD100K MOT evaluation."""

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    gts = group_and_sort(
        load("{}/testcases/track_sample_anns.json".format(cur_dir)).frames
    )
    preds = group_and_sort(
        load("{}/testcases/track_predictions.json".format(cur_dir)).frames
    )
    config = load_label_config(get_test_file("box_track_configs.toml"))
    result = evaluate_track(acc_single_video_mot, gts, preds, config)
    res_dict = result.res_dict
    data_frame = result.data_frame

    def test_result_value(self) -> None:
        """Check evaluation scores' correctness."""
        overall_reference = {
            "IDF1": 0.7101073676416142,
            "MOTA": 0.6420070762302992,
            "MOTP": 0.871614396957838,
            "FP": 126,
            "FN": 942,
            "IDSw": 45,
            "MT": 62,
            "PT": 47,
            "ML": 33,
            "FM": 66,
            "mIDF1": 0.32247819436558384,
            "mMOTA": 0.24324204637536687,
            "mMOTP": 0.5001285135514636,
        }
        for key, val in overall_reference.items():
            self.assertAlmostEqual(self.res_dict[key], val)
        self.assertEqual(len(self.res_dict), len(overall_reference))

    def test_data_frame(self) -> None:
        """Check evaluation scores' correctness."""
        self.assertTupleEqual(self.data_frame.shape, (13, 10))

        metrics = set(METRIC_MAPS.values())
        self.assertSetEqual(metrics, set(self.data_frame.columns.values))

        categories = set(
            [
                "human",
                "pedestrian",
                "rider",
                "vehicle",
                "car",
                "truck",
                "bus",
                "train",
                "bike",
                "motorcycle",
                "bicycle",
                "OVERALL",
                "AVERAGE",
            ]
        )
        self.assertSetEqual(categories, set(self.data_frame.index.values))
        # for cat_name in self.data_frame.index:
        #     self.assertIn(cat_name, categories)
