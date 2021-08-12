"""Test cases for mot.py."""
import os
import unittest

from ..label.io import group_and_sort, load, load_label_config
from ..unittest.util import get_test_file
from .box_track import METRIC_MAPS, acc_single_video_mot, evaluate_track
from .result import (
    nested_dict_to_data_frame,
    result_to_flatten_dict,
    result_to_nested_dict,
)


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
    res_dict = result_to_flatten_dict(result)
    data_frame = nested_dict_to_data_frame(
        result_to_nested_dict(
            result, result._all_classes  # pylint: disable=protected-access
        )
    )

    def test_result_value(self) -> None:
        """Check evaluation scores' correctness."""
        overall_reference = {
            "mMOTA": 24.324204637536685,
            "mMOTP": 50.01285067174096,
            "mIDF1": 32.24781943655838,
            "MOTA": 64.20070762302992,
            "MOTP": 87.16143945073146,
            "IDF1": 71.01073676416142,
            "FP": 126,
            "FN": 942,
            "IDSw": 45,
            "MT": 62,
            "PT": 47,
            "ML": 33,
            "FM": 66,
        }
        self.assertDictEqual(self.res_dict, overall_reference)

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
