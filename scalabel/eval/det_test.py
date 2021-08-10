"""Test cases for evaluation scripts."""
import os
import unittest

from ..label.io import load, load_label_config
from ..unittest.util import get_test_file
from .det import METRICS, evaluate_det
from .result import (
    nested_dict_to_data_frame,
    result_to_flatten_dict,
    result_to_nested_dict,
)


class TestBDD100KDetectEval(unittest.TestCase):
    """Test cases for BDD100K detection evaluation."""

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    gts_path = "{}/testcases/track_sample_anns.json".format(cur_dir)
    preds_path = "{}/testcases/bbox_predictions.json".format(cur_dir)
    gts = load(gts_path).frames
    preds = load(preds_path).frames
    config = load_label_config(get_test_file("det_configs.toml"))
    result = evaluate_det(gts, preds, config)
    res_dict = result_to_flatten_dict(result)
    data_frame = nested_dict_to_data_frame(
        result_to_nested_dict(
            result, result._all_classes  # pylint: disable=protected-access
        )
    )

    def test_result_value(self) -> None:
        """Check evaluation scores' correctness."""
        overall_reference = {
            "AP": 34.02939266840102,
            "AP50": 55.3239004090397,
            "AP75": 34.938733599978766,
            "APs": 20.91624813537171,
            "APm": 48.436465249988514,
            "APl": 64.28530466767323,
            "AR1": 23.877338403079265,
            "AR10": 38.05003678186741,
            "AR100": 40.951604502247774,
            "ARs": 25.55304932799571,
            "ARm": 58.38594871794872,
            "ARl": 66.04261954261954,
        }
        for key, val in self.res_dict.items():
            self.assertAlmostEqual(val, overall_reference[key])
        self.assertEqual(len(self.res_dict), len(overall_reference))

    def test_data_frame(self) -> None:
        """Check evaluation scores' correctness."""
        self.assertTupleEqual(self.data_frame.shape, (11, 12))
        self.assertSetEqual(set(METRICS), set(self.data_frame.columns.values))

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
        self.assertSetEqual(categories, set(self.data_frame.index.values))
