"""Test cases for evaluation scripts."""
import os
import unittest

from ..label.io import load, load_label_config
from ..unittest.util import get_test_file
from .det import evaluate_det
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
    result = evaluate_det(gts, preds, config, nproc=1)
    res_dict = result_to_flatten_dict(result)
    data_frame = nested_dict_to_data_frame(
        result_to_nested_dict(
            result, result._all_classes  # pylint: disable=protected-access
        )
    )

    def test_result_value(self) -> None:
        """Check evaluation scores' correctness."""
        print(self.result)
        overall_reference = {
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
        self.assertDictEqual(self.res_dict, overall_reference)

    def test_data_frame(self) -> None:
        """Check evaluation scores' correctness."""
        self.assertTupleEqual(self.data_frame.shape, (11, 12))

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
