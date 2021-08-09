"""Test cases for evaluation scripts."""
import os
import unittest

from ..label.io import load, load_label_config
from ..unittest.util import get_test_file
from .det import METRICS, evaluate_det


class TestBDD100KDetectEval(unittest.TestCase):
    """Test cases for BDD100K detection evaluation."""

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    gts_path = "{}/testcases/track_sample_anns.json".format(cur_dir)
    preds_path = "{}/testcases/bbox_predictions.json".format(cur_dir)
    gts = load(gts_path).frames
    preds = load(preds_path).frames
    config = load_label_config(get_test_file("det_configs.toml"))
    result = evaluate_det(gts, preds, config)
    res_dict = result.res_dict
    data_frame = result.data_frame

    def test_result_value(self) -> None:
        """Check evaluation scores' correctness."""
        overall_reference = {
            "AP": 0.3402939266840102,
            "AP50": 0.553239004090397,
            "AP75": 0.34938733599978766,
            "APs": 0.2091624813537171,
            "APm": 0.48436465249988514,
            "APl": 0.6428530466767323,
            "AR1": 0.23877338403079265,
            "AR10": 0.3805003678186741,
            "AR100": 0.40951604502247774,
            "ARs": 0.2555304932799571,
            "ARm": 0.5838594871794872,
            "ARl": 0.6604261954261954,
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
