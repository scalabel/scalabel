"""Test cases for evaluation scripts."""
import os
import unittest

import numpy as np

from ..label.io import load, load_label_config
from ..unittest.util import get_test_file
from .detect import evaluate_det


class TestBDD100KDetectEval(unittest.TestCase):
    """Test cases for BDD100K detection evaluation."""

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    gts_path = "{}/testcases/track_sample_anns.json".format(cur_dir)
    preds_path = "{}/testcases/bbox_predictions.json".format(cur_dir)
    gts = load(gts_path).frames
    preds = load(preds_path).frames
    config = load_label_config(get_test_file("det_configs.toml"))
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
        APs = np.array(  # pylint: disable=invalid-name
            [
                41.37670388,
                41.70985554,
                66.14185519,
                50.07078219,
                0.30253025,
                -100.0,
                4.57462895,
                -100.0,
                -100.0,
                -100,
                34.02939267,
            ]
        )
        self.assertTrue(np.isclose(data_arr[:, 0], APs).all())

        overall_scores = np.array(
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
            ]
        )
        self.assertTrue(np.isclose(data_arr[-1], overall_scores).all())

    def test_summary(self) -> None:
        """Check evaluation scores' correctness."""
        summary = self.result.summary()
        overall_reference = {
            "AP/pedestrian": 41.37670388442971,
            "AP/rider": 41.70985553789942,
            "AP/car": 66.14185518719762,
            "AP/truck": 50.07078219289907,
            "AP/bus": 0.3025302530253025,
            "AP/train": -100.0,
            "AP/motorcycle": 4.574628954954978,
            "AP/bicycle": -100.0,
            "AP/traffic light": -100.0,
            "AP/traffic sign": -100.0,
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
        for name, score in overall_reference.items():
            self.assertAlmostEqual(score, summary[name])
