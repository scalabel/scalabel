"""Test cases for pan_seg.py."""
import os
import unittest

from scalabel.label.io import load, load_label_config
from scalabel.label.typing import Category

from .pan_seg import PQStat, evaluate_pan_seg


class TestPQStat(unittest.TestCase):
    """Test cases for the class PQStat."""

    category = Category(name="", isThing=True)
    pq_a = PQStat([category])
    pq_a.pq_per_cats[0].iou += 0.9
    pq_a.pq_per_cats[0].tpos += 1
    pq_b = PQStat([category])
    pq_b.pq_per_cats[0].fpos += 1
    pq_a += pq_b

    def test_iadd(self) -> None:
        """Check the correctness of __iadd__."""
        self.assertEqual(self.pq_a[0].tpos, 1)
        self.assertEqual(self.pq_a[0].fpos, 1)
        self.assertEqual(self.pq_a[0].iou, 0.9)

    def test_pq_average_zero_case(self) -> None:
        """Check the correctness of average when n == 0."""
        result = PQStat([self.category]).pq_average([])
        for val in result.values():
            self.assertAlmostEqual(val, 0)

    def test_pq_average_common_case(self) -> None:
        """Check the correctness of average when n == 1."""
        result = self.pq_a.pq_average([self.category])
        self.assertAlmostEqual(result["PQ"], 60.0)
        self.assertAlmostEqual(result["SQ"], 90.0)
        self.assertAlmostEqual(result["RQ"], 66.6666666666)


class TestScalabelPanSegEval(unittest.TestCase):
    """Test cases for Scalabel panoptic segmentation evaluation."""

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    gt_file = f"{cur_dir}/testcases/pan_seg/pan_seg_sample.json"
    pred_file = f"{cur_dir}/testcases/pan_seg/pan_seg_preds.json"
    config = load_label_config(
        f"{cur_dir}/testcases/pan_seg/pan_seg_configs.toml"
    )

    def test_summary(self) -> None:
        """Check evaluation scores' correctnes."""
        gt_frames = load(self.gt_file).frames
        pred_frames = load(self.pred_file).frames
        result = evaluate_pan_seg(gt_frames, pred_frames, self.config, nproc=1)
        summary = result.summary()
        gt_summary = {
            "PQ": 26.671487325762055,
            "PQ/STUFF": 32.04431646856032,
            "PQ/THING": 5.180170754569009,
            "SQ": 30.815623929417264,
            "SQ/STUFF": 32.04431646856032,
            "SQ/THING": 25.90085377284505,
            "RQ": 34.666666666666664,
            "RQ/STUFF": 41.666666666666664,
            "RQ/THING": 6.666666666666667,
            "NUM": 15,
            "NUM/STUFF": 12,
            "NUM/THING": 3,
        }
        self.assertSetEqual(set(summary.keys()), set(gt_summary.keys()))
        for name, score in gt_summary.items():
            self.assertAlmostEqual(score, summary[name])


class TestScalabelPanSegEvalMissing(unittest.TestCase):
    """Test cases for Scalabel panoptic segmentation with missing preds."""

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    gt_file = f"{cur_dir}/testcases/pan_seg/pan_seg_sample_2.json"
    pred_file = f"{cur_dir}/testcases/pan_seg/pan_seg_preds.json"
    config = load_label_config(
        f"{cur_dir}/testcases/pan_seg/pan_seg_configs.toml"
    )

    def test_summary(self) -> None:
        """Check evaluation scores' correctnes."""
        gt_frames = load(self.gt_file).frames
        pred_frames = load(self.pred_file).frames
        result = evaluate_pan_seg(gt_frames, pred_frames, self.config, nproc=1)
        summary = result.summary()
        gt_summary = {
            "PQ": 17.780991550508038,
            "PQ/STUFF": 21.362877645706877,
            "PQ/THING": 3.4534471697126734,
            "SQ": 30.815623929417264,
            "SQ/STUFF": 32.04431646856032,
            "SQ/THING": 25.90085377284505,
            "RQ": 23.111111111111104,
            "RQ/STUFF": 27.77777777777777,
            "RQ/THING": 4.444444444444445,
            "NUM": 15,
            "NUM/STUFF": 12,
            "NUM/THING": 3,
        }
        self.assertSetEqual(set(summary.keys()), set(gt_summary.keys()))
        for name, score in gt_summary.items():
            self.assertAlmostEqual(score, summary[name])


class TestScalabelPanSegEvalEmpty(unittest.TestCase):
    """Test cases for Scalabel panoptic segmentation on empty test cases."""

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    gt_file = f"{cur_dir}/testcases/pan_seg/pan_seg_sample.json"
    pred_file = f"{cur_dir}/testcases/pan_seg/pan_seg_preds_empty.json"
    config = load_label_config(
        f"{cur_dir}/testcases/pan_seg/pan_seg_configs.toml"
    )

    def test_summary(self) -> None:
        """Check evaluation scores' correctnes."""
        gt_frames = load(self.gt_file).frames
        pred_frames = load(self.pred_file).frames
        result = evaluate_pan_seg(gt_frames, pred_frames, self.config, nproc=1)
        summary = result.summary()
        gt_summary = {
            "PQ": 0.0,
            "PQ/STUFF": 0.0,
            "PQ/THING": 0.0,
            "SQ": 0.0,
            "SQ/STUFF": 0.0,
            "SQ/THING": 0.0,
            "RQ": 0.0,
            "RQ/STUFF": 0.0,
            "RQ/THING": 0.0,
            "NUM": 15,
            "NUM/STUFF": 12,
            "NUM/THING": 3,
        }
        self.assertSetEqual(set(summary.keys()), set(gt_summary.keys()))
        for name, score in gt_summary.items():
            self.assertAlmostEqual(score, summary[name])
