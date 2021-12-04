"""Test cases for utils.py."""
import os
import unittest

from scalabel.label.io import load, load_label_config

from .utils import check_overlap, check_overlap_frame


class TestEvalUtils(unittest.TestCase):
    """Test cases for evaluation utils."""

    cur_dir = os.path.dirname(os.path.abspath(__file__))

    def test_check_overlap_frame(self) -> None:
        """Test check_overlap_frame function."""
        pred_files = [
            f"{self.cur_dir}/testcases/utils/overlap_masks.json",
            f"{self.cur_dir}/testcases/ins_seg/ins_seg_preds.json",
            f"{self.cur_dir}/testcases/sem_seg/sem_seg_preds.json",
            f"{self.cur_dir}/testcases/seg_track/seg_track_preds.json",
            f"{self.cur_dir}/testcases/pan_seg/pan_seg_preds.json",
        ]
        overlaps = [True, False, False, False, False]
        for pred_file, is_overlap in zip(pred_files, overlaps):
            pred_frames = load(pred_file).frames
            self.assertEqual(
                check_overlap_frame(pred_frames[0], ["car"]), is_overlap
            )

    def test_check_overlap(self) -> None:
        """Test check_overlap function."""
        pred_file = f"{self.cur_dir}/testcases/utils/overlap_masks.json"
        config = load_label_config(
            f"{self.cur_dir}/testcases/pan_seg/pan_seg_configs.toml"
        )
        pred_frames = load(pred_file).frames
        self.assertTrue(check_overlap(pred_frames, config, nproc=1))
