"""Test cases for utils.py."""
import os
import unittest

from scalabel.label.io import load, load_label_config

from .utils import (
    check_overlap,
    check_overlap_frame,
    combine_stuff_masks,
    label_ids_to_int,
    reorder_preds,
)


class TestEvalUtils(unittest.TestCase):
    """Test cases for evaluation utils."""

    cur_dir = os.path.dirname(os.path.abspath(__file__))

    def test_label_ids_to_int(self) -> None:
        """Test label_ids_to_int function."""
        gt_file = f"{self.cur_dir}/testcases/pan_seg/pan_seg_sample.json"
        gt_frames = load(gt_file).frames[0]
        assert gt_frames.labels is not None
        old_ids = [label.id for label in gt_frames.labels]
        label_ids_to_int([gt_frames])
        new_ids = [label.id for label in gt_frames.labels]
        self.assertEqual(len(old_ids), len(new_ids))
        self.assertEqual(len(new_ids), len(set(new_ids)))
        for new_id in new_ids:
            self.assertTrue(isinstance(new_id, str))
            self.assertTrue(new_id.isdigit())

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
        self.assertFalse(check_overlap(pred_frames, config, nproc=1))

    def test_combine_stuff_masks(self) -> None:
        """Test combine_stuff_masks function."""
        pred_file = f"{self.cur_dir}/testcases/utils/overlap_masks.json"
        config = load_label_config(
            f"{self.cur_dir}/testcases/pan_seg/pan_seg_configs.toml"
        )
        pred_frames = load(pred_file).frames
        categories = config.categories
        category_names = [category.name for category in categories]
        rles, cids, iids = [], [], []
        assert pred_frames[0].labels is not None
        for label in pred_frames[0].labels:
            assert label.category is not None
            assert label.rle is not None
            rles.append(label.rle)
            cids.append(category_names.index(label.category))
            iids.append(int(label.id))
        out_rle, out_cid, out_iid = combine_stuff_masks(
            rles, cids, iids, categories
        )
        self.assertEqual(len(out_rle), 3)
        self.assertEqual(len(out_cid), 3)
        self.assertEqual(len(out_iid), 3)
        self.assertEqual(out_cid, [0, 34, 34])
        self.assertEqual(len(out_iid), len(set(out_iid)))

    def test_reorder_preds(self) -> None:
        """Test reorder_preds function."""
        pred_file = f"{self.cur_dir}/testcases/utils/preds.json"
        pred_frames = load(pred_file).frames
        gt_file = f"{self.cur_dir}/testcases/utils/gts.json"
        gt_frames = load(gt_file).frames
        pred_frames = reorder_preds(gt_frames, pred_frames)
        self.assertEqual(len(pred_frames), len(gt_frames))
        self.assertEqual(len(set(f.videoName for f in pred_frames)), 2)
        for frame in pred_frames:
            assert frame.labels is not None
            self.assertGreater(len(frame.labels), 0)
