"""Multi-object Segmentation Tracking evaluation code."""
import argparse
import json
import time
from typing import List

from scalabel.common.io import open_write_text
from scalabel.common.logger import logger
from scalabel.common.parallel import NPROC
from scalabel.label.io import group_and_sort, load, load_label_config
from scalabel.label.typing import Config, Frame

from .box_track import bdd100k_to_scalabel, BoxTrackResult
from .hotas import evaluate_seg_track_hota
from .mots import acc_single_video_mots, evaluate_seg_track


def evaluate_seg_track_all(
    gts: List[List[Frame]],
    results: List[List[Frame]],
    config: Config,
    iou_thr: float = 0.5,
    ignore_iof_thr: float = 0.5,
    ignore_unknown_cats: bool = False,
    nproc: int = NPROC,
) -> BoxTrackResult:
    """Evaluate seg track metrics for a Scalabel format dataset.

    Args:
        gts: the ground truth annotations in Scalabel format
        results: the prediction results in Scalabel format.
        config: Config object
        iou_thr: Minimum IoU for a mask to be considered a positive.
        ignore_iof_thr: Min. Intersection over foreground with ignore regions.
        ignore_unknown_cats: if False, raise KeyError when trying to evaluate
            unknown categories.
        nproc: processes number for loading files

    Returns:
        BoxTrackResult: evaluation results.
    """
    logger.info("Seg tracking evaluation.")
    t = time.time()
    gts = [bdd100k_to_scalabel(gt, config) for gt in gts]
    results = [bdd100k_to_scalabel(result, config) for result in results]
    mot_result = evaluate_seg_track(
        acc_single_video_mots,
        gts,
        results,
        config,
        iou_thr,
        ignore_iof_thr,
        ignore_unknown_cats,
        nproc,
    )
    hota_result = evaluate_seg_track_hota(gts, results, config, nproc)
    result = BoxTrackResult(**{**mot_result.dict(), **hota_result.dict()})
    logger.info("evaluation finishes with %.1f s.", t)
    return result


def parse_arguments() -> argparse.Namespace:
    """Parse the arguments."""
    parser = argparse.ArgumentParser(description="Seg track evaluation.")
    parser.add_argument(
        "--gt", "-g", required=True, help="path to ground truth"
    )
    parser.add_argument(
        "--result", "-r", required=True, help="path to results"
    )
    parser.add_argument(
        "--config",
        "-c",
        default=None,
        help="Path to config toml file. Contains definition of categories, "
        "and optionally attributes as well as resolution. For an example "
        "see scalabel/label/testcases/configs.toml",
    )
    parser.add_argument(
        "--out-file", default="", help="Output path for evaluation results."
    )
    parser.add_argument(
        "--iou-thr", type=float, default=0.5, help="iou threshold"
    )
    parser.add_argument(
        "--ignore-iof-thr",
        type=float,
        default=0.5,
        help="ignore iof threshold",
    )
    parser.add_argument(
        "--ignore-unknown-cats",
        type=bool,
        default=False,
        help="ignore unknown categories",
    )
    parser.add_argument(
        "--nproc", "-p", type=int, default=NPROC, help="number of processes"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    dataset = load(args.gt)
    gt_frames, cfg = dataset.frames, dataset.config
    if args.config is not None:
        cfg = load_label_config(args.config)
    if cfg is None:
        raise ValueError(
            "Dataset config is not specified. Please use --config"
            " to specify a config for this dataset."
        )
    eval_result = evaluate_seg_track_all(
        group_and_sort(gt_frames),
        group_and_sort(load(args.result).frames),
        cfg,
        args.iou_thr,
        args.ignore_iof_thr,
        args.ignore_unknown_cats,
        args.nproc,
    )
    logger.info(eval_result)
    logger.info(eval_result.summary())
    if args.out_file:
        with open_write_text(args.out_file) as fp:
            json.dump(eval_result.dict(), fp, indent=2)
