"""BDD100K segmentation tracking evaluation with CLEAR MOT metrics."""
import argparse
import json
import time
from functools import partial
from multiprocessing import Pool
from typing import Callable, List, Optional

import motmetrics as mm
import numpy as np
from pycocotools.mask import iou  # type: ignore

from scalabel.common.io import open_write_text
from scalabel.common.logger import logger
from scalabel.common.parallel import NPROC
from scalabel.common.typing import NDArrayU8
from scalabel.label.io import group_and_sort, load, load_label_config
from scalabel.label.typing import Category, Config, Frame, ImageSize
from scalabel.label.utils import get_leaf_categories, get_parent_categories

from .mot import (
    METRIC_MAPS,
    TrackResult,
    aggregate_accs,
    evaluate_single_class,
    generate_results,
)
from .utils import (
    check_overlap,
    handle_inconsistent_length,
    label_ids_to_int,
    parse_seg_objects,
)

VidFunc = Callable[
    [
        List[Frame],
        List[Frame],
        List[Category],
        float,
        float,
        bool,
        Optional[ImageSize],
    ],
    List[mm.MOTAccumulator],
]


def acc_single_video_mots(
    gts: List[Frame],
    results: List[Frame],
    classes: List[Category],
    iou_thr: float = 0.5,
    ignore_iof_thr: float = 0.5,
    ignore_unknown_cats: bool = False,
    image_size: Optional[ImageSize] = None,
) -> List[mm.MOTAccumulator]:
    """Accumulate results for one video."""
    assert len(gts) == len(results)

    def get_frame_index(frame: Frame) -> int:
        return frame.frameIndex if frame.frameIndex is not None else 0

    num_classes = len(classes)
    gts = sorted(gts, key=get_frame_index)
    results = sorted(results, key=get_frame_index)
    accs = [mm.MOTAccumulator(auto_id=True) for _ in range(num_classes)]

    label_ids_to_int(gts)

    for gt, result in zip(gts, results):
        assert gt.frameIndex == result.frameIndex
        gt_rles, gt_labels, gt_ids, gt_ignores = parse_seg_objects(
            gt.labels if gt.labels is not None else [],
            classes,
            ignore_unknown_cats,
            image_size,
        )
        pred_rles, pred_labels, pred_ids, _ = parse_seg_objects(
            result.labels if result.labels is not None else [],
            classes,
            ignore_unknown_cats,
            image_size,
        )
        for i in range(num_classes):
            gt_inds, pred_inds = gt_labels == i, pred_labels == i
            gt_rles_c = [
                gt_rles[j] for j, gt_ind in enumerate(gt_inds) if gt_ind
            ]
            pred_rles_c = [
                pred_rles[j]
                for j, pred_ind in enumerate(pred_inds)
                if pred_ind
            ]
            gt_ids_c, pred_ids_c = gt_ids[gt_inds], pred_ids[pred_inds]
            if len(gt_rles_c) == 0 and len(pred_rles_c) == 0:
                continue
            if len(gt_rles_c) == 0 and len(pred_rles_c) != 0:
                distances = np.full((0, len(pred_rles_c)), np.nan)
            elif len(gt_rles_c) != 0 and len(pred_rles_c) == 0:
                distances = np.full((len(gt_rles_c), 0), np.nan)
            else:
                ious_c = iou(
                    pred_rles_c,
                    gt_rles_c,
                    [False for _ in range(len(gt_rles_c))],
                ).T
                distances = 1 - ious_c
                distances = np.where(
                    distances > 1 - iou_thr, np.nan, distances
                )
            if len(gt_ignores) > 0 and len(pred_rles_c) > 0:
                # 1. assign gt and preds
                fps: NDArrayU8 = np.ones(len(pred_rles_c)).astype(bool)
                le, ri = mm.lap.linear_sum_assignment(distances)
                for m, n in zip(le, ri):
                    if np.isfinite(distances[m, n]):
                        fps[n] = False
                # 2. ignore by iof
                iofs = iou(
                    pred_rles_c,
                    gt_ignores,
                    [True for _ in range(len(gt_ignores))],
                )
                ignores: bool = np.greater(iofs, ignore_iof_thr).any(axis=1)
                # 3. filter preds
                valid_inds = np.logical_not(np.logical_and(fps, ignores))
                pred_ids_c = pred_ids_c[valid_inds]
                distances = distances[:, valid_inds]
            if distances.shape != (0, 0):
                accs[i].update(gt_ids_c, pred_ids_c, distances)
    return accs


def evaluate_seg_track(
    acc_single_video: VidFunc,
    gts: List[List[Frame]],
    results: List[List[Frame]],
    config: Config,
    iou_thr: float = 0.5,
    ignore_iof_thr: float = 0.5,
    ignore_unknown_cats: bool = False,
    nproc: int = NPROC,
) -> TrackResult:
    """Evaluate CLEAR MOT metrics for a Scalabel format dataset.

    Args:
        acc_single_video: Function for calculating metrics over a single video.
        gts: the ground truth annotations in Scalabel format
        results: the prediction results in Scalabel format.
        config: Config object
        iou_thr: Minimum IoU for a mask to be considered a positive.
        ignore_iof_thr: Min. Intersection over foreground with ignore regions.
        ignore_unknown_cats: if False, raise KeyError when trying to evaluate
            unknown categories.
        nproc: processes number for loading files

    Returns:
        TrackResult: evaluation results.
    """
    logger.info("Tracking evaluation with CLEAR MOT metrics.")
    t = time.time()
    results = handle_inconsistent_length(gts, results)
    assert len(gts) == len(results)
    # check overlap of masks
    logger.info("checking for overlap of masks...")
    if check_overlap(
        [frame for res in results for frame in res], config, nproc
    ):
        logger.critical(
            "Found overlap in prediction bitmasks, but segmentation tracking "
            "evaluation does not allow overlaps. Removing such predictions."
        )

    classes = get_leaf_categories(config.categories)
    super_classes = get_parent_categories(config.categories)

    logger.info("evaluating...")
    image_size = config.imageSize
    if nproc > 1:
        with Pool(nproc) as pool:
            video_accs = pool.starmap(
                partial(
                    acc_single_video,
                    classes=classes,
                    ignore_iof_thr=ignore_iof_thr,
                    ignore_unknown_cats=ignore_unknown_cats,
                    image_size=image_size,
                ),
                zip(gts, results),
            )
    else:
        video_accs = [
            acc_single_video(
                gt,
                result,
                classes,
                iou_thr,
                ignore_iof_thr,
                ignore_unknown_cats,
                image_size,
            )
            for gt, result in zip(gts, results)
        ]

    class_names, metric_names, class_accs = aggregate_accs(
        video_accs, classes, super_classes
    )

    logger.info("accumulating...")
    if nproc > 1:
        with Pool(nproc) as pool:
            flat_dicts = pool.starmap(
                evaluate_single_class, zip(metric_names, class_accs)
            )
    else:
        flat_dicts = [
            evaluate_single_class(names, accs)
            for names, accs in zip(metric_names, class_accs)
        ]

    metrics = list(METRIC_MAPS.values())
    result = generate_results(
        flat_dicts, class_names, metrics, classes, super_classes
    )
    t = time.time() - t
    logger.info("evaluation finishes with %.1f s.", t)
    return result


def parse_arguments() -> argparse.Namespace:
    """Parse the arguments."""
    parser = argparse.ArgumentParser(description="MOTS evaluation.")
    parser.add_argument(
        "--gt", "-g", required=True, help="path to mots ground truth"
    )
    parser.add_argument(
        "--result", "-r", required=True, help="path to mots results"
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
        "--out-file",
        default="",
        help="Output path for mots evaluation results.",
    )
    parser.add_argument(
        "--iou-thr",
        type=float,
        default=0.5,
        help="iou threshold for mots evaluation",
    )
    parser.add_argument(
        "--ignore-iof-thr",
        type=float,
        default=0.5,
        help="ignore iof threshold for mots evaluation",
    )
    parser.add_argument(
        "--ignore-unknown-cats",
        action="store_true",
        help="ignore unknown categories for mots evaluation",
    )
    parser.add_argument(
        "--nproc",
        "-p",
        type=int,
        default=NPROC,
        help="number of processes for mots evaluation",
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
    eval_result = evaluate_seg_track(
        acc_single_video_mots,
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
