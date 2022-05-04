"""Multi-object Tracking evaluation code."""
import argparse
import json
import time
from functools import partial
from multiprocessing import Pool
from typing import Callable, Dict, List, Tuple, Union

import motmetrics as mm
import numpy as np

from ..common.io import open_write_text
from ..common.logger import logger
from ..common.parallel import NPROC
from ..common.typing import NDArrayF64, NDArrayI32, NDArrayU8
from ..label.io import group_and_sort, load, load_label_config
from ..label.transforms import box2d_to_bbox
from ..label.typing import Category, Config, Frame, Label
from ..label.utils import (
    check_crowd,
    check_ignored,
    get_leaf_categories,
    get_parent_categories,
)
from .result import AVERAGE, OVERALL, Result, Scores, ScoresList
from .utils import handle_inconsistent_length, label_ids_to_int

# Video = TypeVar("Video", List[Frame])
VidFunc = Callable[
    [List[Frame], List[Frame], List[str], float, float, bool],
    List[mm.MOTAccumulator],
]

METRIC_MAPS = {
    "idf1": "IDF1",
    "mota": "MOTA",
    "motp": "MOTP",
    "num_false_positives": "FP",
    "num_misses": "FN",
    "num_switches": "IDSw",
    "mostly_tracked": "MT",
    "partially_tracked": "PT",
    "mostly_lost": "ML",
    "num_fragmentations": "FM",
}
METRIC_TO_AVERAGE = set(["IDF1", "MOTA", "MOTP"])


class TrackResult(Result):
    """The class for bounding box tracking evaluation results."""

    mMOTA: float
    mMOTP: float
    mIDF1: float
    MOTA: List[Dict[str, float]]
    MOTP: List[Dict[str, float]]
    IDF1: List[Dict[str, float]]
    FP: List[Dict[str, int]]
    FN: List[Dict[str, int]]
    IDSw: List[Dict[str, int]]
    MT: List[Dict[str, int]]
    PT: List[Dict[str, int]]
    ML: List[Dict[str, int]]
    FM: List[Dict[str, int]]

    # pylint: disable=useless-super-delegation
    def __eq__(self, other: "Result") -> bool:  # type: ignore
        """Check whether two instances are equal."""
        return super().__eq__(other)


def parse_objects(
    objects: List[Label], classes: List[str], ignore_unknown_cats: bool = False
) -> Tuple[NDArrayF64, NDArrayI32, NDArrayI32, NDArrayF64]:
    """Parse objects under Scalabel formats."""
    bboxes, labels, ids, ignore_bboxes = [], [], [], []
    for obj in objects:
        box_2d = obj.box2d
        if box_2d is None:
            continue
        bbox = box2d_to_bbox(box_2d)
        category = obj.category
        if category in classes:
            if check_crowd(obj) or check_ignored(obj):
                ignore_bboxes.append(bbox)
            else:
                bboxes.append(bbox)
                labels.append(classes.index(category))
                ids.append(obj.id)
        else:
            if not ignore_unknown_cats:
                raise KeyError(f"Unknown category: {category}")
    bboxes_arr: NDArrayF64 = np.array(bboxes, dtype=np.float64)
    labels_arr: NDArrayI32 = np.array(labels, dtype=np.int32)
    ids_arr: NDArrayI32 = np.array(ids, dtype=np.int32)
    ignore_bboxes_arr: NDArrayF64 = np.array(ignore_bboxes, dtype=np.float64)
    return (bboxes_arr, labels_arr, ids_arr, ignore_bboxes_arr)


def intersection_over_area(preds: NDArrayF64, gts: NDArrayF64) -> NDArrayF64:
    """Returns the intersection over the area of the predicted box."""
    out: NDArrayF64 = np.zeros((len(preds), len(gts)), dtype=np.float64)
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            w = min(p[0] + p[2], g[0] + g[2]) - max(p[0], g[0])
            h = min(p[1] + p[3], g[1] + g[3]) - max(p[1], g[1])
            out[i][j] = max(w, 0) * max(h, 0) / float(p[2] * p[3])
    return out


def acc_single_video_mot(
    gts: List[Frame],
    results: List[Frame],
    classes: List[str],
    iou_thr: float = 0.5,
    ignore_iof_thr: float = 0.5,
    ignore_unknown_cats: bool = False,
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
        gt_bboxes, gt_labels, gt_ids, gt_ignores = parse_objects(
            gt.labels if gt.labels is not None else [],
            classes,
            ignore_unknown_cats,
        )
        pred_bboxes, pred_labels, pred_ids, _ = parse_objects(
            result.labels if result.labels is not None else [],
            classes,
            ignore_unknown_cats,
        )
        for i in range(num_classes):
            gt_inds, pred_inds = gt_labels == i, pred_labels == i
            gt_bboxes_c, gt_ids_c = gt_bboxes[gt_inds], gt_ids[gt_inds]
            pred_bboxes_c, pred_ids_c = (
                pred_bboxes[pred_inds],
                pred_ids[pred_inds],
            )
            if gt_bboxes_c.shape[0] == 0 and pred_bboxes_c.shape[0] != 0:
                distances = np.full((0, pred_bboxes_c.shape[0]), np.nan)
            elif gt_bboxes_c.shape[0] != 0 and pred_bboxes_c.shape[0] == 0:
                distances = np.full((gt_bboxes_c.shape[0], 0), np.nan)
            else:
                distances = mm.distances.iou_matrix(
                    gt_bboxes_c, pred_bboxes_c, max_iou=1 - iou_thr
                )
            if gt_ignores.shape[0] > 0:
                # 1. assign gt and preds
                fps: NDArrayU8 = np.ones(pred_bboxes_c.shape[0]).astype(bool)
                le, ri = mm.lap.linear_sum_assignment(distances)
                for m, n in zip(le, ri):
                    if np.isfinite(distances[m, n]):
                        fps[n] = False
                # 2. ignore by iof
                iofs = intersection_over_area(pred_bboxes_c, gt_ignores)
                ignores: bool = np.greater(iofs, ignore_iof_thr).any(axis=1)
                # 3. filter preds
                valid_inds = np.logical_not(np.logical_and(fps, ignores))
                pred_ids_c = pred_ids_c[valid_inds]
                distances = distances[:, valid_inds]
            if distances.shape != (0, 0):
                accs[i].update(gt_ids_c, pred_ids_c, distances)
    return accs


def aggregate_accs(
    video_accs: List[List[mm.MOTAccumulator]],
    classes: List[Category],
    super_classes: Dict[str, List[Category]],
) -> Tuple[List[str], List[List[str]], List[List[mm.MOTAccumulator]]]:
    """Aggregate the results of the entire dataset."""
    # accs for each class
    class_names: List[str] = [c.name for c in classes]
    metric_names: List[List[str]] = [[] for _ in classes]
    class_accs: List[List[mm.MOTAccumulator]] = [[] for _ in classes]
    for video_ind, _accs in enumerate(video_accs):
        for cls_ind, acc in enumerate(_accs):
            if (
                len(acc._events["Type"])  # pylint: disable=protected-access
                == 0
            ):
                continue
            name = f"{classes[cls_ind].name}_{video_ind}"
            class_accs[cls_ind].append(acc)
            metric_names[cls_ind].append(name)

    # super categories (if any)
    for super_cls, cls in super_classes.items():
        class_names.append(super_cls)
        metric_names.append(
            [n for c in cls for n in metric_names[classes.index(c)]]
        )
        class_accs.append(
            [a for c in cls for a in class_accs[classes.index(c)]]
        )

    # overall
    class_names.append(OVERALL)
    metric_names.append(
        [n for name in metric_names[: len(classes)] for n in name]
    )
    class_accs.append([a for acc in class_accs[: len(classes)] for a in acc])

    return class_names, metric_names, class_accs


def evaluate_single_class(
    names: List[str], accs: List[mm.MOTAccumulator]
) -> Dict[str, Union[int, float]]:
    """Evaluate results for one class.

    Args:
        names (list[str]): list of metric names
        accs (list[pymotmetrics.MOTAAccumulator]): list of accumulators
    Return:
        flat_dict (dict[str, int | float]):
            the dict that maps metric to score for the given class
    """
    mh = mm.metrics.create()
    summary = mh.compute_many(
        accs, names=names, metrics=METRIC_MAPS.keys(), generate_overall=True
    )
    flat_dict: Dict[str, Union[int, float]] = {
        METRIC_MAPS[k]: v["OVERALL"] for k, v in summary.to_dict().items()
    }
    if np.isnan(flat_dict["MOTP"]):
        num_dets = mh.compute_many(
            accs,
            names=names,
            metrics=["num_detections"],
            generate_overall=True,
        )
        sum_motp = (summary["motp"] * num_dets["num_detections"]).sum()
        motp = mm.math_util.quiet_divide(
            sum_motp, num_dets["num_detections"]["OVERALL"]
        )
        flat_dict["MOTP"] = float(1 - motp)
    else:
        flat_dict["MOTP"] = 1 - flat_dict["MOTP"]

    for metric, score in flat_dict.items():
        if isinstance(score, float):
            flat_dict[metric] = 100 * score
    return flat_dict


def compute_average(
    flat_dicts: List[Dict[str, Union[int, float]]],
    metrics: List[str],
    classes: List[Category],
) -> Dict[str, Union[int, float]]:
    """Calculate the AVERAGE scores."""
    ave_dict: Dict[str, Union[int, float]] = {}
    for metric in metrics:
        dtype = type(flat_dicts[-1][metric])
        v: NDArrayF64 = np.array(
            [flat_dicts[i][metric] for i in range(len(classes))],
            dtype=np.float64,
        )
        v = np.nan_to_num(v, nan=0, posinf=0, neginf=0)
        if dtype == int:
            value = int(v.sum())  # type: Union[int, float]
        elif dtype == float:
            value = float(v.mean())
        else:
            raise TypeError()
        ave_dict[metric] = value
    return ave_dict


def generate_results(
    flat_dicts: List[Dict[str, Union[int, float]]],
    class_names: List[str],
    metrics: List[str],
    classes: List[Category],
    super_classes: Dict[str, List[Category]],
) -> TrackResult:
    """Compute summary metrics for evaluation results."""
    ave_dict = compute_average(flat_dicts, metrics, classes)
    class_names.insert(len(flat_dicts) - 1, AVERAGE)
    flat_dicts.insert(len(flat_dicts) - 1, ave_dict)

    nested_dict: Dict[str, Scores] = {
        metric: {
            class_name: res_dict_[metric]
            for class_name, res_dict_ in zip(class_names, flat_dicts)
        }
        for metric in metrics
    }

    basic_set = [c.name for c in classes]
    super_set = list(super_classes.keys())
    hyper_set = [AVERAGE, OVERALL]
    class_name_sets = [basic_set, hyper_set]
    if [name for name in class_names if name in super_classes]:
        class_name_sets.insert(1, super_set)

    res_dict: Dict[str, Union[int, float, ScoresList]] = {
        metric: [
            {
                class_name: score
                for class_name, score in scores.items()
                if class_name in class_name_set
            }
            for class_name_set in class_name_sets
        ]
        for metric, scores in nested_dict.items()
    }
    res_dict.update(
        {"m" + metric: ave_dict[metric] for metric in METRIC_TO_AVERAGE}
    )
    return TrackResult(**res_dict)


def evaluate_track(
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
        iou_thr: Minimum IoU for a bounding box to be considered a positive.
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

    classes = get_leaf_categories(config.categories)
    super_classes = get_parent_categories(config.categories)

    logger.info("evaluating...")
    class_names = [c.name for c in classes]
    if nproc > 1:
        with Pool(nproc) as pool:
            video_accs = pool.starmap(
                partial(
                    acc_single_video,
                    classes=class_names,
                    ignore_iof_thr=ignore_iof_thr,
                    ignore_unknown_cats=ignore_unknown_cats,
                ),
                zip(gts, results),
            )
    else:
        video_accs = [
            acc_single_video(
                gt,
                result,
                class_names,
                iou_thr,
                ignore_iof_thr,
                ignore_unknown_cats,
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
    parser = argparse.ArgumentParser(description="MOT evaluation.")
    parser.add_argument(
        "--gt", "-g", required=True, help="path to mot ground truth"
    )
    parser.add_argument(
        "--result", "-r", required=True, help="path to mot results"
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
        help="Output path for mot evaluation results.",
    )
    parser.add_argument(
        "--iou-thr",
        type=float,
        default=0.5,
        help="iou threshold for mot evaluation",
    )
    parser.add_argument(
        "--ignore-iof-thr",
        type=float,
        default=0.5,
        help="ignore iof threshold for mot evaluation",
    )
    parser.add_argument(
        "--ignore-unknown-cats",
        type=bool,
        default=False,
        help="ignore unknown categories for mot evaluation",
    )
    parser.add_argument(
        "--nproc",
        "-p",
        type=int,
        default=NPROC,
        help="number of processes for mot evaluation",
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
    eval_result = evaluate_track(
        acc_single_video_mot,
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
