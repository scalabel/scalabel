"""Multi-object Tracking evaluation code."""
import argparse
import json
import os
import time
from functools import partial
from multiprocessing import Pool
from typing import Callable, Dict, List, Tuple, TypeVar, Union

import motmetrics as mm
import numpy as np
import pandas as pd

from ..common.logger import logger
from ..label.io import (
    DEFAULT_LABEL_CONFIG,
    group_and_sort,
    load,
    load_label_config,
)
from ..label.transforms import box2d_to_bbox
from ..label.typing import Category, Config, Frame, Label
from ..label.utils import (
    check_crowd,
    check_ignored,
    get_leaf_categories,
    get_parent_categories,
)

EvalResults = Dict[str, Dict[str, float]]
Video = TypeVar("Video", List[Frame], List[str])
VidFunc = Callable[
    [Video, Video, List[str], float, float],
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


def parse_objects(
    objects: List[Label], classes: List[str]
) -> List[np.ndarray]:
    """Parse objects under Scalable formats."""
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
            raise KeyError(f"Unknown category: {category}")
    return list(map(np.array, [bboxes, labels, ids, ignore_bboxes]))


def intersection_over_area(preds: np.ndarray, gts: np.ndarray) -> np.ndarray:
    """Returns the intersection over the area of the predicted box."""
    out = np.zeros((len(preds), len(gts)))
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
) -> List[mm.MOTAccumulator]:
    """Accumulate results for one video."""
    assert len(gts) == len(results)
    get_frame_index = (
        lambda x: x.frame_index if x.frame_index is not None else 0
    )
    num_classes = len(classes)
    gts = sorted(gts, key=get_frame_index)
    results = sorted(results, key=get_frame_index)
    accs = [mm.MOTAccumulator(auto_id=True) for _ in range(num_classes)]
    for gt, result in zip(gts, results):
        assert gt.frame_index == result.frame_index
        gt_bboxes, gt_labels, gt_ids, gt_ignores = parse_objects(
            gt.labels if gt.labels is not None else [], classes
        )
        pred_bboxes, pred_labels, pred_ids, _ = parse_objects(
            result.labels if result.labels is not None else [], classes
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
                fps = np.ones(pred_bboxes_c.shape[0]).astype(np.bool8)
                le, ri = mm.lap.linear_sum_assignment(distances)
                for m, n in zip(le, ri):
                    if np.isfinite(distances[m, n]):
                        fps[n] = False
                # 2. ignore by iof
                iofs = intersection_over_area(pred_bboxes_c, gt_ignores)
                ignores = (iofs > ignore_iof_thr).any(axis=1)
                # 3. filter preds
                valid_inds = ~(fps & ignores)
                pred_ids_c = pred_ids_c[valid_inds]
                distances = distances[:, valid_inds]
            if distances.shape != (0, 0):
                accs[i].update(gt_ids_c, pred_ids_c, distances)
    return accs


def aggregate_accs(
    accumulators: List[List[mm.MOTAccumulator]],
    classes: List[Category],
    super_classes: Dict[str, List[Category]],
) -> Tuple[List[List[str]], List[List[mm.MOTAccumulator]], List[str]]:
    """Aggregate the results of the entire dataset."""
    # accs for each class
    items = [c.name for c in classes]

    names: List[List[str]] = [[] for _ in items]
    accs: List[List[str]] = [[] for _ in items]
    for video_ind, _accs in enumerate(accumulators):
        for cls_ind, acc in enumerate(_accs):
            if (
                len(acc._events["Type"])  # pylint: disable=protected-access
                == 0
            ):
                continue
            name = f"{classes[cls_ind].name}_{video_ind}"
            names[cls_ind].append(name)
            accs[cls_ind].append(acc)

    # super categories (if any)
    for super_cls, cls in super_classes.items():
        items.append(super_cls)
        names.append([n for c in cls for n in names[classes.index(c)]])
        accs.append([a for c in cls for a in accs[classes.index(c)]])

    # overall
    items.append("OVERALL")
    names.append([n for name in names[: len(classes)] for n in name])
    accs.append([a for acc in accs[: len(classes)] for a in acc])

    return names, accs, items


def evaluate_single_class(
    names: List[str], accs: List[mm.MOTAccumulator]
) -> List[float]:
    """Evaluate results for one class."""
    mh = mm.metrics.create()
    summary = mh.compute_many(
        accs, names=names, metrics=METRIC_MAPS.keys(), generate_overall=True
    )
    results = [v["OVERALL"] for k, v in summary.to_dict().items()]
    motp_ind = list(METRIC_MAPS).index("motp")
    if np.isnan(results[motp_ind]):
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
        results[motp_ind] = float(1 - motp)
    else:
        results[motp_ind] = 1 - results[motp_ind]
    return results


def render_results(
    summaries: List[List[float]],
    items: List[str],
    metrics: List[str],
    classes: List[Category],
    super_classes: Dict[str, List[Category]],
) -> EvalResults:
    """Render the evaluation results."""
    eval_results = pd.DataFrame(columns=metrics)
    # category, super-category and overall results
    for i, item in enumerate(items):
        eval_results.loc[item] = summaries[i]
    dtypes = {m: type(d) for m, d in zip(metrics, summaries[0])}
    # average results
    avg_results: List[Union[int, float]] = []
    for i, m in enumerate(metrics):
        v = np.array([s[i] for s in summaries[: len(classes)]])
        v = np.nan_to_num(v, nan=0)
        if dtypes[m] == int:
            avg_results.append(int(v.sum()))
        elif dtypes[m] == float:
            avg_results.append(float(v.mean()))
        else:
            raise TypeError()
    eval_results.loc["AVERAGE"] = avg_results
    eval_results = eval_results.astype(dtypes)

    metric_host = mm.metrics.create()
    metric_host.register(mm.metrics.motp, formatter="{:.1%}".format)
    strsummary = mm.io.render_summary(
        eval_results,
        formatters=metric_host.formatters,
        namemap=METRIC_MAPS,
    )
    strsummary = strsummary.split("\n")
    len_all = len(classes) + 3
    if super_classes is not None:
        len_all += len(super_classes)
    assert len(strsummary) == len_all
    split_line = "-" * len(strsummary[0])
    strsummary.insert(1, split_line)
    strsummary.insert(2 + len(classes), split_line)
    strsummary.insert(len_all, split_line)
    strsummary = "".join([f"{s}\n" for s in strsummary])
    strsummary = "\n" + strsummary
    logger.info(strsummary)

    outputs: EvalResults = dict()
    for i, item in enumerate(items):
        outputs[item] = dict()
        for j, metric in enumerate(METRIC_MAPS.values()):
            outputs[item][metric] = summaries[i][j]
    outputs["OVERALL"]["mIDF1"] = eval_results.loc["AVERAGE"]["idf1"]
    outputs["OVERALL"]["mMOTA"] = eval_results.loc["AVERAGE"]["mota"]
    outputs["OVERALL"]["mMOTP"] = eval_results.loc["AVERAGE"]["motp"]

    return outputs


def evaluate_track(
    acc_single_video: VidFunc[Video],
    gts: List[Video],
    results: List[Video],
    config: Config,
    iou_thr: float = 0.5,
    ignore_iof_thr: float = 0.5,
    nproc: int = 4,
) -> EvalResults:
    """Evaluate CLEAR MOT metrics for a Scalabel format dataset.

    Args:
        acc_single_video: Function for calculating metrics over a single video.
        gts: (paths to) the ground truth annotations in Scalabel format
        results: (paths to) the prediction results in Scalabel format.
        config: Config object
        iou_thr: Minimum IoU for a bounding box to be considered a positive.
        ignore_iof_thr: Min. Intersection over foreground with ignore regions.
        nproc: processes number for loading files

    Returns:
        dict: CLEAR MOT metric scores
    """
    logger.info("Tracking evaluation with CLEAR MOT metrics.")
    t = time.time()
    assert len(gts) == len(results)
    metrics = list(METRIC_MAPS.keys())

    classes = get_leaf_categories(config.categories)
    super_classes = get_parent_categories(config.categories)

    logger.info("accumulating...")
    with Pool(nproc) as pool:
        accs = pool.starmap(
            partial(
                acc_single_video,
                classes=[c.name for c in classes],
                iou_thr=iou_thr,
                ignore_iof_thr=ignore_iof_thr,
            ),
            zip(gts, results),
        )

    names, accs, items = aggregate_accs(accs, classes, super_classes)

    logger.info("evaluating...")
    with Pool(nproc) as pool:
        summaries = pool.starmap(evaluate_single_class, zip(names, accs))

    logger.info("rendering...")
    eval_results = render_results(
        summaries, items, metrics, classes, super_classes
    )
    t = time.time() - t
    logger.info("evaluation finishes with %.1f s.", t)
    return eval_results


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
        default=DEFAULT_LABEL_CONFIG,
        help="Path to config toml file. Contains definition of categories, "
        "and optionally attributes as well as resolution. For an example "
        "see scalabel/label/configs.toml",
    )
    parser.add_argument(
        "--out-dir",
        "-o",
        default="none",
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
        "--nproc",
        "-p",
        type=int,
        default=4,
        help="number of processes for mot evaluation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    dataset = load(args.gt)
    gt_frames, cfg = dataset.frames, dataset.config
    if args.config is not None:
        cfg = load_label_config(args.config)
    assert cfg is not None
    scores = evaluate_track(
        acc_single_video_mot,
        group_and_sort(gt_frames),
        group_and_sort(load(args.result).frames),
        cfg,
        args.iou_thr,
        args.ignore_iof_thr,
        args.nproc,
    )
    if args.out_dir != "none":
        output_filename = os.path.join(args.out_dir, "scores.json")
        with open(output_filename, "w") as fp:
            json.dump(scores, fp)
