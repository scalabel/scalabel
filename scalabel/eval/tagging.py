"""Evaluation procedures for image tagging."""
import argparse
import json
from itertools import chain
from typing import (
    AbstractSet,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

import numpy as np

from ..common.io import open_write_text
from ..common.logger import logger
from ..common.parallel import NPROC
from ..common.typing import NDArrayF64, NDArrayI32
from ..label.io import load, load_label_config
from ..label.typing import Config, Frame
from ..label.utils import get_parent_categories
from .result import AVERAGE, Result, Scores, ScoresList
from .utils import reorder_preds


def column_or_1d(arr: NDArrayI32) -> NDArrayI32:
    """Ravel column or 1d numpy array, else raises an error."""
    np_arr = np.asarray(arr)
    shape = np.shape(np_arr)
    if len(shape) == 1:
        return np.ravel(arr)
    if len(shape) == 2 and shape[1] == 1:
        return np.ravel(arr)

    raise ValueError(
        f"input should be a 1d array, got an array of shape {shape} instead."
    )


def unique_labels(y_true: NDArrayI32, y_pred: NDArrayI32) -> NDArrayI32:
    """Chain and remove duplicates from the input."""
    ys_labels: Iterable[np.int32] = set(
        chain.from_iterable(set(np.unique(y)) for y in (y_true, y_pred))
    )
    return np.array(sorted(ys_labels))


def confusion_matrix(
    y_true: NDArrayI32, y_pred: NDArrayI32, labels: NDArrayI32
) -> NDArrayI32:
    """Compute a confusion matrix for each class or sample."""
    present_labels = unique_labels(y_true, y_pred)
    n_labels = labels.size
    labels = np.hstack(
        [labels, np.setdiff1d(present_labels, labels, assume_unique=True)]
    )

    sorted_labels = column_or_1d(labels)

    y_true = np.searchsorted(sorted_labels, y_true).astype(np.int32)
    y_pred = np.searchsorted(sorted_labels, y_pred).astype(np.int32)

    # labels are now from 0 to len(labels) - 1 -> use bincount
    tp = y_true == y_pred
    tp_bins = y_true[tp]

    true_sum = np.zeros(len(labels))
    pred_sum = tp_sum = true_sum
    if len(tp_bins):
        tp_sum = np.bincount(tp_bins, minlength=len(labels))  # type: ignore

    if len(y_pred):
        pred_sum = np.bincount(y_pred, minlength=len(labels))  # type: ignore
    if len(y_true):
        true_sum = np.bincount(y_true, minlength=len(labels))  # type: ignore

    # retain only selected labels
    indices = np.searchsorted(sorted_labels, labels[:n_labels])
    tp_sum = tp_sum[indices]
    true_sum = true_sum[indices]
    pred_sum = pred_sum[indices]

    fp = pred_sum - tp_sum
    fn = true_sum - tp_sum
    tp = tp_sum

    tn = y_true.shape[0] - tp - fp - fn
    # pylint: disable-next=too-many-function-args
    return np.array([tn, fp, fn, tp]).T.reshape(-1, 2, 2)


def _prf_divide(numerator: NDArrayI32, denominator: NDArrayI32) -> NDArrayF64:
    """Performs division and handles divide-by-zero."""
    mask = denominator == 0.0
    denominator = denominator.copy()
    denominator[mask] = 1  # avoid infs/nans
    result = numerator / denominator

    if not np.any(mask):
        return result

    # address zero division by setting 0s to 1s
    result[mask] = 0
    return result


def precision_recall_fscore(
    y_true: NDArrayI32,
    y_pred: NDArrayI32,
    labels: NDArrayI32,
    average: Optional[str] = None,
    beta: float = 1.0,
) -> Tuple[NDArrayF64, NDArrayF64, NDArrayF64, Optional[NDArrayI32]]:
    """Compute precision, recall, F-measure and support for each class."""
    mcm = confusion_matrix(y_true, y_pred, labels)
    tp_sum: NDArrayI32 = mcm[:, 1, 1]
    pred_sum = tp_sum + mcm[:, 0, 1]
    true_sum: Optional[NDArrayI32] = tp_sum + mcm[:, 1, 0]
    assert true_sum is not None

    if average == "micro":
        tp_sum = np.array([tp_sum.sum()])
        pred_sum = np.array([pred_sum.sum()])
        true_sum = np.array([true_sum.sum()])

    beta2 = beta**2

    # divide and set scores
    precision = _prf_divide(tp_sum, pred_sum)
    recall = _prf_divide(tp_sum, true_sum)

    # if tp == 0 F will be 1 only if all predictions are zero, all labels are
    # zero, and zero_division=1. In all other case, 0
    if np.isposinf(beta):
        f_score = recall
    else:
        denom = beta2 * precision + recall

        denom[denom == 0.0] = 1  # avoid division by 0
        f_score = (1 + beta2) * precision * recall / denom

    weights = None
    if average is not None:
        assert average != "binary" or len(precision) == 1
        precision = np.average(precision, weights=weights)  # type: ignore
        recall = np.average(recall, weights=weights)  # type: ignore
        f_score = np.average(f_score, weights=weights)  # type: ignore
        true_sum = None  # return no support

    return precision, recall, f_score, true_sum


def compute_scores(
    y_true: NDArrayI32,
    y_pred: NDArrayI32,
    target_names: List[str],
) -> Dict[str, Union[float, Dict[str, float]]]:
    """Build a text report showing the main classification metrics.

    Adapted from:
    https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/metrics/_classification.py
    """
    labels = unique_labels(y_true, y_pred)

    if target_names and labels.size != len(target_names):
        raise ValueError(
            f"Number of classes, {labels.size}, does not match size of "
            f"target_names, {len(target_names)}. Try specifying the labels "
            "parameter"
        )

    headers = ["precision", "recall", "f1-score", "support"]
    # compute per-class results without averaging
    p, r, f1, s = precision_recall_fscore(y_true, y_pred, labels)
    assert s is not None
    rows = zip(target_names, p, r, f1, s)
    average_options = ("micro", "macro")

    report_dict: Dict[str, Union[float, Dict[str, float]]] = {}
    scores_dict = {label[0]: label[1:] for label in rows}
    for label, scores in scores_dict.items():
        report_dict[label] = dict(
            zip(
                headers,
                [i.item() for i in scores],
            )
        )

    # compute all applicable averages
    for average in average_options:
        if average.startswith("micro"):
            line_heading = "accuracy"
        else:
            line_heading = average + " avg"

        # compute averages with specified averaging method
        avg_p, avg_r, avg_f1, _ = precision_recall_fscore(
            y_true,
            y_pred,
            labels=labels,
            average=average,
        )
        assert s is not None
        avg = [avg_p, avg_r, avg_f1, np.sum(s, dtype=np.float64)]

        report_dict[line_heading] = dict(
            zip(
                headers,
                [i.item() for i in avg],
            )
        )

    if "accuracy" in report_dict:
        report_dict["accuracy"] = cast(
            Dict[str, float], report_dict["accuracy"]
        )["precision"]
    return report_dict


class TaggingResult(Result):
    """The class for general image tagging evaluation results."""

    precision: List[Dict[str, float]]
    recall: List[Dict[str, float]]
    f1_score: List[Dict[str, float]]
    accuracy: List[Dict[str, float]]

    def __eq__(self, other: "TaggingResult") -> bool:  # type: ignore
        """Check whether two instances are equal."""
        return super().__eq__(other)

    def summary(
        self,
        include: Optional[AbstractSet[str]] = None,
        exclude: Optional[AbstractSet[str]] = None,
    ) -> Scores:
        """Convert tagging results into a flattened dict as the summary."""
        summary_dict: Dict[str, Union[int, float]] = {}
        for metric, scores_list in self.dict(
            include=include, exclude=exclude  # type: ignore
        ).items():
            for category, score in scores_list[-2].items():
                summary_dict[f"{metric}/{category}"] = score
            summary_dict[metric] = scores_list[-1][AVERAGE]
        return summary_dict


def evaluate_tagging(
    ann_frames: List[Frame],
    pred_frames: List[Frame],
    config: Config,
    nproc: int = NPROC,  # pylint: disable=unused-argument
) -> TaggingResult:
    """Evaluate image tagging with Scalabel format.

    Args:
        ann_frames: the ground truth frames.
        pred_frames: the prediction frames.
        config: Metadata config.
        nproc: the number of process.

    Returns:
        TaggingResult: evaluation results.
    """
    pred_frames = reorder_preds(ann_frames, pred_frames)
    tag_classes = get_parent_categories(config.categories)
    assert tag_classes, "Tag attributes must be specified as supercategories"
    metrics = ["precision", "recall", "f1_score", "accuracy"]
    outputs: Dict[str, ScoresList] = {m: [] for m in metrics}
    avgs: Dict[str, Scores] = {m: {} for m in metrics}
    for tag, class_list in tag_classes.items():
        classes = [c.name for c in class_list]
        preds_cls, gts_cls = [], []
        for p, g in zip(pred_frames, ann_frames):
            if g.attributes is None:
                continue
            assert p.attributes is not None
            p_attr, g_attr = p.attributes[tag], g.attributes[tag]
            assert isinstance(p_attr, str) and isinstance(g_attr, str)
            assert p_attr in classes and g_attr in classes
            preds_cls.append(classes.index(p_attr))
            gts_cls.append(classes.index(g_attr))
        parray: NDArrayI32 = np.array(preds_cls, dtype=np.int32)
        garray: NDArrayI32 = np.array(gts_cls, dtype=np.int32)
        gt_classes = [classes[cid] for cid in sorted(set(gts_cls + preds_cls))]
        scores = compute_scores(garray, parray, gt_classes)
        out: Dict[str, Scores] = {}
        for metric in ["precision", "recall", "f1-score"]:
            met = metric if metric != "f1-score" else "f1_score"
            out[met] = {}
            for cat in classes:
                m: float = cast(Dict[str, float], scores.get(cat, {})).get(
                    metric, 0.0
                )
                out[met][f"{tag}.{cat}"] = (
                    m * 100.0 if cat in scores else np.nan
                )
            avgs[met][tag.upper()] = (
                cast(Dict[str, float], scores["macro avg"])[metric] * 100.0
                if len(scores) > 3
                else np.nan
            )
        out["accuracy"] = {f"{tag}.{cat}": np.nan for cat in classes}
        avgs["accuracy"][tag.upper()] = cast(float, scores["accuracy"]) * 100.0
        for m, v in out.items():  # type: ignore
            assert isinstance(m, str)
            outputs[m].append(v)
    for m, v in avgs.items():  # type: ignore
        assert isinstance(m, str)
        outputs[m].append(v)
        outputs[m].append({AVERAGE: np.nanmean(list(v.values()))})
    return TaggingResult(**outputs)


def parse_arguments() -> argparse.Namespace:
    """Parse the arguments."""
    parser = argparse.ArgumentParser(description="Tagging evaluation.")
    parser.add_argument(
        "--gt", "-g", required=True, help="path to tagging ground truth"
    )
    parser.add_argument(
        "--result", "-r", required=True, help="path to tagging results"
    )
    parser.add_argument(
        "--config",
        "-c",
        default=None,
        help="Path to config toml file. Contains definition of categories, "
        "and optionally attributes and resolution. For an example "
        "see scalabel/label/testcases/configs.toml",
    )
    parser.add_argument(
        "--out-file",
        default="",
        help="Output file for tagging evaluation results.",
    )
    parser.add_argument(
        "--nproc",
        "-p",
        type=int,
        default=NPROC,
        help="number of processes for tagging evaluation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    dataset = load(args.gt, args.nproc)
    gts, cfg = dataset.frames, dataset.config
    preds = load(args.result).frames
    if args.config is not None:
        cfg = load_label_config(args.config)
    if cfg is None:
        raise ValueError(
            "Dataset config is not specified. Please use --config"
            " to specify a config for this dataset."
        )
    eval_result = evaluate_tagging(gts, preds, cfg, args.nproc)
    logger.info(eval_result)
    logger.info(eval_result.summary())
    if args.out_file:
        with open_write_text(args.out_file) as file:
            json.dump(eval_result.json(), file)
