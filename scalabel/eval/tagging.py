"""Evaluation procedures for image tagging."""
import argparse
import json
import math
import numbers
from contextlib import suppress
from itertools import chain
from typing import (
    AbstractSet,
    Any,
    Counter,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np

from ..common.io import open_write_text
from ..common.logger import logger
from ..common.parallel import NPROC
from ..common.typing import NDArrayI32
from ..label.io import load, load_label_config
from ..label.typing import Config, Frame
from ..label.utils import get_parent_categories
from .result import AVERAGE, Result, Scores, ScoresList
from .utils import reorder_preds


def _column_or_1d(y: NDArrayI32) -> NDArrayI32:
    """Ravel column or 1d numpy array, else raises an error."""
    y = np.asarray(y)
    shape = np.shape(y)
    if len(shape) == 1:
        return np.ravel(y)
    if len(shape) == 2 and shape[1] == 1:
        return np.ravel(y)

    raise ValueError(
        "y should be a 1d array, got an array of shape {} instead.".format(
            shape
        )
    )


def _num_samples(x: NDArrayI32) -> int:
    """Return number of samples in array-like x."""
    message = "Expected sequence or array-like, got %s" % type(x)
    if not hasattr(x, "__len__") and not hasattr(x, "shape"):
        if hasattr(x, "__array__"):
            x = np.asarray(x)
        else:
            raise TypeError(message)

    if hasattr(x, "shape") and x.shape is not None:
        if len(x.shape) == 0:
            raise TypeError(
                "Singleton array %r cannot be considered a valid collection."
                % x
            )
        # Check that shape is returning an integer or default to len
        # Dask dataframes may not return numeric shape[0] value
        if isinstance(x.shape[0], numbers.Integral):
            return x.shape[0]

    try:
        return len(x)
    except TypeError as type_error:
        raise TypeError(message) from type_error


class _MissingValues(NamedTuple):
    """Data class for missing data information."""

    nan: bool
    none: bool

    def to_list(self) -> List[Optional[float]]:
        """Convert tuple to a list where None is always first."""
        output: List[Optional[float]] = []
        if self.none:
            output.append(None)
        if self.nan:
            output.append(np.nan)
        return output


def _extract_missing(
    values: Set[np.int32],
) -> Tuple[Set[np.int32], _MissingValues]:
    """Extract missing values from `values`."""
    missing_values_set = {
        value for value in values if value is None or _is_scalar_nan(value)
    }

    if not missing_values_set:
        return values, _MissingValues(nan=False, none=False)

    if None in missing_values_set:
        if len(missing_values_set) == 1:
            output_missing_values = _MissingValues(nan=False, none=True)
        else:
            # If there is more than one missing value, then it has to be
            # float('nan') or np.nan
            output_missing_values = _MissingValues(nan=True, none=True)
    else:
        output_missing_values = _MissingValues(nan=True, none=False)

    # create set without the missing values
    output = values - missing_values_set
    return output, output_missing_values


def _is_scalar_nan(x: np.int32) -> bool:
    """Tests if x is NaN."""
    return isinstance(x, numbers.Real) and math.isnan(x)


class _nandict(dict):
    """Dictionary with support for nans."""

    def __init__(self, mapping):
        super().__init__(mapping)
        for key, value in mapping.items():
            if _is_scalar_nan(key):
                self.nan_value = value
                break

    def __missing__(self, key):
        if hasattr(self, "nan_value") and _is_scalar_nan(key):
            return self.nan_value
        raise KeyError(key)


class _NaNCounter(Counter):
    """Counter with support for nan values."""

    def __init__(self, items):
        super().__init__(self._generate_items(items))

    def _generate_items(self, items):
        """Generate items without nans. Stores the nan counts separately."""
        for item in items:
            if not _is_scalar_nan(item):
                yield item
                continue
            if not hasattr(self, "nan_count"):
                self.nan_count = 0
            self.nan_count += 1

    def __missing__(self, key):
        if hasattr(self, "nan_count") and _is_scalar_nan(key):
            return self.nan_count
        raise KeyError(key)


def _encode(values: NDArrayI32, uniques: Sequence[np.int32]) -> NDArrayI32:
    """Helper function to encode values into [0, n_uniques - 1]."""
    if values.dtype.kind in "OUS":
        try:
            table = _nandict({val: i for i, val in enumerate(uniques)})
            return np.array([table[v] for v in values])
        except KeyError as e:
            raise ValueError(f"y contains previously unseen labels: {str(e)}")
    else:
        return np.searchsorted(uniques, values)


def _unique_np(
    values: Sequence[np.int32],
    return_inverse: bool = False,
    return_counts: bool = False,
) -> Sequence[np.int32]:
    """Helper function to find unique values, accounts for nans."""
    uniques: NDArrayI32 = np.unique(
        values, return_inverse=return_inverse, return_counts=return_counts
    )

    inverse, counts = None, None

    if return_counts:
        *uniques, counts = uniques

    if return_inverse:
        *uniques, inverse = uniques

    if return_counts or return_inverse:
        uniques = uniques[0]

    # np.unique will have duplicate missing values at the end of `uniques`
    # here we clip the nans and remove it from uniques
    if (
        uniques.size
        and isinstance(uniques[-1], numbers.Real)
        and math.isnan(uniques[-1])
    ):
        nan_idx = np.searchsorted(uniques, np.nan)
        uniques = uniques[: nan_idx + 1]
        if return_inverse:
            inverse[inverse > nan_idx] = nan_idx

        if return_counts:
            counts[nan_idx] = np.sum(counts[nan_idx:])
            counts = counts[: nan_idx + 1]

    ret = (uniques,)

    if return_inverse:
        ret += (inverse,)

    if return_counts:
        ret += (counts,)

    return ret[0] if len(ret) == 1 else ret


def _get_counts(values: NDArrayI32, uniques: Sequence[np.int32]):
    """Get the count of each of the `uniques` in `values`.

    The counts will use the order passed in by `uniques`. For non-object dtypes,
    `uniques` is assumed to be sorted and `np.nan` is at the end.
    """
    if values.dtype.kind in "OU":
        counter = _NaNCounter(values)
        output = np.zeros(len(uniques), dtype=np.int64)
        for i, item in enumerate(uniques):
            with suppress(KeyError):
                output[i] = counter[item]
        return output

    unique_values, counts = _unique_np(values, return_counts=True)

    # Recorder unique_values based on input: `uniques`
    uniques_in_values = np.isin(uniques, unique_values, assume_unique=True)
    if np.isnan(unique_values[-1]) and np.isnan(uniques[-1]):
        uniques_in_values[-1] = True

    unique_valid_indices = np.searchsorted(
        unique_values, uniques[uniques_in_values]
    )
    output = np.zeros_like(uniques, dtype=np.int64)
    output[uniques_in_values] = counts[unique_valid_indices]
    return output


def _unique_python(values: NDArrayI32) -> Sequence[np.int32]:
    # Only used in `_uniques`, see docstring there for details
    try:
        uniques_set = set(values)
        uniques_set, missing_values = _extract_missing(uniques_set)

        uniques = sorted(uniques_set)
        uniques.extend(missing_values.to_list())
        uniques = np.array(uniques, dtype=values.dtype)
    except TypeError:
        types = sorted(t.__qualname__ for t in set(type(v) for v in values))
        raise TypeError(
            "Encoders require their input to be uniformly "
            f"strings or numbers. Got {types}"
        )
    ret = (uniques,)

    return ret[0] if len(ret) == 1 else ret


def _unique(values: NDArrayI32) -> Sequence[np.int32]:
    """Helper function to find unique values with support for python objects."""
    if values.dtype == object:
        return _unique_python(values)
    # numerical
    return _unique_np(values)


class _LabelEncoder:
    """Encode target labels with value between 0 and n_classes-1."""

    def __init__(self):
        self.classes_ = []

    def fit(self, y: NDArrayI32):
        """Fit label encoder."""
        y = _column_or_1d(y)
        self.classes_ = _unique(y)
        return self

    def transform(self, y: NDArrayI32):
        """Transform labels to normalized encoding."""
        y = _column_or_1d(y)
        # transform of empty array is empty array
        if _num_samples(y) == 0:
            return np.array([])

        return _encode(y, self.classes_)


def _unique_labels(y_true: NDArrayI32, y_pred: NDArrayI32) -> NDArrayI32:
    def __unique_labels(y):
        if hasattr(y, "__array__"):
            return np.unique(np.asarray(y))
        else:
            return set(y)

    ys_labels = set(
        chain.from_iterable(__unique_labels(y) for y in (y_true, y_pred))
    )
    return np.array(sorted(ys_labels))


def _count_nonzero(x, axis=None, sample_weight=None):
    """A variant of x.getnnz() with extension to weighting on axis 0.

    Useful in efficiently calculating multilabel metrics.
    """
    if axis == -1:
        axis = 1
    elif axis == -2:
        axis = 0
    elif x.format != "csr":
        raise TypeError("Expected CSR sparse format, got {0}".format(x.format))

    # We rely here on the fact that np.diff(Y.indptr) for a CSR
    # will return the number of nonzero entries in each row.
    # A bincount over Y.indices will return the number of nonzeros
    # in each column. See ``csr_matrix.getnnz`` in scipy >= 0.14.
    if axis is None:
        return np.dot(np.diff(x.indptr), sample_weight)
    elif axis == 1:
        out = np.diff(x.indptr)
        return out * sample_weight
    elif axis == 0:
        weights = np.repeat(sample_weight, np.diff(x.indptr))
        return np.bincount(x.indices, minlength=x.shape[1], weights=weights)
    else:
        raise ValueError("Unsupported axis: {0}".format(axis))


def _multilabel_confusion_matrix(
    y_true: NDArrayI32, y_pred: NDArrayI32, labels: NDArrayI32
) -> NDArrayI32:
    """Compute a confusion matrix for each class or sample."""
    present_labels = _unique_labels(y_true, y_pred)
    n_labels = len(labels)
    labels = np.hstack(
        [labels, np.setdiff1d(present_labels, labels, assume_unique=True)]
    )

    if y_true.ndim == 1:
        le = _LabelEncoder()
        le.fit(labels)
        y_true = le.transform(y_true)
        y_pred = le.transform(y_pred)
        sorted_labels = le.classes_

        # labels are now from 0 to len(labels) - 1 -> use bincount
        tp = y_true == y_pred
        tp_bins = y_true[tp]
        tp_bins_weights = None

        true_sum = pred_sum = tp_sum = np.zeros(len(labels))
        if len(tp_bins):
            tp_sum = np.bincount(
                tp_bins, weights=tp_bins_weights, minlength=len(labels)
            )

        if len(y_pred):
            pred_sum = np.bincount(y_pred, minlength=len(labels))
        if len(y_true):
            true_sum = np.bincount(y_true, minlength=len(labels))

        # retain only selected labels
        indices = np.searchsorted(sorted_labels, labels[:n_labels])
        tp_sum = tp_sum[indices]
        true_sum = true_sum[indices]
        pred_sum = pred_sum[indices]

    else:
        sum_axis = 0
        if n_labels is not None:
            y_true = y_true[:, labels[:n_labels]]
            y_pred = y_pred[:, labels[:n_labels]]

        # calculate weighted counts
        true_and_pred = y_true.multiply(y_pred)
        tp_sum = _count_nonzero(true_and_pred, axis=sum_axis)
        pred_sum = _count_nonzero(y_pred, axis=sum_axis)
        true_sum = _count_nonzero(y_true, axis=sum_axis)

    fp = pred_sum - tp_sum
    fn = true_sum - tp_sum
    tp = tp_sum

    tn = y_true.shape[0] - tp - fp - fn
    return np.array([tn, fp, fn, tp]).T.reshape(-1, 2, 2)


def _prf_divide(numerator, denominator):
    """Performs division and handles divide-by-zero."""
    mask = denominator == 0.0
    denominator = denominator.copy()
    denominator[mask] = 1  # avoid infs/nans
    result = numerator / denominator

    if not np.any(mask):
        return result

    # address zero division by setting 0s to 1s
    result[mask] = 1.0
    return result


def _precision_recall_fscore_support(
    y_true: NDArrayI32,
    y_pred: NDArrayI32,
    labels: NDArrayI32,
    beta: float = 1.0,
):
    """Compute precision, recall, F-measure and support for each class."""
    mcm = _multilabel_confusion_matrix(y_true, y_pred, labels)
    tp_sum = mcm[:, 1, 1]
    pred_sum = tp_sum + mcm[:, 0, 1]
    true_sum = tp_sum + mcm[:, 1, 0]

    beta2 = beta ** 2

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

    return precision, recall, f_score, true_sum


def compute_scores(
    y_true: NDArrayI32,
    y_pred: NDArrayI32,
    target_names: List[str],
):
    """Build a text report showing the main classification metrics."""
    labels = _unique_labels(y_true, y_pred)

    if target_names and len(labels) != len(target_names):
        raise ValueError(
            "Number of classes, {0}, does not match size of "
            "target_names, {1}. Try specifying the labels "
            "parameter".format(len(labels), len(target_names))
        )

    headers = ["precision", "recall", "f1-score", "support"]
    # compute per-class results without averaging
    p, r, f1, s = _precision_recall_fscore_support(y_true, y_pred, labels)
    rows = zip(target_names, p, r, f1, s)

    report_dict = {label[0]: label[1:] for label in rows}
    for label, scores in report_dict.items():
        report_dict[label] = dict(zip(headers, [i.item() for i in scores]))

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
        for metric, scores_list in self.dict(include=include, exclude=exclude).items():  # type: ignore
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
                out[met][f"{tag}.{cat}"] = (
                    scores[cat][metric] * 100.0 if cat in scores else np.nan
                )
            avgs[met][tag.upper()] = (
                scores["macro avg"][metric] * 100.0
                if len(scores) > 3
                else np.nan
            )
        out["accuracy"] = {f"{tag}.{cat}": np.nan for cat in classes}
        avgs["accuracy"][tag.upper()] = scores["accuracy"] * 100.0
        for m, v in out.items():
            outputs[m].append(v)
    for m, v in avgs.items():
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
        with open_write_text(args.out_file) as fp:
            json.dump(eval_result.json(), fp)
