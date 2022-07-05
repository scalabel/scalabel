import math
import numbers
from collections import Counter
from contextlib import suppress
from itertools import chain
from typing import NamedTuple

import numpy as np


def _column_or_1d(y):
    """Ravel column or 1d numpy array, else raises an error."""
    y = np.asarray(y)
    shape = np.shape(y)
    if len(shape) == 1:
        return np.ravel(y)
    if len(shape) == 2 and shape[1] == 1:
        return np.ravel(y)

    raise ValueError("y should be a 1d array, got an array of shape {} instead.".format(shape))


def _num_samples(x):
    """Return number of samples in array-like x."""
    message = "Expected sequence or array-like, got %s" % type(x)
    if hasattr(x, "fit") and callable(x.fit):
        # Don't get num_samples from an ensembles length!
        raise TypeError(message)

    if not hasattr(x, "__len__") and not hasattr(x, "shape"):
        if hasattr(x, "__array__"):
            x = np.asarray(x)
        else:
            raise TypeError(message)

    if hasattr(x, "shape") and x.shape is not None:
        if len(x.shape) == 0:
            raise TypeError("Singleton array %r cannot be considered a valid collection." % x)
        # Check that shape is returning an integer or default to len
        # Dask dataframes may not return numeric shape[0] value
        if isinstance(x.shape[0], numbers.Integral):
            return x.shape[0]

    try:
        return len(x)
    except TypeError as type_error:
        raise TypeError(message) from type_error


class _MissingValues(NamedTuple):
    """Data class for missing data information"""

    nan: bool
    none: bool

    def to_list(self):
        """Convert tuple to a list where None is always first."""
        output = []
        if self.none:
            output.append(None)
        if self.nan:
            output.append(np.nan)
        return output


def _extract_missing(values):
    """Extract missing values from `values`.
    Parameters
    ----------
    values: set
        Set of values to extract missing from.
    Returns
    -------
    output: set
        Set with missing values extracted.
    missing_values: _MissingValues
        Object with missing value information.
    """
    missing_values_set = {value for value in values if value is None or _is_scalar_nan(value)}

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


def _is_scalar_nan(x):
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


def _encode(values, uniques):
    """Helper function to encode values into [0, n_uniques - 1]."""
    if values.dtype.kind in "OUS":
        try:
            table = _nandict({val: i for i, val in enumerate(uniques)})
            return np.array([table[v] for v in values])
        except KeyError as e:
            raise ValueError(f"y contains previously unseen labels: {str(e)}")
    else:
        return np.searchsorted(uniques, values)


def _unique_np(values, return_inverse=False, return_counts=False):
    """Helper function to find unique values for numpy arrays that correctly
    accounts for nans. See `_unique` documentation for details."""
    uniques = np.unique(values, return_inverse=return_inverse, return_counts=return_counts)

    inverse, counts = None, None

    if return_counts:
        *uniques, counts = uniques

    if return_inverse:
        *uniques, inverse = uniques

    if return_counts or return_inverse:
        uniques = uniques[0]

    # np.unique will have duplicate missing values at the end of `uniques`
    # here we clip the nans and remove it from uniques
    if uniques.size and isinstance(uniques[-1], numbers.Real) and math.isnan(uniques[-1]):
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


def _get_counts(values, uniques):
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

    unique_valid_indices = np.searchsorted(unique_values, uniques[uniques_in_values])
    output = np.zeros_like(uniques, dtype=np.int64)
    output[uniques_in_values] = counts[unique_valid_indices]
    return output


def _unique_python(values, *, return_inverse, return_counts):
    # Only used in `_uniques`, see docstring there for details
    try:
        uniques_set = set(values)
        uniques_set, missing_values = _extract_missing(uniques_set)

        uniques = sorted(uniques_set)
        uniques.extend(missing_values.to_list())
        uniques = np.array(uniques, dtype=values.dtype)
    except TypeError:
        types = sorted(t.__qualname__ for t in set(type(v) for v in values))
        raise TypeError("Encoders require their input to be uniformly " f"strings or numbers. Got {types}")
    ret = (uniques,)

    if return_inverse:
        table = _nandict({val: i for i, val in enumerate(uniques)})
        ret += (np.array([table[v] for v in values]),)

    if return_counts:
        ret += (_get_counts(values, uniques),)

    return ret[0] if len(ret) == 1 else ret


def _unique(values, *, return_inverse=False, return_counts=False):
    """Helper function to find unique values with support for python objects."""
    if values.dtype == object:
        return _unique_python(values, return_inverse=return_inverse, return_counts=return_counts)
    # numerical
    return _unique_np(values, return_inverse=return_inverse, return_counts=return_counts)


class _LabelEncoder:
    """Encode target labels with value between 0 and n_classes-1."""

    def __init__(self):
        self.classes_ = []

    def fit(self, y):
        """Fit label encoder."""
        y = _column_or_1d(y)
        self.classes_ = _unique(y)
        return self

    def transform(self, y):
        """Transform labels to normalized encoding."""
        y = _column_or_1d(y)
        # transform of empty array is empty array
        if _num_samples(y) == 0:
            return np.array([])

        return _encode(y, self.classes_)


def _unique_labels(y_true, y_pred):
    def __unique_labels(y):
        if hasattr(y, "__array__"):
            return np.unique(np.asarray(y))
        else:
            return set(y)

    ys_labels = set(chain.from_iterable(__unique_labels(y) for y in (y_true, y_pred)))

    # check that string types and number types are not mixed
    if len(set(isinstance(label, str) for label in ys_labels)) > 1:
        raise ValueError("Mix of label input types (string and number)")

    return np.array(sorted(ys_labels))


def _count_nonzero(X, axis=None, sample_weight=None):
    """A variant of X.getnnz() with extension to weighting on axis 0
    Useful in efficiently calculating multilabel metrics.
    """
    if axis == -1:
        axis = 1
    elif axis == -2:
        axis = 0
    elif X.format != "csr":
        raise TypeError("Expected CSR sparse format, got {0}".format(X.format))

    # We rely here on the fact that np.diff(Y.indptr) for a CSR
    # will return the number of nonzero entries in each row.
    # A bincount over Y.indices will return the number of nonzeros
    # in each column. See ``csr_matrix.getnnz`` in scipy >= 0.14.
    if axis is None:
        return np.dot(np.diff(X.indptr), sample_weight)
    elif axis == 1:
        out = np.diff(X.indptr)
        return out * sample_weight
    elif axis == 0:
        weights = np.repeat(sample_weight, np.diff(X.indptr))
        return np.bincount(X.indices, minlength=X.shape[1], weights=weights)
    else:
        raise ValueError("Unsupported axis: {0}".format(axis))


def _multilabel_confusion_matrix(y_true, y_pred, labels):
    """Compute a confusion matrix for each class or sample."""
    present_labels = _unique_labels(y_true, y_pred)
    n_labels = len(labels)
    labels = np.hstack([labels, np.setdiff1d(present_labels, labels, assume_unique=True)])

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
            tp_sum = np.bincount(tp_bins, weights=tp_bins_weights, minlength=len(labels))

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

        # all labels are index integers for multilabel
        if not np.array_equal(labels, present_labels):
            if np.max(labels) > np.max(present_labels):
                raise ValueError(
                    "All labels must be in [0, n labels) for "
                    "multilabel targets. "
                    "Got %d > %d" % (np.max(labels), np.max(present_labels))
                )
            if np.min(labels) < 0:
                raise ValueError(
                    "All labels must be in [0, n labels) for " "multilabel targets. " "Got %d < 0" % np.min(labels)
                )

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
    y_true,
    y_pred,
    labels,
    beta=1.0,
    average=None,
):
    """Compute precision, recall, F-measure and support for each class."""
    mcm = _multilabel_confusion_matrix(
        y_true,
        y_pred,
        labels,
    )
    tp_sum = mcm[:, 1, 1]
    pred_sum = tp_sum + mcm[:, 0, 1]
    true_sum = tp_sum + mcm[:, 1, 0]

    if average == "micro":
        tp_sum = np.array([tp_sum.sum()])
        pred_sum = np.array([pred_sum.sum()])
        true_sum = np.array([true_sum.sum()])

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

    # average the results
    if average == "weighted":
        weights = true_sum
        if weights.sum() == 0:
            zero_division_value = np.float64(1.0)
            # precision is zero_division if there are no positive predictions
            # recall is zero_division if there are no positive labels
            # fscore is zero_division if all labels AND predictions are
            # negative
            if pred_sum.sum() == 0:
                return (
                    zero_division_value,
                    zero_division_value,
                    zero_division_value,
                    None,
                )
            else:
                return (
                    np.float64(0.0),
                    zero_division_value,
                    np.float64(0.0),
                    None,
                )
    else:
        weights = None

    if average is not None:
        assert average != "binary" or len(precision) == 1
        precision = np.average(precision, weights=weights)
        recall = np.average(recall, weights=weights)
        f_score = np.average(f_score, weights=weights)
        true_sum = None  # return no support

    return precision, recall, f_score, true_sum


def classification_report(
    y_true,
    y_pred,
    target_names,
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
    p, r, f1, s = _precision_recall_fscore_support(
        y_true,
        y_pred,
        labels,
    )
    rows = zip(target_names, p, r, f1, s)
    average_options = ("micro", "macro", "weighted")

    report_dict = {label[0]: label[1:] for label in rows}
    for label, scores in report_dict.items():
        report_dict[label] = dict(zip(headers, [i.item() for i in scores]))

    # compute all applicable averages
    for average in average_options:
        if average.startswith("micro"):
            line_heading = "accuracy"
        else:
            line_heading = average + " avg"

        # compute averages with specified averaging method
        avg_p, avg_r, avg_f1, _ = _precision_recall_fscore_support(
            y_true,
            y_pred,
            labels,
            average=average,
        )
        avg = [avg_p, avg_r, avg_f1, np.sum(s)]
        report_dict[line_heading] = dict(zip(headers, [i.item() for i in avg]))

    if "accuracy" in report_dict.keys():
        report_dict["accuracy"] = report_dict["accuracy"]["precision"]
    return report_dict
