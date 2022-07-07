from itertools import chain

import numpy as np


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

    report_dict = {label[0]: label[1:] for label in rows}
    for label, scores in report_dict.items():
        report_dict[label] = dict(zip(headers, [i.item() for i in scores]))

    return report_dict
