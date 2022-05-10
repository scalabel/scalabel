"""Evaluation procedures for semantic segmentation.

For dataset with `n` classes, we treat the index `n` as the ignored class.
When computing IoUs, this ignored class is considered.
However, IoU(ignored) doesn't influence mIoU.
"""
import argparse
import json
from functools import partial
from multiprocessing import Pool
from typing import AbstractSet, Dict, List, Optional, Set, Tuple, Union, cast

import numpy as np
from tqdm import tqdm

from scalabel.common.io import open_write_text
from scalabel.common.logger import logger
from scalabel.common.parallel import NPROC
from scalabel.common.typing import NDArrayF64, NDArrayI32, NDArrayU8
from scalabel.label.io import load, load_label_config
from scalabel.label.transforms import poly2ds_to_mask, rle_to_mask
from scalabel.label.typing import Config, Frame, ImageSize
from scalabel.label.utils import get_leaf_categories

from .result import AVERAGE, Result, Scores, ScoresList
from .utils import filter_labels, reorder_preds


class SegResult(Result):
    """The class for general segmentation evaluation results."""

    IoU: List[Dict[str, float]]
    Acc: List[Dict[str, float]]
    fIoU: float
    pAcc: float

    # pylint: disable=useless-super-delegation
    def __eq__(self, other: "SegResult") -> bool:  # type: ignore
        """Check whether two instances are equal."""
        return super().__eq__(other)

    def summary(
        self,
        include: Optional[AbstractSet[str]] = None,
        exclude: Optional[AbstractSet[str]] = None,
    ) -> Scores:
        """Convert the seg result into a flattened dict as the summary."""
        summary_dict: Dict[str, Union[int, float]] = {}
        for metric, scores_list in self.dict(
            include=include, exclude=exclude  # type: ignore
        ).items():
            if not isinstance(scores_list, list):
                summary_dict[metric] = scores_list
            else:
                summary_dict["m" + metric] = scores_list[-1][AVERAGE]
        return summary_dict


def fast_hist(
    groundtruth: NDArrayU8, prediction: NDArrayU8, size: int
) -> NDArrayI32:
    """Compute the histogram."""
    prediction = prediction.copy()
    # Out-of-range values as `ignored`
    prediction[prediction >= size] = size - 1

    k = np.logical_and(
        # `ignored` is not considered
        np.greater_equal(groundtruth, 0),
        np.less(groundtruth, size - 1),
    )
    return np.bincount(  # type: ignore
        size * groundtruth[k].astype(int) + prediction[k],
        minlength=size**2,
    ).reshape(size, size)


def per_class_iou(hist: NDArrayI32) -> NDArrayF64:
    """Calculate per class iou."""
    ious: NDArrayF64 = np.diag(hist) / (
        hist.sum(1) + hist.sum(0) - np.diag(hist)
    )
    ious[np.isnan(ious)] = 0
    # Last class as `ignored`
    return ious[:-1]  # type: ignore


def per_class_acc(hist: NDArrayI32) -> NDArrayF64:
    """Calculate per class accuracy."""
    accs: NDArrayF64 = np.diag(hist) / hist.sum(axis=0)
    accs[np.isnan(accs)] = 0
    # Last class as `ignored`
    return accs[:-1]  # type: ignore


def whole_acc(hist: NDArrayI32) -> float:
    """Calculate whole accuray."""
    hist = hist[:-1]
    return cast(float, np.diag(hist).sum() / hist.sum())


def freq_iou(hist: NDArrayI32) -> float:
    """Calculate frequency iou."""
    ious = per_class_iou(hist)
    hist = hist[:-1]
    freq = hist.sum(axis=1) / hist.sum()
    return cast(float, (ious * freq).sum())


def frame_to_mask(
    frame: Frame,
    categories: Dict[str, int],
    image_size: Optional[ImageSize] = None,
    ignore_label: int = 255,
) -> NDArrayU8:
    """Convert list of labels to a mask."""
    if image_size is not None:
        out_mask: NDArrayU8 = (
            np.ones((image_size.height, image_size.width))
            * ignore_label  # type: ignore
        ).astype(np.uint8)
    else:
        out_mask = np.zeros((0), dtype=np.uint8)
    if frame.labels is None:
        return out_mask
    for label in frame.labels:
        if label.category not in categories:
            continue
        if label.rle is None and label.poly2d is None:
            continue
        cat_id = categories[label.category]
        if label.rle is not None:
            mask = rle_to_mask(label.rle)
            if len(out_mask) == 0:
                out_mask = np.empty_like(mask)
                out_mask.fill(ignore_label)
        elif label.poly2d is not None:
            assert (
                image_size is not None
            ), "Requires ImageSize for Poly2D conversion to RLE"
            mask = poly2ds_to_mask(image_size, label.poly2d)
        out_mask[mask > 0] = cat_id
    return out_mask


def per_image_hist(
    ann_frame: Frame,
    pred_frame: Frame,
    categories: Dict[str, int],
    image_size: Optional[ImageSize] = None,
    ignore_label: int = 255,
) -> Tuple[NDArrayI32, Set[int]]:
    """Calculate per image hist."""
    num_classes = len(categories) + 1  # add an `ignored` class
    assert num_classes >= 2
    assert num_classes <= ignore_label
    gt = frame_to_mask(ann_frame, categories, image_size, ignore_label)
    gt = gt.copy()
    gt[gt == ignore_label] = num_classes - 1
    gt_id_set = set(np.unique(gt).tolist())

    # remove `ignored`
    if num_classes - 1 in gt_id_set:
        gt_id_set.remove(num_classes - 1)

    pred = frame_to_mask(pred_frame, categories, image_size, ignore_label)
    if len(pred) == 0:
        # empty mask
        pred = np.empty_like(gt)
        pred.fill(ignore_label)
    hist = fast_hist(gt.flatten(), pred.flatten(), num_classes)
    return hist, gt_id_set


def evaluate_sem_seg(
    ann_frames: List[Frame],
    pred_frames: List[Frame],
    config: Config,
    nproc: int = NPROC,
) -> SegResult:
    """Evaluate segmentation with Scalabel format.

    Args:
        ann_frames: the ground truth frames.
        pred_frames: the prediction frames.
        config: Metadata config.
        nproc: the number of process.

    Returns:
        SegResult: evaluation results.
    """
    categories = get_leaf_categories(config.categories)
    cat_map = {cat.name: id for id, cat in enumerate(categories)}
    ignore_label = 255
    pred_frames = reorder_preds(ann_frames, pred_frames)
    ann_frames = filter_labels(ann_frames, categories)
    pred_frames = filter_labels(pred_frames, categories)

    logger.info("evaluating...")
    if nproc > 1:
        with Pool(nproc) as pool:
            hist_and_gt_id_sets = pool.starmap(
                partial(
                    per_image_hist,
                    categories=cat_map,
                    image_size=config.imageSize,
                    ignore_label=ignore_label,
                ),
                tqdm(zip(ann_frames, pred_frames), total=len(ann_frames)),
            )
    else:
        hist_and_gt_id_sets = [
            per_image_hist(
                ann_frame,
                pred_frame,
                categories=cat_map,
                image_size=config.imageSize,
                ignore_label=ignore_label,
            )
            for ann_frame, pred_frame in tqdm(
                zip(ann_frames, pred_frames), total=len(ann_frames)
            )
        ]

    logger.info("accumulating...")
    num_classes = len(cat_map) + 1
    hist: NDArrayI32 = np.zeros((num_classes, num_classes), dtype=np.int32)
    gt_id_set = set()
    for (hist_, gt_id_set_) in hist_and_gt_id_sets:
        hist += hist_
        gt_id_set.update(gt_id_set_)

    ious = per_class_iou(hist)
    accs = per_class_acc(hist)
    iou_scores = [
        {cat_name: 100 * score for cat_name, score in zip(cat_map, ious)},
        {AVERAGE: np.multiply(np.mean(ious[list(gt_id_set)]), 100)},
    ]
    acc_scores = [
        {cat_name: 100 * score for cat_name, score in zip(cat_map, accs)},
        {AVERAGE: np.multiply(np.mean(accs[list(gt_id_set)]), 100)},
    ]
    res_dict: Dict[str, Union[float, ScoresList]] = dict(
        IoU=iou_scores,
        Acc=acc_scores,
        fIoU=np.multiply(freq_iou(hist), 100),  # pylint: disable=invalid-name
        pAcc=np.multiply(whole_acc(hist), 100),  # pylint: disable=invalid-name
    )

    logger.info("GT id set [%s]", ",".join(str(s) for s in gt_id_set))
    return SegResult(**res_dict)


def parse_arguments() -> argparse.Namespace:
    """Parse the arguments."""
    parser = argparse.ArgumentParser(description="Segmentation evaluation.")
    parser.add_argument(
        "--gt", "-g", required=True, help="path to seg ground truth"
    )
    parser.add_argument(
        "--result", "-r", required=True, help="path to seg results"
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
        help="Output file for seg evaluation results.",
    )
    parser.add_argument(
        "--nproc",
        "-p",
        type=int,
        default=NPROC,
        help="number of processes for seg evaluation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    dataset = load(args.gt, args.nproc)
    gts, cfg = dataset.frames, dataset.config
    preds = load(args.result).frames
    if args.config is not None:
        cfg = load_label_config(args.config)
    assert cfg is not None
    eval_result = evaluate_sem_seg(gts, preds, cfg, args.nproc)
    logger.info(eval_result)
    logger.info(eval_result.summary())
    if args.out_file:
        with open_write_text(args.out_file) as fp:
            json.dump(eval_result.dict(), fp, indent=2)
