"""Evaluation code for panoptic segmentation.

############################################################################
Code adapted from:
https://github.com/cocodataset/panopticapi/blob/master/panopticapi/evaluation.py
Copyright (c) 2018, Alexander Kirillov
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of the FreeBSD Project.
############################################################################
"""
import argparse
import json
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from typing import AbstractSet, Dict, List, Optional, Union
from tqdm import tqdm

import numpy as np
from PIL import Image
from pycocotools.mask import iou  # type: ignore
from scalabel.common.io import open_write_text
from scalabel.common.logger import logger
from scalabel.common.parallel import NPROC
from scalabel.label.io import load, load_label_config
from scalabel.label.typing import Category, Config, Frame, ImageSize
from .result import OVERALL, Result, Scores, ScoresList
from .utils import label_ids_to_int, parse_seg_objects, reorder_preds

STUFF = "STUFF"
THING = "THING"


class PanSegResult(Result):
    """The class for panoptic segmentation evaluation results."""

    PQ: List[Dict[str, float]]
    SQ: List[Dict[str, float]]
    RQ: List[Dict[str, float]]
    N: List[Dict[str, int]]  # pylint: disable=invalid-name

    # pylint: disable=useless-super-delegation
    def __eq__(self, other: "PanSegResult") -> bool:  # type: ignore
        """Check whether two instances are equal."""
        return super().__eq__(other)

    def summary(
        self,
        include: Optional[AbstractSet[str]] = None,
        exclude: Optional[AbstractSet[str]] = None,
    ) -> Scores:
        """Convert the pan_seg data into a flattened dict as the summary."""
        summary_dict: Dict[str, Union[int, float]] = {}
        for metric, scores_list in self.dict(
            include=include, exclude=exclude  # type: ignore
        ).items():
            summary_dict[f"{metric}/{STUFF}"] = scores_list[1][STUFF]
            summary_dict[f"{metric}/{THING}"] = scores_list[1][THING]
            summary_dict[metric] = scores_list[-1][OVERALL]
        return summary_dict


class PQStatCat:
    """PQ statistics for each category."""

    def __init__(self) -> None:
        """Initialize method."""
        self.iou: float = 0.0
        self.tp: int = 0  # pylint: disable=invalid-name
        self.fp: int = 0  # pylint: disable=invalid-name
        self.fn: int = 0  # pylint: disable=invalid-name

    def __iadd__(self, pq_stat_cat: "PQStatCat") -> "PQStatCat":
        """Adding definition."""
        self.iou += pq_stat_cat.iou
        self.tp += pq_stat_cat.tp
        self.fp += pq_stat_cat.fp
        self.fn += pq_stat_cat.fn
        return self


class PQStat:
    """PQ statistics for an image of the whole dataset."""

    def __init__(self) -> None:
        """Initialize the PQStatCat dict."""
        self.pq_per_cats: Dict[int, PQStatCat] = defaultdict(PQStatCat)

    def __getitem__(self, category_id: int) -> PQStatCat:
        """Get a PQStatCat object given category."""
        return self.pq_per_cats[category_id]

    def __iadd__(self, pq_stat: "PQStat") -> "PQStat":
        """Adding definition."""
        for category_id, pq_stat_cat in pq_stat.pq_per_cats.items():
            self.pq_per_cats[category_id] += pq_stat_cat
        return self

    def pq_average(self, categories: List[Category]) -> Dict[str, float]:
        """Calculate averatge metrics over categories."""
        pq, sq, rq, n = 0.0, 0.0, 0.0, 0
        for category_id, _ in enumerate(categories):
            iou = self.pq_per_cats[category_id].iou
            tp = self.pq_per_cats[category_id].tp
            fp = self.pq_per_cats[category_id].fp
            fn = self.pq_per_cats[category_id].fn

            if tp + fp + fn == 0:
                continue
            pq += (iou / (tp + 0.5 * fp + 0.5 * fn)) * 100
            sq += (iou / tp if tp != 0 else 0) * 100
            rq += (tp / (tp + 0.5 * fp + 0.5 * fn)) * 100
            n += 1

        if n > 0:
            return dict(PQ=pq / n, SQ=sq / n, RQ=rq / n, N=n)
        return dict(PQ=0, SQ=0, RQ=0, N=0)


def pq_per_image(
    ann_frame: Frame,
    pred_frame: Frame,
    categories: List[Category],
    image_size: Optional[ImageSize] = None,
) -> PQStat:
    """Calculate PQStar for each image."""
    gt_rles, gt_labels, gt_ids, gt_ignores = parse_seg_objects(
        ann_frame.labels if ann_frame.labels is not None else [],
        categories,
        image_size=image_size,
    )
    pred_rles, pred_labels, pred_ids, _ = parse_seg_objects(
        pred_frame.labels if pred_frame.labels is not None else [],
        categories,
        image_size=image_size,
    )

    ious = iou(
        pred_rles,
        gt_rles,
        [False for _ in range(len(gt_rles))],
    ).T
    iofs = iou(
        pred_rles,
        gt_ignores,
        [True for _ in range(len(gt_ignores))],
    )
    print(iofs)
    cat_equals = gt_labels.reshape(-1, 1) == pred_labels.reshape(1, -1)
    ious *= cat_equals

    max_ious = ious.max(axis=1)
    max_idxs = ious.argmax(axis=1)
    inv_iofs = 1 - iofs.sum(axis=0)
    assert False
    # gt_bitmask = np.asarray(Image.open(gt_path), dtype=np.uint8)
    # if not pred_path:
    #     pred_bitmask = gen_blank_bitmask(gt_bitmask.shape)
    # else:
    #     pred_bitmask = np.asarray(Image.open(pred_path), dtype=np.uint8)

    # gt_masks, gt_ids, gt_attrs, gt_cats = parse_bitmask(gt_bitmask)
    # pred_masks, pred_ids, pred_attrs, pred_cats = parse_bitmask(pred_bitmask)

    # gt_valids = np.logical_not(np.bitwise_and(gt_attrs, 3).astype(bool))
    # pred_valids = np.logical_not(np.bitwise_and(pred_attrs, 3).astype(bool))

    # ious, iofs = bitmask_intersection_rate(gt_masks, pred_masks)
    # cat_equals = gt_cats.reshape(-1, 1) == pred_cats.reshape(1, -1)
    # ious *= cat_equals

    # max_ious = ious.max(axis=1)
    # max_idxs = ious.argmax(axis=1)
    # inv_iofs = 1 - iofs[gt_valids].sum(axis=0)

    pq_stat = PQStat()
    pred_matched = set()
    for i in range(len(gt_ids)):
        if not gt_valids[i]:
            continue
        cat_i = gt_cats[i]
        if max_ious[i] <= 0.5 or not pred_valids[max_idxs[i]]:
            pq_stat[cat_i].fn += 1
        else:
            pq_stat[cat_i].tp += 1
            pq_stat[cat_i].iou += max_ious[i]
            pred_matched.add(max_idxs[i])

    for j in range(len(pred_ids)):
        if not pred_valids[j] or j in pred_matched or inv_iofs[j] > 0.5:
            continue
        pq_stat[pred_cats[j]].fp += 1
    return pq_stat


def evaluate_pan_seg(
    ann_frames: List[Frame],
    pred_frames: List[Frame],
    config: Config,
    nproc: int = NPROC,
) -> PanSegResult:
    """Evaluate panoptic segmentation with Scalabel format.

    Args:
        ann_frames: the ground truth frames.
        pred_frames: the prediction frames.
        config: Metadata config.
        nproc: the number of process.

    Returns:
        PanSegResult: evaluation results.
    """
    categories = config.categories
    categories_stuff = [
        category for category in categories if not category.isThing
    ]
    categories_thing = [
        category for category in categories if category.isThing
    ]
    category_names = [category.name for category in categories]
    pred_frames = reorder_preds(ann_frames, pred_frames)
    label_ids_to_int(ann_frames)

    logger.info("evaluating...")
    if nproc > 1:
        with Pool(nproc) as pool:
            pq_stats = pool.starmap(
                partial(
                    pq_per_image,
                    categories=categories,
                    image_size=config.imageSize,
                ),
                tqdm(zip(ann_frames, pred_frames), total=len(ann_frames)),
            )
    else:
        pq_stats = [
            pq_per_image(
                ann_frame,
                pred_frame,
                categories=categories,
                image_size=config.imageSize,
            )
            for ann_frame, pred_frame in tqdm(
                zip(ann_frames, pred_frames), total=len(ann_frames)
            )
        ]
    pq_stat = PQStat()
    for pq_stat_ in pq_stats:
        pq_stat += pq_stat_

    logger.info("accumulating...")
    res_dict: Dict[str, ScoresList] = {}
    for category_name, category in zip(category_names, categories):
        result = pq_stat.pq_average([category])
        for metric, score in result.items():
            if metric not in res_dict:
                res_dict[metric] = [{}, {}, {}]
            res_dict[metric][0][category_name] = score

    result = pq_stat.pq_average(categories_stuff)
    for metric, score in result.items():
        res_dict[metric][1][STUFF] = score
    result = pq_stat.pq_average(categories_thing)
    for metric, score in result.items():
        res_dict[metric][1][THING] = score
    result = pq_stat.pq_average(categories)
    for metric, score in result.items():
        res_dict[metric][2][OVERALL] = score

    return PanSegResult(**res_dict)


def parse_arguments() -> argparse.Namespace:
    """Parse the arguments."""
    parser = argparse.ArgumentParser(
        description="Panoptic segmentation evaluation."
    )
    parser.add_argument(
        "--gt", "-g", required=True, help="path to panseg ground truth"
    )
    parser.add_argument(
        "--result", "-r", required=True, help="path to panseg results"
    )
    parser.add_argument(
        "--config",
        "-c",
        default=None,
        help="Path to config toml file. Contains definition of categories, "
        "and optionally attributes and resolution. For an example "
        "see scalabel/label/configs.toml",
    )
    parser.add_argument(
        "--out-file",
        default="",
        help="Output file for panseg evaluation results.",
    )
    parser.add_argument(
        "--nproc",
        "-p",
        type=int,
        default=NPROC,
        help="number of processes for panseg evaluation",
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
    eval_result = evaluate_pan_seg(gts, preds, cfg, args.nproc)
    logger.info(eval_result)
    logger.info(eval_result.summary())
    if args.out_file:
        with open_write_text(args.out_file) as fp:
            json.dump(eval_result.dict(), fp, indent=2)
