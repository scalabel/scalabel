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

import pycocotools.mask as coco_mask  # type: ignore
from tqdm import tqdm

from scalabel.common.io import open_write_text
from scalabel.common.logger import logger
from scalabel.common.parallel import NPROC
from scalabel.label.io import load, load_label_config
from scalabel.label.typing import Category, Config, Frame, ImageSize
from scalabel.label.utils import get_leaf_categories

from .result import OVERALL, Result, Scores, ScoresList
from .utils import (
    check_overlap,
    label_ids_to_int,
    parse_seg_objects,
    reorder_preds,
)

STUFF = "STUFF"
THING = "THING"


class PanSegResult(Result):
    """The class for panoptic segmentation evaluation results."""

    PQ: List[Dict[str, float]]
    SQ: List[Dict[str, float]]
    RQ: List[Dict[str, float]]
    NUM: List[Dict[str, int]]

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
        self.tpos: int = 0
        self.fpos: int = 0
        self.fneg: int = 0

    def __iadd__(self, pq_stat_cat: "PQStatCat") -> "PQStatCat":
        """Adding definition."""
        self.iou += pq_stat_cat.iou
        self.tpos += pq_stat_cat.tpos
        self.fpos += pq_stat_cat.fpos
        self.fneg += pq_stat_cat.fneg
        return self


class PQStat:
    """PQ statistics for an image of the whole dataset."""

    def __init__(self, categories: List[Category]) -> None:
        """Initialize the PQStatCat dict."""
        self.pq_per_cats: Dict[int, PQStatCat] = defaultdict(PQStatCat)
        self.categories = categories
        self.category_names = [category.name for category in categories]

    def __getitem__(self, category_id: int) -> PQStatCat:
        """Get a PQStatCat object given category."""
        return self.pq_per_cats[category_id]

    def __iadd__(self, pq_stat: "PQStat") -> "PQStat":
        """Adding definition."""
        for category_id, pq_stat_cat in pq_stat.pq_per_cats.items():
            self.pq_per_cats[category_id] += pq_stat_cat
        return self

    def pq_average(self, categories: List[Category]) -> Dict[str, float]:
        """Calculate average metrics over categories."""
        pq, sq, rq, n = 0.0, 0.0, 0.0, 0
        for category in categories:
            category_id = self.category_names.index(category.name)
            iou = self.pq_per_cats[category_id].iou
            tpos = self.pq_per_cats[category_id].tpos
            fpos = self.pq_per_cats[category_id].fpos
            fneg = self.pq_per_cats[category_id].fneg

            if tpos + fpos + fneg == 0:
                continue
            pq += (iou / (tpos + 0.5 * fpos + 0.5 * fneg)) * 100
            sq += (iou / tpos if tpos != 0 else 0) * 100
            rq += (tpos / (tpos + 0.5 * fpos + 0.5 * fneg)) * 100
            n += 1

        if n > 0:
            return dict(PQ=pq / n, SQ=sq / n, RQ=rq / n, NUM=n)
        return dict(PQ=0, SQ=0, RQ=0, NUM=0)


def pq_per_image(
    ann_frame: Frame,
    pred_frame: Frame,
    categories: List[Category],
    ignore_unknown_cats: bool = False,
    image_size: Optional[ImageSize] = None,
) -> PQStat:
    """Calculate PQStar for each image."""
    pq_stat = PQStat(categories)
    gt_rles, gt_labels, _, _ = parse_seg_objects(
        ann_frame.labels if ann_frame.labels is not None else [],
        categories,
        ignore_unknown_cats=ignore_unknown_cats,
        image_size=image_size,
    )

    if (
        pred_frame.labels is None
        or not pred_frame.labels
        or all(
            label.rle is None and label.poly2d is None
            for label in pred_frame.labels
        )
    ):
        # no predictions for image
        for gt_cat in gt_labels:
            pq_stat[gt_cat].fneg += 1
        return pq_stat

    pred_rles, pred_labels, _, _ = parse_seg_objects(
        pred_frame.labels if pred_frame.labels is not None else [],
        categories,
        ignore_unknown_cats=ignore_unknown_cats,
        image_size=image_size,
    )

    ious = coco_mask.iou(
        pred_rles,
        gt_rles,
        [False for _ in range(len(gt_rles))],
    ).T
    iofs = coco_mask.iou(
        pred_rles,
        gt_rles,
        [True for _ in range(len(gt_rles))],
    ).T
    cat_equals = gt_labels.reshape(-1, 1) == pred_labels.reshape(1, -1)
    ious *= cat_equals

    max_ious = ious.max(axis=1)
    max_idxs = ious.argmax(axis=1)
    inv_iofs = 1 - iofs.sum(axis=0)

    pred_matched = set()
    for gt_cat, max_iou, max_idx in zip(gt_labels, max_ious, max_idxs):
        if max_iou <= 0.5:
            pq_stat[gt_cat].fneg += 1
        else:
            pq_stat[gt_cat].tpos += 1
            pq_stat[gt_cat].iou += max_iou
            pred_matched.add(max_idx)

    for j, pred_label in enumerate(pred_labels):
        if j in pred_matched or inv_iofs[j] > 0.5:
            continue
        pq_stat[pred_label].fpos += 1
    return pq_stat


def evaluate_pan_seg(
    ann_frames: List[Frame],
    pred_frames: List[Frame],
    config: Config,
    ignore_unknown_cats: bool = False,
    nproc: int = NPROC,
) -> PanSegResult:
    """Evaluate panoptic segmentation with Scalabel format.

    Args:
        ann_frames: the ground truth frames.
        pred_frames: the prediction frames.
        config: Metadata config.
        ignore_unknown_cats: ignore unknown categories.
        nproc: the number of process.

    Returns:
        PanSegResult: evaluation results.
    """
    categories = get_leaf_categories(config.categories)
    assert all(
        category.isThing is not None for category in categories
    ), "isThing should be defined for all categories for PanSeg."
    categories_stuff = [
        category for category in categories if not category.isThing
    ]
    categories_thing = [
        category for category in categories if category.isThing
    ]
    category_names = [category.name for category in categories]
    pred_frames = reorder_preds(ann_frames, pred_frames)
    label_ids_to_int(ann_frames)
    # check overlap of masks
    logger.info("checking for overlap of masks...")
    if check_overlap(pred_frames, config, nproc):
        logger.critical(
            "Found overlap in prediction bitmasks, but panoptic segmentation "
            "evaluation does not allow overlaps. Removing such predictions."
        )

    logger.info("evaluating...")
    if nproc > 1:
        with Pool(nproc) as pool:
            pq_stats = pool.starmap(
                partial(
                    pq_per_image,
                    categories=categories,
                    ignore_unknown_cats=ignore_unknown_cats,
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
                ignore_unknown_cats=ignore_unknown_cats,
                image_size=config.imageSize,
            )
            for ann_frame, pred_frame in tqdm(
                zip(ann_frames, pred_frames), total=len(ann_frames)
            )
        ]
    pq_stat = PQStat(categories)
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
        "see scalabel/label/testcases/configs.toml",
    )
    parser.add_argument(
        "--out-file",
        default="",
        help="Output file for panseg evaluation results.",
    )
    parser.add_argument(
        "--ignore-unknown-cats",
        action="store_true",
        help="ignore unknown categories for panseg evaluation",
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
    eval_result = evaluate_pan_seg(
        gts, preds, cfg, args.ignore_unknown_cats, args.nproc
    )
    logger.info(eval_result)
    logger.info(eval_result.summary())
    if args.out_file:
        with open_write_text(args.out_file) as fp:
            json.dump(eval_result.dict(), fp, indent=2)
