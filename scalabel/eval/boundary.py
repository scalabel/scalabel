"""Evaluation code for boundary detection.

Code adapted from:
https://github.com/fperazzi/davis/blob/master/python/lib/davis/measures/f_boundary.py

Source License

BSD 3-Clause License

Copyright (c) 2017,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.s
############################################################################

Based on:
----------------------------------------------------------------------------
A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation
Copyright (c) 2016 Federico Perazzi
Licensed under the BSD License [see LICENSE for details]
Written by Federico Perazzi
----------------------------------------------------------------------------
"""
import argparse
import json
from functools import partial
from multiprocessing import Pool
from typing import AbstractSet, Dict, List, Optional, Union

import numpy as np
from skimage.morphology import binary_dilation, disk  # type: ignore
from tqdm import tqdm

from ..common.io import open_write_text
from ..common.logger import logger
from ..common.parallel import NPROC
from ..common.typing import NDArrayF64, NDArrayU8
from ..label.io import load, load_label_config
from ..label.transforms import rle_to_mask
from ..label.typing import Category, Config, Frame, ImageSize
from ..label.utils import get_leaf_categories, get_parent_categories
from .result import AVERAGE, Result, Scores, ScoresList
from .utils import filter_labels, reorder_preds

BOUND_PIXELS = [1, 2, 5]


class BoundaryResult(Result):
    """The class for boundary detection evaluation results."""

    F1_pix1: List[Dict[str, float]]
    F1_pix2: List[Dict[str, float]]
    F1_pix5: List[Dict[str, float]]

    # pylint: disable=useless-super-delegation
    def __eq__(self, other: "BoundaryResult") -> bool:  # type: ignore
        """Check whether two instances are equal."""
        return super().__eq__(other)

    def summary(
        self,
        include: Optional[AbstractSet[str]] = None,
        exclude: Optional[AbstractSet[str]] = None,
    ) -> Scores:
        """Convert data into a flattened dict as the summary."""
        summary_dict: Dict[str, Union[int, float]] = {}
        for metric, scores_list in self.dict(
            include=include, exclude=exclude  # type: ignore
        ).items():
            for category, score in scores_list[-2].items():
                summary_dict[f"{metric}/{category}"] = score
            summary_dict[metric] = scores_list[-1][AVERAGE]
        return summary_dict


def eval_bdry_per_thr(
    gt_mask: NDArrayU8, pd_mask: NDArrayU8, bound_pix: int
) -> float:
    """Compute mean, recall, and decay from per-threshold evaluation."""
    gt_dil = binary_dilation(gt_mask, disk(bound_pix))
    pd_dil = binary_dilation(pd_mask, disk(bound_pix))

    # Get the intersection
    gt_match = gt_mask * pd_dil
    pd_match = pd_mask * gt_dil

    # Area of the intersection
    n_gt = np.sum(gt_mask)
    n_pd = np.sum(pd_mask)

    # Compute precision and recall
    if n_pd == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_pd > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_pd == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(pd_match) / float(n_pd)
        recall = np.sum(gt_match) / float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        f_score = 0.0
    else:
        f_score = 2.0 * precision * recall / (precision + recall)

    return f_score


def eval_bdry_per_frame(
    gt_frame: Frame,
    pred_frame: Frame,
    image_size: ImageSize,
    categories: Dict[str, List[Category]],
) -> Dict[str, NDArrayF64]:
    """Compute mean, recall, and decay from per-frame evaluation."""
    w, h = image_size.width, image_size.height
    blank_mask = np.zeros((h, w))
    task2arr: Dict[str, NDArrayF64] = {}  # str -> 2d array
    if gt_frame.labels is None:
        gt_frame.labels = []
    if pred_frame.labels is None:
        pred_frame.labels = []
    gt_masks = {
        l.category: l.rle for l in gt_frame.labels if l.rle is not None
    }
    pd_masks = {
        l.category: l.rle for l in pred_frame.labels if l.rle is not None
    }
    for task_name, cats in categories.items():
        task_scores: List[List[float]] = []
        for cat in cats:
            gt_mask = (
                rle_to_mask(gt_masks[cat.name])
                if cat.name in gt_masks
                else blank_mask
            )
            gt_mask = gt_mask > 0
            pd_mask = (
                rle_to_mask(pd_masks[cat.name])
                if cat.name in pd_masks
                else blank_mask
            )
            pd_mask = pd_mask > 0
            cat_scores = [
                eval_bdry_per_thr(
                    gt_mask,
                    pd_mask,
                    bound_pixel,
                )
                for bound_pixel in BOUND_PIXELS
            ]
            task_scores.append(cat_scores)
        task2arr[task_name] = np.array(task_scores)

    return task2arr


def merge_results(
    task2arr_list: List[Dict[str, NDArrayF64]],
    categories: Dict[str, List[Category]],
) -> Dict[str, NDArrayF64]:
    """Merge F-score results from all images."""
    task2arr: Dict[str, NDArrayF64] = {
        task_name: np.stack(
            [task2arr_img[task_name] for task2arr_img in task2arr_list]
        ).mean(axis=0)
        for task_name in categories
    }

    for task_name, arr2d in task2arr.items():
        arr2d *= 100
        arr_mean = arr2d.mean(axis=0, keepdims=True)
        task2arr[task_name] = np.concatenate([arr2d, arr_mean], axis=0)

    avg_arr: NDArrayF64 = np.stack([arr2d[-1] for arr2d in task2arr.values()])
    task2arr[AVERAGE] = avg_arr.mean(axis=0)

    return task2arr


def generate_results(
    task2arr: Dict[str, NDArrayF64], categories: Dict[str, List[Category]]
) -> BoundaryResult:
    """Render the evaluation results."""
    res_dict: Dict[str, ScoresList] = {
        f"F1_pix{bound_pixel}": [{} for _ in range(5)]
        for bound_pixel in BOUND_PIXELS
    }

    cur_ind = 0
    for task_name, arr2d in task2arr.items():
        if task_name == AVERAGE:
            continue
        for cat, arr1d in zip(categories[task_name], arr2d):
            for bound_pixel, f_score in zip(BOUND_PIXELS, arr1d):
                res_dict[f"F1_pix{bound_pixel}"][cur_ind][cat.name] = f_score
        cur_ind += 1

    for task_name, arr2d in task2arr.items():
        task_name = task_name.upper()
        if task_name == AVERAGE:
            continue
        arr1d = arr2d[-1]
        for bound_pixel, f_score in zip(BOUND_PIXELS, arr1d):
            res_dict[f"F1_pix{bound_pixel}"][-2][task_name] = f_score

    for bound_pixel, f_score in zip(BOUND_PIXELS, task2arr[AVERAGE]):
        res_dict[f"F1_pix{bound_pixel}"][-1][AVERAGE] = f_score

    return BoundaryResult(**res_dict)


def evaluate_boundary(
    ann_frames: List[Frame],
    pred_frames: List[Frame],
    config: Config,
    nproc: int = NPROC,
) -> BoundaryResult:
    """Evaluate boundary detection with Scalabel format.

    Args:
        ann_frames: the ground truth frames.
        pred_frames: the prediction frames.
        config: Metadata config.
        nproc: the number of process.

    Returns:
        BoundaryResult: evaluation results.
    """
    logger.info("evaluating...")
    categories = get_parent_categories(config.categories)
    if not categories:
        categories = {"boundary": get_leaf_categories(config.categories)}
    all_cats = [c for cat in categories.values() for c in cat]
    pred_frames = reorder_preds(ann_frames, pred_frames)
    ann_frames = filter_labels(ann_frames, all_cats)
    pred_frames = filter_labels(pred_frames, all_cats)
    assert config.imageSize is not None
    if nproc > 1:
        with Pool(nproc) as pool:
            task2arr_list = pool.starmap(
                partial(
                    eval_bdry_per_frame,
                    image_size=config.imageSize,
                    categories=categories,
                ),
                tqdm(zip(ann_frames, pred_frames), total=len(ann_frames)),
            )
    else:
        task2arr_list = [
            eval_bdry_per_frame(
                gt_path,
                pred_path,
                image_size=config.imageSize,
                categories=categories,
            )
            for gt_path, pred_path in tqdm(
                zip(ann_frames, pred_frames), total=len(ann_frames)
            )
        ]
    logger.info("accumulating...")
    task2arr = merge_results(task2arr_list, categories)
    result = generate_results(task2arr, categories)
    return result


def parse_arguments() -> argparse.Namespace:
    """Parse the arguments."""
    parser = argparse.ArgumentParser(description="Boundary evaluation.")
    parser.add_argument(
        "--gt", "-g", required=True, help="path to boundary ground truth"
    )
    parser.add_argument(
        "--result", "-r", required=True, help="path to boundary results"
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
        help="Output file for boundary evaluation results.",
    )
    parser.add_argument(
        "--nproc",
        "-p",
        type=int,
        default=NPROC,
        help="number of processes for boundary evaluation",
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
    eval_result = evaluate_boundary(gts, preds, cfg, args.nproc)
    logger.info(eval_result)
    logger.info(eval_result.summary())
    if args.out_file:
        with open_write_text(args.out_file) as fp:
            json.dump(eval_result.json(), fp)
