"""Segmentation Tracking HOTA evaluation code.

Code adapted from:
https://github.com/JonathonLuiten/TrackEval
"""
import argparse
import json
import time
from multiprocessing import Pool
from typing import Dict, List

import numpy as np
from pycocotools import mask as mask_utils

from trackeval.datasets import KittiMOTS as TrackEvalKittiMOTS
from trackeval.utils import TrackEvalException

from ..common.io import open_write_text
from ..common.logger import logger
from ..common.parallel import NPROC
from ..label.io import group_and_sort, load, load_label_config
from ..label.typing import Config, Frame
from ..label.utils import get_leaf_categories, get_parent_categories
from .hota import (
    HOTAScore,
    HOTAResult,
    METRICS,
    METRIC_NAMES,
    combine_results,
    generate_results,
    eval_sequence,
)
from .utils import check_overlap, label_ids_to_int


class BDD100K(TrackEvalKittiMOTS):  # type: ignore
    """HOTA dataset class for BDD100K."""

    def __init__(  # pylint: disable=super-init-not-called
        self, gts: List[Frame], results: List[Frame]
    ) -> None:
        """Initialize dataset."""
        assert len(gts) == len(results), (
            f"len(gts)={len(gts)}, len(results)={len(results)}"
        )
        assert gts[0].videoName == results[0].videoName
        self.gts, self.results = gts, results

        # Get classes to eval
        valid_classes = [
            "pedestrian",
            "rider",
            "car",
            "bus",
            "truck",
            "train",
            "motorcycle",
            "bicycle",
        ]
        self.classes_to_eval = valid_classes
        self.class_list = [
            cls.lower() if cls.lower() in valid_classes else None
            for cls in self.classes_to_eval
        ]
        if not all(self.class_list):
            raise TrackEvalException(
                "Attempted to evaluate an invalid class. Only classes "
                "[pedestrian, rider, car, "
                "bus, truck, train, motorcycle, bicycle] are valid."
            )
        self.super_categories = {
            "human": [
                cls
                for cls in ["pedestrian", "rider"]
                if cls in self.class_list
            ],
            "vehicle": [
                cls
                for cls in ["car", "truck", "bus", "train"]
                if cls in self.class_list
            ],
            "bike": [
                cls
                for cls in ["motorcycle", "bicycle"]
                if cls in self.class_list
            ],
        }
        self.distractor_classes = ["other person", "trailer", "other vehicle"]
        self.class_name_to_class_id = {
            "pedestrian": 1,
            "rider": 2,
            "other person": 3,
            "car": 4,
            "bus": 5,
            "truck": 6,
            "train": 7,
            "trailer": 8,
            "other vehicle": 9,
            "motorcycle": 10,
            "bicycle": 11,
        }

    def get_display_name(self, tracker: str) -> str:
        """Get display name of tracker."""
        return tracker

    def get_output_fol(self, tracker: str) -> str:
        """Get output folder (useless)."""
        return ""

    def _load_raw_file(self, tracker, seq, is_gt):
        """Setup HOTA raw data."""
        data = self.gts if is_gt else self.results
        label_ids_to_int(data)
        # Convert data to required format
        num_timesteps = len(data)
        data_keys = ["ids", "classes", "dets"]
        if is_gt:
            data_keys += ["gt_ignore_region"]
        raw_data = {key: [None] * num_timesteps for key in data_keys}

        for t in range(num_timesteps):
            ig_ids = []
            keep_ids = []
            all_masks = []
            labels = data[t].labels
            if labels is None:
                continue
            for i, ann in enumerate(labels):
                if is_gt and (
                    ann.category in self.distractor_classes
                    or (
                        ann.attributes is not None
                        and "crowd" in ann.attributes
                        and ann.attributes["crowd"]
                    )
                ):
                    ig_ids.append(i)
                else:
                    keep_ids.append(i)

            if keep_ids:
                raw_data["dets"][t] = [
                    {
                        "size": list(labels[i].rle.size),
                        "counts": labels[i].rle.counts.encode(
                            encoding="UTF-8"
                        ),
                    }
                    for i in keep_ids
                ]
                raw_data["ids"][t] = np.atleast_1d(
                    [labels[i].id for i in keep_ids]
                ).astype(int)
                raw_data["classes"][t] = np.atleast_1d(
                    [
                        self.class_name_to_class_id[labels[i].category]
                        for i in keep_ids
                    ]
                ).astype(int)
                all_masks += raw_data["dets"][t]
            else:
                raw_data["dets"][t] = []
                raw_data["ids"][t] = np.empty(0).astype(int)
                raw_data["classes"][t] = np.empty(0).astype(int)

            if is_gt:
                if ig_ids:
                    time_ignore = [
                        {
                            "size": list(labels[i].rle.size),
                            "counts": labels[i].rle.counts.encode(
                                encoding="UTF-8"
                            ),
                        }
                        for i in ig_ids
                    ]
                    raw_data["gt_ignore_region"][t] = mask_utils.merge(
                        list(time_ignore), intersect=False
                    )
                    all_masks += [raw_data["gt_ignore_region"][t]]
                else:
                    raw_data["gt_ignore_region"][t] = mask_utils.merge(
                        [], intersect=False
                    )

        if is_gt:
            key_map = {
                "ids": "gt_ids",
                "classes": "gt_classes",
                "dets": "gt_dets",
            }
        else:
            key_map = {
                "ids": "tracker_ids",
                "classes": "tracker_classes",
                "dets": "tracker_dets",
            }
        for k, v in key_map.items():
            raw_data[v] = raw_data.pop(k)
        raw_data["num_timesteps"] = num_timesteps
        raw_data["seq"] = seq
        return raw_data


def evaluate_videos(
    gts: List[Frame], results: List[Frame]
) -> Dict[str, HOTAScore]:
    """Evaluate videos."""
    dset, seq = BDD100K(gts, results), gts[0].videoName
    assert seq is not None
    res = {
        seq: eval_sequence(
            seq, dset, "tracker", dset.class_list, METRICS, METRIC_NAMES
        )
    }
    return res


def evaluate_seg_track_hota(
    gts: List[List[Frame]],
    results: List[List[Frame]],
    config: Config,
    nproc: int = NPROC,
) -> HOTAResult:
    """Evaluate HOTA metrics for a Scalabel format dataset.

    Args:
        gts: the ground truth annotations in Scalabel format.
        results: the prediction results in Scalabel format.
        config: Config object.
        nproc: number of processes.

    Returns:
        TrackResult: evaluation results.
    """
    logger.info("Tracking evaluation with HOTA metrics.")
    t = time.time()
    assert len(gts) == len(results)
    # check overlap of masks
    logger.info("checking for overlap of masks...")
    if check_overlap(
        [frame for res in results for frame in res], config, nproc
    ):
        logger.critical(
            "Found overlap in prediction bitmasks, but segmentation tracking "
            "evaluation does not allow overlaps. Removing such predictions."
        )

    classes = get_leaf_categories(config.categories)
    super_classes = get_parent_categories(config.categories)

    logger.info("evaluating...")
    if nproc > 1:
        with Pool(nproc) as pool:
            result_list = pool.starmap(evaluate_videos, zip(gts, results))
    else:
        result_list = [
            evaluate_videos(gt, res) for gt, res in zip(gts, results)
        ]
    res_per_vid = {k: v for d in result_list for k, v in d.items()}

    logger.info("accumulating...")
    scores = combine_results(
        res_per_vid, [c.name for c in classes], super_classes
    )

    result = generate_results(scores, classes, super_classes)
    t = time.time() - t
    logger.info("evaluation finishes with %.1f s.", t)
    return result


def parse_arguments() -> argparse.Namespace:
    """Parse the arguments."""
    parser = argparse.ArgumentParser(description="MOTS evaluation with HOTA.")
    parser.add_argument(
        "--gt", "-g", required=True, help="path to mots ground truth"
    )
    parser.add_argument(
        "--result", "-r", required=True, help="path to mots results"
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
        help="Output path for mots evaluation results.",
    )
    parser.add_argument(
        "--nproc",
        "-p",
        type=int,
        default=NPROC,
        help="number of processes for mots evaluation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    dataset = load(args.gt, args.nproc)
    gt_frames, cfg = dataset.frames, dataset.config
    if args.config is not None:
        cfg = load_label_config(args.config)
    if cfg is None:
        raise ValueError(
            "Dataset config is not specified. Please use --config"
            " to specify a config for this dataset."
        )
    eval_result = evaluate_seg_track_hota(
        group_and_sort(gt_frames),
        group_and_sort(load(args.result, args.nproc).frames),
        cfg,
        args.nproc,
    )
    logger.info(eval_result)
    logger.info(eval_result.summary())
    if args.out_file:
        with open_write_text(args.out_file) as fp:
            json.dump(eval_result.dict(), fp, indent=2)
