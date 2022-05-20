"""Multi-object Tracking HOTA evaluation code.

Code adapted from:
https://github.com/JonathonLuiten/TrackEval
"""
import argparse
import json
import time
from multiprocessing import Pool
from typing import Dict, List, Union

import numpy as np

import trackeval
from trackeval.datasets import BDD100K as TrackEvalBDD100K
from trackeval.eval import eval_sequence
from trackeval.utils import TrackEvalException

from ..common.io import open_write_text
from ..common.logger import logger
from ..common.parallel import NPROC
from ..common.typing import NDArrayF64
from ..label.io import group_and_sort, load, load_label_config
from ..label.typing import Category, Config, Frame
from ..label.utils import get_leaf_categories, get_parent_categories
from .result import AVERAGE, OVERALL, Result, ScoresList
from .utils import label_ids_to_int

HOTAScore = Dict[str, Dict[str, Dict[str, NDArrayF64]]]

METRICS = [trackeval.metrics.HOTA(), trackeval.metrics.Count()]
METRIC_NAMES = [metric.get_name() for metric in METRICS]
SCORES = ["HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr", "LocA"]
SCORE_TO_AVERAGE = set(["HOTA", "DetA", "AssA"])


class HOTAResult(Result):
    """The class for HOTA tracking evaluation results."""

    mHOTA: float
    mDetA: float
    mAssA: float
    HOTA: List[Dict[str, float]]
    DetA: List[Dict[str, float]]
    AssA: List[Dict[str, float]]
    DetRe: List[Dict[str, float]]
    DetPr: List[Dict[str, float]]
    AssRe: List[Dict[str, float]]
    AssPr: List[Dict[str, float]]
    LocA: List[Dict[str, float]]
    Dets: List[Dict[str, int]]
    IDs: List[Dict[str, int]]


class BDD100K(TrackEvalBDD100K):  # type: ignore
    """HOTA dataset class for BDD100K."""

    def __init__(  # pylint: disable=super-init-not-called
        self, gts: List[Frame], results: List[Frame]
    ) -> None:
        """Initialize dataset."""
        assert len(gts) == len(results)
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
            data_keys += ["gt_crowd_ignore_regions"]
        raw_data = {key: [None] * num_timesteps for key in data_keys}
        for t in range(num_timesteps):
            ig_ids = []
            keep_ids = []
            labels = data[t].labels
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
                raw_data["dets"][t] = np.atleast_2d(
                    [
                        [
                            labels[i].box2d.x1,
                            labels[i].box2d.y1,
                            labels[i].box2d.x2,
                            labels[i].box2d.y2,
                        ]
                        for i in keep_ids
                    ]
                ).astype(float)
                raw_data["ids"][t] = np.atleast_1d(
                    [data[t].labels[i].id for i in keep_ids]
                ).astype(int)
                raw_data["classes"][t] = np.atleast_1d(
                    [
                        self.class_name_to_class_id[data[t].labels[i].category]
                        for i in keep_ids
                    ]
                ).astype(int)
            else:
                raw_data["dets"][t] = np.empty((0, 4)).astype(float)
                raw_data["ids"][t] = np.empty(0).astype(int)
                raw_data["classes"][t] = np.empty(0).astype(int)

            if is_gt:
                if ig_ids:
                    raw_data["gt_crowd_ignore_regions"][t] = np.atleast_2d(
                        [
                            [
                                labels[i].box2d.x1,
                                labels[i].box2d.y1,
                                labels[i].box2d.x2,
                                labels[i].box2d.y2,
                            ]
                            for i in ig_ids
                        ]
                    ).astype(float)
                else:
                    raw_data["gt_crowd_ignore_regions"][t] = np.empty(
                        (0, 4)
                    ).astype(float)

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


def combine_results(
    res: Dict[str, HOTAScore],
    class_list: List[str],
    super_classes: Dict[str, List[Category]],
) -> HOTAScore:
    """Combine evaluation results."""
    combined_cls_keys = []
    res["COMBINED_SEQ"] = {}
    # combine sequences for each class
    for c_cls in class_list:
        res["COMBINED_SEQ"][c_cls] = {}
        for metric, metric_name in zip(METRICS, METRIC_NAMES):
            curr_res = {
                seq_key: seq_value[c_cls][metric_name]
                for seq_key, seq_value in res.items()
                if seq_key != "COMBINED_SEQ"
            }
            res["COMBINED_SEQ"][c_cls][metric_name] = metric.combine_sequences(
                curr_res
            )
    # combine classes
    combined_cls_keys += [AVERAGE, OVERALL, "all"]
    res["COMBINED_SEQ"][AVERAGE] = {}
    res["COMBINED_SEQ"][OVERALL] = {}
    for metric, metric_name in zip(METRICS, METRIC_NAMES):
        cls_res = {
            cls_key: cls_value[metric_name]
            for cls_key, cls_value in res["COMBINED_SEQ"].items()
            if cls_key not in combined_cls_keys
        }
        res["COMBINED_SEQ"][AVERAGE][
            metric_name
        ] = metric.combine_classes_class_averaged(cls_res)
        res["COMBINED_SEQ"][OVERALL][
            metric_name
        ] = metric.combine_classes_det_averaged(cls_res)
    # combine classes to super classes
    if super_classes:
        for cat, sub_cats in super_classes.items():
            sub_cats_names = [c.name for c in sub_cats]
            combined_cls_keys.append(cat)
            res["COMBINED_SEQ"][cat] = {}
            for metric, metric_name in zip(METRICS, METRIC_NAMES):
                cat_res = {
                    cls_key: cls_value[metric_name]
                    for cls_key, cls_value in res["COMBINED_SEQ"].items()
                    if cls_key in sub_cats_names
                }
                res["COMBINED_SEQ"][cat][
                    metric_name
                ] = metric.combine_classes_det_averaged(cat_res)
    return res["COMBINED_SEQ"]


def generate_results(
    scores: HOTAScore,
    classes: List[Category],
    super_classes: Dict[str, List[Category]],
) -> HOTAResult:
    """Compute summary metrics for evaluation results."""
    basic_set = [c.name for c in classes]
    super_set = list(super_classes.keys())
    hyper_set = [AVERAGE, OVERALL]
    class_name_sets = [basic_set, hyper_set]
    if [name for name in scores if name in super_classes]:
        class_name_sets.insert(1, super_set)

    res_dict: Dict[str, Union[int, float, ScoresList]] = {
        metric: [
            {
                class_name: score["HOTA"][metric].mean() * 100.0
                for class_name, score in scores.items()
                if class_name in class_name_set
            }
            for class_name_set in class_name_sets
        ]
        for metric in SCORES
    }
    res_dict.update(
        {
            metric: [
                {
                    class_name: score["Count"][metric]  # type: ignore
                    for class_name, score in scores.items()
                    if class_name in class_name_set
                }
                for class_name_set in class_name_sets
            ]
            for metric in ["Dets", "IDs"]
        }
    )
    res_dict.update(
        {
            "m" + metric: scores[AVERAGE]["HOTA"][metric].mean() * 100.0
            for metric in SCORE_TO_AVERAGE
        }
    )
    return HOTAResult(**res_dict)


def evaluate_track(
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
    parser = argparse.ArgumentParser(description="MOT evaluation with HOTA.")
    parser.add_argument(
        "--gt", "-g", required=True, help="path to mot ground truth"
    )
    parser.add_argument(
        "--result", "-r", required=True, help="path to mot results"
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
        help="Output path for mot evaluation results.",
    )
    parser.add_argument(
        "--nproc",
        "-p",
        type=int,
        default=NPROC,
        help="number of processes for mot evaluation",
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
    eval_result = evaluate_track(
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
