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
from pycocotools import mask as mask_utils

import teta
from teta.datasets import BDDMOTS as TrackEvalBDD100KMOTS
from teta.eval import eval_sequence
from trackeval.utils import TrackEvalException

from ..common.io import open_write_text
from ..common.logger import logger
from ..common.parallel import NPROC
from ..common.typing import NDArrayF64
from ..label.io import group_and_sort, load, load_label_config
from ..label.typing import Category, Config, Frame
from ..label.utils import get_leaf_categories, get_parent_categories
from .result import AVERAGE, OVERALL, Result, ScoresList
from .utils import check_overlap, label_ids_to_int

TETAScore = Dict[str, Dict[str, Dict[str, NDArrayF64]]]

METRICS = [teta.metrics.TETA(exhaustive=True)]
METRIC_NAMES = [metric.get_name() for metric in METRICS]
SCORES = ["TETA", "LocA", "AssocA", "ClsA", "LocRe", "LocPr",
          "AssocRe", "AssocPr", "ClsRe", "ClsPr"]
SCORE_TO_AVERAGE = set(["TETA", "LocA", "AssocA", "ClsA"])


class TETAResult(Result):
    """The class for TETA tracking evaluation results."""

    mTETA: float
    mLocA: float
    mAssocA: float
    TETA: List[Dict[str, float]]
    LocA: List[Dict[str, float]]
    AssocA: List[Dict[str, float]]
    ClsA: List[Dict[str, float]]
    LocRe: List[Dict[str, float]]
    LocPr: List[Dict[str, float]]
    AssocRe: List[Dict[str, float]]
    AssocPr: List[Dict[str, float]]
    ClsRe: List[Dict[str, float]]
    ClsPr: List[Dict[str, float]]


class BDD100K(TrackEvalBDD100KMOTS):  # type: ignore
    """TETA dataset class for BDD100K."""

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

        self.classes_to_eval = valid_classes
        self.class_list = [
            cls.lower() if cls.lower() in valid_classes else None
            for cls in self.classes_to_eval
        ]

        self.cls_name2clsid = {
            k: v for k, v in self.class_name_to_class_id.items() if k in self.class_list
        }
        self.clsid2cls_name = {
            v: k for k, v in self.class_name_to_class_id.items() if k in self.class_list
        }
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


    def get_display_name(self, tracker: str) -> str:
        """Get display name of tracker."""
        return tracker

    def get_output_fol(self, tracker: str) -> str:
        """Get output folder (useless)."""
        return ""

    def _compute_valid_imgs_mappings(self, gts: List[Frame]):
        """Computes mappings from videos to corresponding tracks and images."""

        valid_img_ids = []

        for frame in gts:
            if frame.labels is None or frame.labels == []:
                continue
            else:
                valid_img_ids.append(frame.videoName + str(frame.frameIndex))

        return  valid_img_ids

    def _compute_image_to_timestep_mappings(self):
        """Computes a mapping from images to timestep in sequence."""
        images = {}
        for image in self.gts:
            images[image.videoName + str(image.frameIndex)] = image

        curr_imgs = [img_id for img_id in self.valid_img_ids]
        curr_imgs = sorted(curr_imgs, key=lambda x: images[x].frameIndex)
        imgs_to_timestep = {
            curr_imgs[i]: i for i in range(len(curr_imgs))
        }

        return imgs_to_timestep



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
                "ids": "tk_ids",
                "classes": "tk_classes",
                "dets": "tk_dets",
            }
        for k, v in key_map.items():
            raw_data[v] = raw_data.pop(k)
        raw_data["num_timesteps"] = num_timesteps
        raw_data["seq"] = seq
        return raw_data


def evaluate_videos(
    gts: List[Frame], results: List[Frame]
) -> Dict[str, TETAScore]:
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
    res: Dict[str, TETAScore],
    class_list: List[str],
    super_classes: Dict[str, List[Category]],
) -> TETAResult:
    """Combine evaluation results."""
    # collecting combined cls keys (cls averaged, det averaged, super classes)
    cls_keys = []
    res["COMBINED_SEQ"] = {}
    # combine sequences for each class
    for c_cls in class_list:
        res["COMBINED_SEQ"][c_cls] = {}
        for metric, mname in zip(METRICS, METRIC_NAMES):
            curr_res = {
                seq_key: seq_value[c_cls][mname]
                for seq_key, seq_value in res.items()
                if seq_key != "COMBINED_SEQ"
            }
            # combine results over all sequences and then over all classes
            res["COMBINED_SEQ"][c_cls][mname] = metric.combine_sequences(curr_res)

    # combine classes

    video_keys = ["COMBINED_SEQ"]
    for v_key in video_keys:
        cls_keys += [AVERAGE,OVERALL]
        res[v_key][AVERAGE] = {}
        for metric, mname in zip(METRICS, METRIC_NAMES):
            cls_res = {
                cls_key: cls_value[mname]
                for cls_key, cls_value in res[v_key].items()
                if cls_key not in cls_keys
            }
            res[v_key]["AVERAGE"][
                mname
            ] = metric.combine_classes_class_averaged(
                cls_res, ignore_empty=True
            )
        res[v_key][OVERALL] = {}
        for metric, mname in zip(METRICS, METRIC_NAMES):
            cls_res = {
                cls_key: cls_value[mname]
                for cls_key, cls_value in res[v_key].items()
                if cls_key not in cls_keys
            }
            res[v_key]["OVERALL"][
                mname
            ] = metric.combine_classes_det_averaged(
                cls_res
            )

    # combine classes to super classes
    if super_classes:
        for cat, sub_cats in super_classes.items():
            sub_cats_names = [c.name for c in sub_cats]
            cls_keys.append(cat)
            res["COMBINED_SEQ"][cat] = {}
            for metric, mname in zip(METRICS, METRIC_NAMES):
                cat_res = {
                    cls_key: cls_value[mname]
                    for cls_key, cls_value in res["COMBINED_SEQ"].items()
                    if cls_key in sub_cats_names
                }
                res["COMBINED_SEQ"][cat][
                    mname
                ] = metric.combine_classes_det_averaged(cat_res)

    return res["COMBINED_SEQ"]


def generate_results(
    scores: TETAScore,
    classes: List[Category],
    super_classes: Dict[str, List[Category]],
) -> TETAResult:
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
                class_name: score["TETA"][50][metric].mean() * 100.0
                # class_name: METRICS[0]._summary_row(
                #             score['TETA'][50]
                #         )
                for class_name, score in scores.items()
                if class_name in class_name_set
            }
            for class_name_set in class_name_sets
        ]
        for metric in SCORES
    }

    res_dict.update(
        {
            "m" + metric: scores['AVERAGE']["TETA"][50][metric].mean() * 100.0
            for metric in SCORE_TO_AVERAGE
        }
    )
    return TETAResult(**res_dict)


def evaluate_seg_track_teta(
    gts: List[List[Frame]],
    results: List[List[Frame]],
    config: Config,
    nproc: int = NPROC,
) -> TETAResult:
    """Evaluate HOTA metrics for a Scalabel format dataset.

    Args:
        gts: the ground truth annotations in Scalabel format.
        results: the prediction results in Scalabel format.
        config: Config object.
        nproc: number of processes.

    Returns:
        TrackResult: evaluation results.
    """
    logger.info("Tracking evaluation with TETA metrics.")
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
    parser = argparse.ArgumentParser(description="MOTS evaluation with TETA.")
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
    eval_result = evaluate_seg_track_teta(
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
