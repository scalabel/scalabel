"""Multi-object Tracking evaluation code."""
import argparse
import json
import time
import math
from typing import Dict, List, Optional,  AbstractSet, Union

import numpy as np

from ..common.io import open_write_text
from ..common.logger import logger
from ..common.parallel import NPROC
from ..label.io import group_and_sort, load, load_label_config
from ..label.transforms import box2d_to_bbox
from ..label.typing import Config, Frame, Label
from ..label.utils import get_leaf_categories
from .hota import evaluate_track_hota
from .teta import evaluate_track_teta
from .mot import acc_single_video_mot, evaluate_track
from .result import Result

EXCLUDE_METRICS = ["Dets", "IDs", "FP", "FN", "MT", "PT", "ML", "FM"]
IGNORE_MAPPING = {
    "other person": "pedestrian",
    "other vehicle": "car",
    "trailer": "truck",
}
NAME_MAPPING = {
    "bike": "bicycle",
    "caravan": "car",
    "motor": "motorcycle",
    "person": "pedestrian",
    "van": "car",
}

Scores = Dict[str, Union[int, float]]
ScoresList = List[Scores]
AVERAGE = "AVERAGE"
OVERALL = "OVERALL"

class BoxTrackResult(Result):
    """The class for bounding box tracking evaluation results."""
    mTETA: float
    mHOTA: float
    mMOTA: float
    mIDF1: float
    mDetA: float
    mAssA: float
    mMOTP: float
    TETA: List[Dict[str, float]]
    HOTA: List[Dict[str, float]]
    MOTA: List[Dict[str, float]]
    IDF1: List[Dict[str, float]]
    LocA: List[Dict[str, float]]
    AssocA: List[Dict[str, float]]
    ClsA: List[Dict[str, float]]
    LocRe: List[Dict[str, float]]
    LocPr: List[Dict[str, float]]
    AssocRe: List[Dict[str, float]]
    AssocPr: List[Dict[str, float]]
    ClsRe: List[Dict[str, float]]
    ClsPr: List[Dict[str, float]]
    DetA: List[Dict[str, float]]
    AssA: List[Dict[str, float]]
    DetRe: List[Dict[str, float]]
    DetPr: List[Dict[str, float]]
    AssRe: List[Dict[str, float]]
    AssPr: List[Dict[str, float]]
    LocA: List[Dict[str, float]]
    MOTP: List[Dict[str, float]]
    FP: List[Dict[str, int]]
    FN: List[Dict[str, int]]
    IDSw: List[Dict[str, int]]
    MT: List[Dict[str, int]]
    PT: List[Dict[str, int]]
    ML: List[Dict[str, int]]
    FM: List[Dict[str, int]]
    Dets: List[Dict[str, int]]
    IDs: List[Dict[str, int]]

    # pylint: disable=useless-super-delegation
    def __eq__(self, other: "Result") -> bool:  # type: ignore
        """Check whether two instances are equal."""
        return super().__eq__(other)

    def __str__(self) -> str:
        """Convert the data into a printable string."""
        return self.table(exclude=set(EXCLUDE_METRICS))

    def summary(
        self,
        include: Optional[AbstractSet[str]] = None,
        exclude: Optional[AbstractSet[str]] = None,
    ) -> Scores:
        """Convert the data into a flattened dict as the summary.

        This function is different to the `.dict()` function.
        As a comparison, `.dict()` will export all data fields as a nested
        dict, While `.summary()` only exports most important information,
        like the overall scores, as a flattened compact dict.

        Args:
            include (set[str]): Optional, the metrics to convert
            exclude (set[str]): Optional, the metrics not to convert
        Returns:
            dict[str, int | float]: returned summary of the result
        """
        summary_dict: Dict[str, Union[int, float]] = {}
        for metric, scores_list in self.dict(
            include=include, exclude=exclude  # type: ignore
        ).items():
            if not isinstance(scores_list, list):
                summary_dict[metric] = scores_list
            elif metric =='TETA':
                summary_dict[metric] = scores_list[-1].get(
                    OVERALL)
                for key in scores_list[0]:
                    summary_dict[metric +'-'+key] = scores_list[0][key]

            elif metric == 'HOTA':
                summary_dict[metric] = scores_list[-1].get(
                    OVERALL)
                for key in scores_list[0]:
                    summary_dict[metric +'-'+key] = scores_list[0][key]

            elif metric == 'MOTA':
                summary_dict[metric] = scores_list[-1].get(
                    OVERALL)
                for key in scores_list[0]:
                    summary_dict[metric + '-' + key] = scores_list[0][key]

            elif metric == 'IDF1':
                summary_dict[metric] = scores_list[-1].get(
                    OVERALL)
                for key in scores_list[0]:
                    summary_dict[metric + '-' + key] = scores_list[0][key]

            else:
                summary_dict[metric] = scores_list[-1].get(
                    OVERALL, scores_list[-1].get(AVERAGE)
                )
        for metric in summary_dict:
            if math.isnan(summary_dict[metric]):
                summary_dict[metric] = '-'
        return summary_dict

def deal_bdd100k_category(
    label: Label, cat_name2id: Dict[str, int]
) -> Optional[Label]:
    """Deal with BDD100K category."""
    category_name = label.category
    if category_name in NAME_MAPPING:
        category_name = NAME_MAPPING[category_name]

    if category_name not in cat_name2id:
        assert category_name in IGNORE_MAPPING
        category_name = IGNORE_MAPPING[category_name]
        if label.attributes is None:
            label.attributes = {}
        label.attributes["ignored"] = True
        label.category = category_name
        result = label
    else:
        label.category = category_name
        result = label
    return result


def bdd100k_to_scalabel(frames: List[Frame], config: Config) -> List[Frame]:
    """Converting BDD100K to Scalabel format."""
    categories = get_leaf_categories(config.categories)
    cat_name2id = {cat.name: i + 1 for i, cat in enumerate(categories)}
    for image_anns in frames:
        if image_anns.labels is not None:
            for i in reversed(range(len(image_anns.labels))):
                label = deal_bdd100k_category(
                    image_anns.labels[i], cat_name2id
                )
                if label is None:
                    image_anns.labels.pop(i)

    return frames


def evaluate_box_track(
    gts: List[List[Frame]],
    results: List[List[Frame]],
    config: Config,
    iou_thr: float = 0.5,
    ignore_iof_thr: float = 0.5,
    ignore_unknown_cats: bool = False,
    nproc: int = NPROC,
) -> BoxTrackResult:
    """Evaluate track metrics for a Scalabel format dataset.

    Args:
        gts: the ground truth annotations in Scalabel format
        results: the prediction results in Scalabel format.
        config: Config object
        iou_thr: Minimum IoU for a bounding box to be considered a positive.
        ignore_iof_thr: Min. Intersection over foreground with ignore regions.
        ignore_unknown_cats: if False, raise KeyError when trying to evaluate
            unknown categories.
        nproc: processes number for loading files

    Returns:
        BoxTrackResult: evaluation results.
    """
    logger.info("Tracking evaluation.")
    t = time.time()
    gts = [bdd100k_to_scalabel(gt, config) for gt in gts]
    results = [bdd100k_to_scalabel(result, config) for result in results]
    hota_result = evaluate_track_hota(gts, results, config, nproc)
    teta_result = evaluate_track_teta(gts, results, config, nproc)
    mot_result = evaluate_track(
        acc_single_video_mot,
        gts,
        results,
        config,
        iou_thr,
        ignore_iof_thr,
        ignore_unknown_cats,
        nproc,
    )
    result = BoxTrackResult(**{**mot_result.dict(), **hota_result.dict(), **teta_result.dict()})
    t = time.time() - t
    logger.info("evaluation finishes with %.1f s.", t)
    return result


def parse_arguments() -> argparse.Namespace:
    """Parse the arguments."""
    parser = argparse.ArgumentParser(description="Box track evaluation.")
    parser.add_argument(
        "--gt", "-g", required=True, help="path to ground truth"
    )
    parser.add_argument(
        "--result", "-r", required=True, help="path to results"
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
        "--out-file", default="", help="Output path for evaluation results."
    )
    parser.add_argument(
        "--iou-thr", type=float, default=0.5, help="iou threshold"
    )
    parser.add_argument(
        "--ignore-iof-thr",
        type=float,
        default=0.5,
        help="ignore iof threshold",
    )
    parser.add_argument(
        "--ignore-unknown-cats",
        type=bool,
        default=False,
        help="ignore unknown categories",
    )
    parser.add_argument(
        "--nproc", "-p", type=int, default=NPROC, help="number of processes"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    dataset = load(args.gt)
    gt_frames, cfg = dataset.frames, dataset.config
    if args.config is not None:
        cfg = load_label_config(args.config)
    if cfg is None:
        raise ValueError(
            "Dataset config is not specified. Please use --config"
            " to specify a config for this dataset."
        )
    eval_result = evaluate_box_track(
        group_and_sort(gt_frames),
        group_and_sort(load(args.result).frames),
        cfg,
        args.iou_thr,
        args.ignore_iof_thr,
        args.ignore_unknown_cats,
        args.nproc,
    )
    logger.info(eval_result)
    logger.info(eval_result.summary())
    if args.out_file:
        with open_write_text(args.out_file) as fp:
            json.dump(eval_result.dict(), fp, indent=2)
