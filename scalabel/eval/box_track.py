"""Multi-object Tracking evaluation code."""
import argparse
import json
import time
from typing import Dict, List

from ..common.io import open_write_text
from ..common.logger import logger
from ..common.parallel import NPROC
from ..label.io import group_and_sort, load, load_label_config
from ..label.transforms import box2d_to_bbox
from ..label.typing import Config, Frame
from .hota import evaluate_track_hota
from .mot import acc_single_video_mot, evaluate_track
from .result import Result


EXCLUDE_METRICS = ["Dets", "IDs", "FP", "FN", "MT", "PT", "ML", "FM"]


class BoxTrackResult(Result):
    """The class for bounding box tracking evaluation results."""

    mHOTA: float
    mMOTA: float
    mIDF1: float
    mDetA: float
    mAssA: float
    mMOTP: float
    HOTA: List[Dict[str, float]]
    MOTA: List[Dict[str, float]]
    IDF1: List[Dict[str, float]]
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
    hota_result = evaluate_track_hota(gts, results, config, nproc)
    result = BoxTrackResult(**{**mot_result.dict(), **hota_result.dict()})
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
        action="store_true",
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
