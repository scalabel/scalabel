"""Detection evaluation code.

The prediction and ground truth are expected in scalabel format. The evaluation
results are from the COCO toolkit.
"""
import argparse
import datetime
import json
import os
from typing import Dict, List, Optional

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval  # type: ignore
from tabulate import tabulate

from ..common.typing import DictStrAny, ListAny
from ..label.coco_typing import GtType
from ..label.io import DEFAULT_LABEL_CONFIG, load, load_label_config
from ..label.to_coco import scalabel2coco_detection
from ..label.typing import Config, Frame


class COCOV2(COCO):  # type: ignore
    """Modify the COCO API to support annotations dictionary as input."""

    def __init__(
        self,
        annotation_file: Optional[str] = None,
        annotations: Optional[GtType] = None,
    ) -> None:
        """Init."""
        super().__init__(annotation_file)
        # initialize the annotations in COCO format without saving as json.

        if annotation_file is None:
            print("using the loaded annotations")
            assert isinstance(
                annotations, dict
            ), "annotation file format {} not supported".format(
                type(annotations)
            )
            self.dataset = annotations
            self.createIndex()


def evaluate_det(
    ann_frames: List[Frame],
    pred_frames: List[Frame],
    metadata_cfg: Config,
    out_dir: str = "none",
) -> Dict[str, float]:
    """Load the ground truth and prediction results.

    Args:
        ann_frames: the ground truth annotations in Scalabel format
        pred_frames: the prediction results in Scalabel format.
        metadata_cfg: Metadata config.
        out_dir: output_directory

    Returns:
        dict: detection metric scores

    Example usage:
        evaluate_det(
            "/path/to/gts",
            "/path/to/results",
            "/path/to/cfg",
            nproc=4,
        )
    """
    # Convert the annotation file to COCO format
    ann_frames = sorted(ann_frames, key=lambda frame: frame.name)
    ann_coco = scalabel2coco_detection(ann_frames, metadata_cfg)
    coco_gt = COCOV2(None, ann_coco)

    # Load results and convert the predictions
    pred_frames = sorted(pred_frames, key=lambda frame: frame.name)
    pred_res = scalabel2coco_detection(pred_frames, metadata_cfg)[
        "annotations"
    ]
    coco_dt = coco_gt.loadRes(pred_res)

    cat_ids = coco_dt.getCatIds()
    cat_names = [cat["name"] for cat in coco_dt.loadCats(cat_ids)]

    img_ids = sorted(coco_gt.getImgIds())
    ann_type = "bbox"
    coco_eval = COCOeval(coco_gt, coco_dt, ann_type)
    coco_eval.params.imgIds = img_ids

    return evaluate_workflow(coco_eval, cat_ids, cat_names, out_dir)


def evaluate_workflow(
    coco_eval: COCOeval,
    cat_ids: List[int],
    cat_names: List[str],
    out_dir: str = "none",
) -> Dict[str, float]:
    """Execute evaluation."""
    n_tit = 12  # number of evaluation titles
    n_cls = len(cat_ids)  # 10/8 classes for BDD100K detection/tracking
    n_thr = 10  # [.5:.05:.95] T=10 IoU thresholds for evaluation
    n_rec = 101  # [0:.01:1] R=101 recall thresholds for evaluation
    n_area = 4  # A=4 object area ranges for evaluation
    n_mdet = 3  # [1 10 100] M=3 thresholds on max detections per image

    eval_param = {
        "params": {
            "imgIds": [],
            "catIds": [],
            "iouThrs": np.linspace(
                0.5,
                0.95,
                int(np.round((0.95 - 0.5) / 0.05) + 1),
                endpoint=True,
            ).tolist(),
            "recThrs": np.linspace(
                0.0,
                1.00,
                int(np.round((1.00 - 0.0) / 0.01) + 1),
                endpoint=True,
            ).tolist(),
            "maxDets": [1, 10, 100],
            "areaRng": [
                [0 ** 2, 1e5 ** 2],
                [0 ** 2, 32 ** 2],
                [32 ** 2, 96 ** 2],
                [96 ** 2, 1e5 ** 2],
            ],
            "useSegm": 0,
            "useCats": 1,
        },
        "date": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[
            :-3
        ],
        "counts": [n_thr, n_rec, n_cls, n_area, n_mdet],
        "precision": -np.ones(
            (n_thr, n_rec, n_cls, n_area, n_mdet), order="F"
        ),
        "recall": -np.ones((n_thr, n_cls, n_area, n_mdet), order="F"),
    }
    stats_all = -np.ones((n_cls, n_tit))

    for i, (cat_id, cat_name) in enumerate(zip(cat_ids, cat_names)):
        print("\nEvaluate category: %s" % cat_name)
        coco_eval.params.catIds = [cat_id]
        # coco_eval.params.useSegm = ann_type == "segm"
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats_all[i, :] = coco_eval.stats
        eval_param["precision"][:, :, i, :, :] = coco_eval.eval[
            "precision"
        ].reshape((n_thr, n_rec, n_area, n_mdet))
        eval_param["recall"][:, i, :, :] = coco_eval.eval["recall"].reshape(
            (n_thr, n_area, n_mdet)
        )

    # Print evaluation results
    stats = np.zeros((n_tit, 1))
    print("\nOverall performance")
    coco_eval.eval = eval_param
    coco_eval.summarize()

    for i in range(n_tit):
        column = stats_all[:, i]
        if len(column > -1) == 0:
            stats[i] = -1
        else:
            stats[i] = np.mean(column[column > -1], axis=0)

    score_titles = [
        "AP",
        "AP_50",
        "AP_75",
        "AP_small",
        "AP_medium",
        "AP_large",
        "AR_max_1",
        "AR_max_10",
        "AR_max_100",
        "AR_small",
        "AR_medium",
        "AR_large",
    ]
    scores: Dict[str, float] = {}

    for title, stat in zip(score_titles, stats):
        scores[title] = stat.item()

    if out_dir != "none":
        write_eval(out_dir, scores, eval_param)
    return scores


def write_eval(
    out_dir: str, scores: Dict[str, float], eval_param: DictStrAny
) -> None:
    """Write the evaluation results to file, print in tabulate format."""
    output_filename = os.path.join(out_dir, "scores.json")
    with open(output_filename, "w") as fp:
        json.dump(scores, fp)

    # print the overall performance in the tabulate format
    print(create_small_table(scores))

    eval_param["precision"] = eval_param["precision"].flatten().tolist()
    eval_param["recall"] = eval_param["recall"].flatten().tolist()

    with open(os.path.join(out_dir, "eval.json"), "w") as fp:
        json.dump(eval_param, fp)


def create_small_table(small_dict: Dict[str, float]) -> str:
    """Create a small table using the keys of small_dict as headers.

    Args:
        small_dict (dict): a result dictionary of only a few items.

    Returns:
        str: the table as a string.
    """
    keys, values_t = tuple(zip(*small_dict.items()))
    values = ["{:.1f}".format(val * 100) for val in values_t]
    stride = 3
    items: ListAny = []
    for i in range(0, len(keys), stride):
        items.append(keys[i : min(i + stride, len(keys))])
        items.append(values[i : min(i + stride, len(keys))])
    table = tabulate(
        items[1:],
        headers=items[0],
        tablefmt="fancy_grid",
        floatfmt=".3f",
        stralign="center",
        numalign="center",
    )
    return table


def parse_arguments() -> argparse.Namespace:
    """Parse the arguments."""
    parser = argparse.ArgumentParser(description="Detection evaluation.")
    parser.add_argument(
        "--gt", "-g", required=True, help="path to detection ground truth"
    )
    parser.add_argument(
        "--result", "-r", required=True, help="path to detection results"
    )
    parser.add_argument(
        "--cfg-path",
        "-c",
        default=DEFAULT_LABEL_CONFIG,
        help="Config path, contains metadata like available categories.",
    )
    parser.add_argument(
        "--out-dir",
        "-o",
        default="none",
        help="Output path for detection evaluation results.",
    )
    parser.add_argument(
        "--nproc",
        "-p",
        type=int,
        default=4,
        help="number of processes for detection evaluation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    dataset = load(args.gt, args.nproc)
    gts, config = dataset.frames, dataset.config
    preds = load(args.result).frames
    if args.cfg_path is not None:
        config = load_label_config(args.cfg_path)
    assert config is not None
    evaluate_det(gts, preds, config, args.out_dir)
