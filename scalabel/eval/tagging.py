"""Evaluation procedures for image tagging."""
import argparse
import json
from typing import Dict, List

import numpy as np
from sklearn.metrics import classification_report

from .result import Result
from ..common.io import open_write_text
from ..common.logger import logger
from ..common.parallel import NPROC
from ..label.io import load, load_label_config
from ..label.typing import Config, Frame


class TaggingResult(Result):
    """The class for general image tagging evaluation results."""

    precision: List[Dict[str, float]]
    recall: List[Dict[str, float]]
    f1_score: List[Dict[str, float]]
    accuracy: List[Dict[str, float]]

    # pylint: disable=useless-super-delegation
    def __eq__(self, other: "TaggingResult") -> bool:  # type: ignore
        """Check whether two instances are equal."""
        return super().__eq__(other)


def evaluate_tagging(
    ann_frames: List[Frame],
    pred_frames: List[Frame],
    config: Config,
    tag_attr: str,
    nproc: int = NPROC,  # pylint: disable=unused-argument
) -> TaggingResult:
    """Evaluate image tagging with Scalabel format.

    Args:
        ann_frames: the ground truth frames.
        pred_frames: the prediction frames.
        config: Metadata config.
        tag_attr: image attribute to evaluate.
        nproc: the number of process.

    Returns:
        TaggingResult: evaluation results.
    """
    classes = [cat.name for cat in config.categories]
    preds_cls, gts_cls = [], []
    for p, g in zip(pred_frames, ann_frames):
        if g.attributes is None:
            continue
        assert p.attributes is not None
        p.attributes = {tag_attr: p.attributes[tag_attr]}
        p_attr, g_attr = p.attributes[tag_attr], g.attributes[tag_attr]
        assert isinstance(p_attr, str) and isinstance(g_attr, str)
        preds_cls.append(classes.index(p_attr))
        gts_cls.append(classes.index(g_attr))
    parray, garray = np.array(preds_cls), np.array(gts_cls)
    gt_classes = [classes[cid] for cid in sorted(set(gts_cls + preds_cls))]
    scores = classification_report(
        garray, parray, target_names=gt_classes, output_dict=True
    )
    output = {
        metric: [
            {
                cat: scores[cat][metric] * 100.0 if cat in scores else np.nan
                for cat in classes
            },
            {
                "AVERAGE": scores["macro avg"][metric] * 100.0
                if len(scores) > 3
                else np.nan
            },
        ]
        for metric in ["precision", "recall", "f1-score"]
    }
    output["accuracy"] = [
        {cat: np.nan for cat in classes},
        {"AVERAGE": scores["accuracy"] * 100.0},
    ]
    output["f1_score"] = output.pop("f1-score")
    return TaggingResult(**output)


def parse_arguments() -> argparse.Namespace:
    """Parse the arguments."""
    parser = argparse.ArgumentParser(description="Tagging evaluation.")
    parser.add_argument(
        "--gt", "-g", required=True, help="path to detection ground truth"
    )
    parser.add_argument(
        "--result", "-r", required=True, help="path to detection results"
    )
    parser.add_argument(
        "--tag-attr", required=True, help="tagging attribute to evaluate"
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
        help="Output file for detection evaluation results.",
    )
    parser.add_argument(
        "--nproc",
        "-p",
        type=int,
        default=NPROC,
        help="number of processes for detection evaluation",
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
    eval_result = evaluate_tagging(gts, preds, cfg, args.tag_attr, args.nproc)
    logger.info(eval_result)
    logger.info(eval_result.summary())
    if args.out_file:
        with open_write_text(args.out_file) as fp:
            json.dump(eval_result.json(), fp)
