"""Evaluation procedures for image tagging."""
import argparse
import json
from typing import AbstractSet, Dict, List, Optional, Union

import numpy as np
from sklearn.metrics import classification_report  # type: ignore

from ..common.io import open_write_text
from ..common.logger import logger
from ..common.parallel import NPROC
from ..common.typing import NDArrayI32
from ..label.io import load, load_label_config
from ..label.typing import Config, Frame
from ..label.utils import get_parent_categories
from .result import AVERAGE, Result, Scores, ScoresList
from .utils import reorder_preds


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

    def summary(
        self,
        include: Optional[AbstractSet[str]] = None,
        exclude: Optional[AbstractSet[str]] = None,
    ) -> Scores:
        """Convert tagging results into a flattened dict as the summary."""
        summary_dict: Dict[str, Union[int, float]] = {}
        for metric, scores_list in self.dict(
            include=include, exclude=exclude  # type: ignore
        ).items():
            for category, score in scores_list[-2].items():
                summary_dict[f"{metric}/{category}"] = score
            summary_dict[metric] = scores_list[-1][AVERAGE]
        return summary_dict


def evaluate_tagging(
    ann_frames: List[Frame],
    pred_frames: List[Frame],
    config: Config,
    nproc: int = NPROC,  # pylint: disable=unused-argument
) -> TaggingResult:
    """Evaluate image tagging with Scalabel format.

    Args:
        ann_frames: the ground truth frames.
        pred_frames: the prediction frames.
        config: Metadata config.
        nproc: the number of process.

    Returns:
        TaggingResult: evaluation results.
    """
    pred_frames = reorder_preds(ann_frames, pred_frames)
    tag_classes = get_parent_categories(config.categories)
    assert tag_classes, "Tag attributes must be specified as supercategories"
    metrics = ["precision", "recall", "f1_score", "accuracy"]
    outputs: Dict[str, ScoresList] = {m: [] for m in metrics}
    avgs: Dict[str, Scores] = {m: {} for m in metrics}
    for tag, class_list in tag_classes.items():
        classes = [c.name for c in class_list]
        preds_cls, gts_cls = [], []
        for p, g in zip(pred_frames, ann_frames):
            if g.attributes is None:
                continue
            assert p.attributes is not None
            p_attr, g_attr = p.attributes[tag], g.attributes[tag]
            assert isinstance(p_attr, str) and isinstance(g_attr, str)
            assert p_attr in classes and g_attr in classes
            preds_cls.append(classes.index(p_attr))
            gts_cls.append(classes.index(g_attr))
        parray: NDArrayI32 = np.array(preds_cls, dtype=np.int32)
        garray: NDArrayI32 = np.array(gts_cls, dtype=np.int32)
        gt_classes = [classes[cid] for cid in sorted(set(gts_cls + preds_cls))]
        scores = classification_report(
            garray, parray, target_names=gt_classes, output_dict=True
        )
        out: Dict[str, Scores] = {}
        for metric in ["precision", "recall", "f1-score"]:
            met = metric if metric != "f1-score" else "f1_score"
            out[met] = {}
            for cat in classes:
                out[met][f"{tag}.{cat}"] = (
                    scores[cat][metric] * 100.0 if cat in scores else np.nan
                )
            avgs[met][tag.upper()] = (
                scores["macro avg"][metric] * 100.0
                if len(scores) > 3
                else np.nan
            )
        out["accuracy"] = {f"{tag}.{cat}": np.nan for cat in classes}
        avgs["accuracy"][tag.upper()] = scores["accuracy"] * 100.0
        for m, v in out.items():
            outputs[m].append(v)
    for m, v in avgs.items():
        outputs[m].append(v)
        outputs[m].append({AVERAGE: np.nanmean(list(v.values()))})
    return TaggingResult(**outputs)


def parse_arguments() -> argparse.Namespace:
    """Parse the arguments."""
    parser = argparse.ArgumentParser(description="Tagging evaluation.")
    parser.add_argument(
        "--gt", "-g", required=True, help="path to tagging ground truth"
    )
    parser.add_argument(
        "--result", "-r", required=True, help="path to tagging results"
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
        help="Output file for tagging evaluation results.",
    )
    parser.add_argument(
        "--nproc",
        "-p",
        type=int,
        default=NPROC,
        help="number of processes for tagging evaluation",
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
    eval_result = evaluate_tagging(gts, preds, cfg, args.nproc)
    logger.info(eval_result)
    logger.info(eval_result.summary())
    if args.out_file:
        with open_write_text(args.out_file) as fp:
            json.dump(eval_result.json(), fp)
