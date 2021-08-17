"""Detection evaluation code.

The prediction and ground truth are expected in scalabel format. The evaluation
results are from the COCO toolkit.
"""
import argparse
import copy
import json
import time
from functools import partial
from multiprocessing import Pool
from typing import AbstractSet, Callable, Dict, List, Optional

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval  # type: ignore

from ..common.logger import logger
from ..common.parallel import NPROC
from ..common.typing import DictStrAny
from ..label.coco_typing import GtType
from ..label.io import load, load_label_config
from ..label.to_coco import scalabel2coco_detection
from ..label.typing import Config, Frame
from .result import OVERALL, Result, Scores, ScoresList


class DetResult(Result):
    """The class for bounding box detection evaluation results."""

    AP: List[Dict[str, float]]
    AP50: List[Dict[str, float]]
    AP75: List[Dict[str, float]]
    APs: List[Dict[str, float]]
    APm: List[Dict[str, float]]
    APl: List[Dict[str, float]]
    AR1: List[Dict[str, float]]
    AR10: List[Dict[str, float]]
    AR100: List[Dict[str, float]]
    ARs: List[Dict[str, float]]
    ARm: List[Dict[str, float]]
    ARl: List[Dict[str, float]]

    def __init__(self, **data: ScoresList) -> None:
        """Set extra parameters."""
        super().__init__(**data)
        self._formatters = {
            metric: "{:.1f}".format for metric in self.__fields__
        }

    # pylint: disable=useless-super-delegation
    def __eq__(self, other: "Result") -> bool:  # type: ignore
        """Check whether two instances are equal."""
        return super().__eq__(other)

    def summary(
        self,
        include: Optional[AbstractSet[str]] = None,
        exclude: Optional[AbstractSet[str]] = None,
    ) -> Scores:
        """Convert the data into a flattened dict as the summary."""
        summary_dict = super().summary(include, exclude)
        for scores in self.AP:
            for category, score in scores.items():
                if category == OVERALL:
                    continue
                summary_dict["{}/{}".format("AP", category)] = score
        return summary_dict


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


class COCOevalV2(COCOeval):  # type: ignore
    """Modify the COCOeval API to speedup and suppress the printing."""

    def __init__(
        self,
        cat_names: List[str],
        cocoGt: Optional[COCO] = None,
        cocoDt: Optional[COCO] = None,
        iouType: str = "segm",
        nproc: int = NPROC,
    ):
        """Init."""
        super().__init__(cocoGt=cocoGt, cocoDt=cocoDt, iouType=iouType)
        self.cat_names = cat_names
        self.nproc = nproc

        max_dets = self.params.maxDets  # type: ignore
        self.get_score_funcs: Dict[
            str, Callable[[Optional[int]], float]
        ] = dict(
            AP=self.get_score,
            AP50=partial(
                self.get_score,
                metric="precision",
                iou_thr=0.5,
                max_dets=max_dets[2],
            ),
            AP75=partial(
                self.get_score,
                metric="precision",
                iou_thr=0.75,
                max_dets=max_dets[2],
            ),
            APs=partial(
                self.get_score,
                metric="precision",
                area_rng="small",
                max_dets=max_dets[2],
            ),
            APm=partial(
                self.get_score,
                metric="precision",
                area_rng="medium",
                max_dets=max_dets[2],
            ),
            APl=partial(
                self.get_score,
                metric="precision",
                area_rng="large",
                max_dets=max_dets[2],
            ),
            AR1=partial(self.get_score, metric="recall", max_dets=max_dets[0]),
            AR10=partial(
                self.get_score, metric="recall", max_dets=max_dets[1]
            ),
            AR100=partial(
                self.get_score, metric="recall", max_dets=max_dets[2]
            ),
            ARs=partial(
                self.get_score,
                metric="recall",
                area_rng="small",
                max_dets=max_dets[2],
            ),
            ARm=partial(
                self.get_score,
                metric="recall",
                area_rng="medium",
                max_dets=max_dets[2],
            ),
            ARl=partial(
                self.get_score,
                metric="recall",
                area_rng="large",
                max_dets=max_dets[2],
            ),
        )

    def evaluate(self) -> None:
        """Run per image evaluation on given images."""
        tic = time.time()
        print("Running per image evaluation...")
        p = self.params  # type: ignore
        # add backward compatibility if useSegm is specified in params
        print("Evaluate annotation type *{}*".format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params = p

        self._prepare()
        # loop through images, area range, max detection number
        cat_ids = p.catIds if p.useCats else [-1]

        self.ious = {
            (imgId, catId): self.computeIoU(imgId, catId)
            for imgId in p.imgIds
            for catId in cat_ids
        }

        if self.nproc > 1:
            with Pool(self.nproc) as pool:
                to_updates: List[Dict[int, DictStrAny]] = pool.map(
                    self.compute_match, range(len(p.imgIds))
                )
        else:
            to_updates = list(map(self.compute_match, range(len(p.imgIds))))

        eval_num = len(p.catIds) * len(p.areaRng) * len(p.imgIds)
        self.evalImgs: List[DictStrAny] = [dict() for _ in range(eval_num)]
        for to_update in to_updates:
            for ind, item in to_update.items():
                self.evalImgs[ind] = item

        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print("DONE (t={:0.2f}s).".format(toc - tic))

    def compute_match(self, img_ind: int) -> Dict[int, DictStrAny]:
        """Compute matching results for each image."""
        p = self.params
        area_num = len(p.areaRng)
        img_num = len(p.imgIds)

        to_updates: Dict[int, DictStrAny] = dict()
        for cat_ind, cat_id in enumerate(p.catIds):
            for area_ind, area_rng in enumerate(p.areaRng):
                eval_ind: int = (
                    cat_ind * area_num * img_num + area_ind * img_num + img_ind
                )
                to_updates[eval_ind] = self.evaluateImg(
                    p.imgIds[img_ind], cat_id, area_rng, p.maxDets[-1]
                )
        return to_updates

    def get_score(
        self,
        cat_id: Optional[int],
        metric: str = "precision",
        iou_thr: Optional[float] = None,
        area_rng: str = "all",
        max_dets: int = 100,
    ) -> float:
        """Extract the score according the metric and category."""
        p = self.params
        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == area_rng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == max_dets]
        s = self.eval[metric]
        cat_ids = np.array(p.catIds)
        if iou_thr is not None:
            t = np.where(iou_thr == p.iouThrs)[0]
            s = s[t]
        if metric == "precision":
            # dimension of precision: [TxRxKxAxM]
            if cat_id is not None:
                k = np.where(cat_id == cat_ids)[0]
                s = s[:, :, k, aind, mind]
            else:
                s = s[:, :, :, aind, mind]
        elif metric == "recall":
            # dimension of recall: [TxKxAxM]
            if cat_id is not None:
                k = np.where(cat_id == cat_ids)[0]
                s = s[:, k, aind, mind]
            else:
                s = s[:, :, aind, mind]
        else:
            raise NotImplementedError
        if len(s[s > -1]) == 0:
            mean_s = float("nan")
        else:
            mean_s = np.mean(s[s > -1])
        return mean_s * 100

    def summarize(self) -> DetResult:
        """Compute summary metrics for evaluation results."""
        cat_ids = self.params.catIds + [None]
        res_dict = {
            metric: [
                {
                    cat_name: get_score_func(cat_id)
                    for cat_name, cat_id in zip(self.cat_names, cat_ids)
                },
                {OVERALL: get_score_func(None)},
            ]
            for metric, get_score_func in self.get_score_funcs.items()
        }
        return DetResult(**res_dict)


def evaluate_det(
    ann_frames: List[Frame],
    pred_frames: List[Frame],
    config: Config,
    nproc: int = NPROC,
    with_logs: bool = True,
) -> DetResult:
    """Load the ground truth and prediction results.

    Args:
        ann_frames: the ground truth annotations in Scalabel format
        pred_frames: the prediction results in Scalabel format.
        config: Metadata config.
        nproc: the number of process.
        with_logs: whether to print logs

    Returns:
        DetResult: rendered eval results.

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
    ann_coco = scalabel2coco_detection(ann_frames, config)
    coco_gt = COCOV2(None, ann_coco)

    # Load results and convert the predictions
    pred_frames = sorted(pred_frames, key=lambda frame: frame.name)
    pred_res = scalabel2coco_detection(pred_frames, config)["annotations"]
    coco_dt = coco_gt.loadRes(pred_res)

    cat_ids = coco_dt.getCatIds()
    cat_names = [cat["name"] for cat in coco_dt.loadCats(cat_ids)]

    img_ids = sorted(coco_gt.getImgIds())
    ann_type = "bbox"
    coco_eval = COCOevalV2(cat_names, coco_gt, coco_dt, ann_type, nproc)
    coco_eval.params.imgIds = img_ids

    if with_logs:
        logger.info("evaluating...")
    coco_eval.evaluate()
    if with_logs:
        logger.info("accumulating...")
    coco_eval.accumulate()
    result = coco_eval.summarize()
    return result


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
    parser.add_argument(
        "--quite",
        "-q",
        action="store_true",
        help="without logging",
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
    eval_result = evaluate_det(gts, preds, cfg, args.nproc, not args.quite)
    logger.info(eval_result)
    logger.info(eval_result.summary())
    if args.out_file:
        with open(args.out_file, "w") as fp:
            json.dump(eval_result.json(), fp)
