"""Pose estimation evaluation code.

The prediction and ground truth are expected in scalabel format. The evaluation
results are from the COCO toolkit.
"""
import argparse
import copy
import json
from functools import partial
from multiprocessing import Pool
from typing import Callable, Dict, List, Optional

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval, Params  # type: ignore

from ..common.io import open_write_text
from ..common.logger import logger
from ..common.parallel import NPROC
from ..common.typing import DictStrAny, NDArrayF64
from ..label.coco_typing import GtType
from ..label.io import load, load_label_config
from ..label.to_coco import scalabel2coco_pose
from ..label.typing import Config, Frame
from .result import OVERALL, Result
from .utils import reorder_preds


class PoseResult(Result):
    """The class for pose estimation evaluation results."""

    AP: List[Dict[str, float]]
    AP50: List[Dict[str, float]]
    AP75: List[Dict[str, float]]
    APm: List[Dict[str, float]]
    APl: List[Dict[str, float]]
    AR: List[Dict[str, float]]
    AR50: List[Dict[str, float]]
    AR75: List[Dict[str, float]]
    ARm: List[Dict[str, float]]
    ARl: List[Dict[str, float]]

    # pylint: disable=useless-super-delegation
    def __eq__(self, other: "Result") -> bool:  # type: ignore
        """Check whether two instances are equal."""
        return super().__eq__(other)

    @classmethod
    def empty(cls, coco_gt: "COCOV2") -> "PoseResult":
        """Return empty results."""
        metrics = cls.__fields__.keys()
        cat_ids_in = set(ann["category_id"] for ann in coco_gt.anns.values())
        empty_scores = {
            metric: [
                {OVERALL: 0.0 if len(cat_ids_in) > 0 else np.nan},
            ]
            for metric in metrics
        }
        return cls(**empty_scores)


class ParamsV2(Params):  # type: ignore
    """Modify COCO API params to set the keypoint OKS sigmas."""

    def __init__(
        self, iouType: str = "keypoints", sigmas: Optional[List[float]] = None
    ):
        """Init."""
        super().__init__(iouType)
        self.maxDets = [20]
        if sigmas is not None:
            self.kpt_oks_sigmas: NDArrayF64 = np.array(
                sigmas, dtype=np.float64
            )


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
            assert isinstance(
                annotations, dict
            ), f"annotation file format {type(annotations)} not supported"
            self.dataset = annotations
            self.createIndex()


class COCOevalV2(COCOeval):  # type: ignore
    """Modify the COCOeval API to speedup and suppress the printing."""

    def __init__(
        self,
        cat_names: List[str],
        cocoGt: Optional[COCO] = None,
        cocoDt: Optional[COCO] = None,
        iouType: str = "keypoints",
        sigmas: Optional[List[float]] = None,
        nproc: int = NPROC,
    ):
        """Init."""
        super().__init__(cocoGt=cocoGt, cocoDt=cocoDt, iouType=iouType)
        cat_ids = self.params.catIds  # type: ignore
        self.params = ParamsV2(iouType, sigmas)
        self.params.catIds = cat_ids
        self.cat_names = cat_names
        self.nproc = nproc

        self.get_score_funcs: Dict[
            str, Callable[[Optional[int]], float]
        ] = dict(
            AP=self.get_score,
            AP50=partial(self.get_score, iou_thr=0.5),
            AP75=partial(self.get_score, iou_thr=0.75),
            APm=partial(self.get_score, area_rng="medium"),
            APl=partial(self.get_score, area_rng="large"),
            AR=partial(self.get_score, metric="recall"),
            AR50=partial(self.get_score, metric="recall", iou_thr=0.5),
            AR75=partial(self.get_score, metric="recall", iou_thr=0.75),
            ARm=partial(self.get_score, metric="recall", area_rng="medium"),
            ARl=partial(self.get_score, metric="recall", area_rng="large"),
        )

        # useSegm is deprecated
        assert self.params.useSegm is None

    def evaluate(self) -> None:
        """Run per image evaluation on given images."""
        p = self.params
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params = p

        self._prepare()
        # loop through images, area range, max detection number
        cat_ids = p.catIds if p.useCats else [-1]

        self.ious = {
            (imgId, catId): self.computeOks(imgId, catId)
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
        self.evalImgs: List[DictStrAny] = [{} for _ in range(eval_num)]
        for to_update in to_updates:
            for ind, item in to_update.items():
                self.evalImgs[ind] = item

        self._paramsEval = copy.deepcopy(self.params)

    def compute_match(self, img_ind: int) -> Dict[int, DictStrAny]:
        """Compute matching results for each image."""
        p = self.params
        area_num = len(p.areaRng)
        img_num = len(p.imgIds)

        to_updates: Dict[int, DictStrAny] = {}
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
        cat_id: Optional[int] = None,
        metric: str = "precision",
        iou_thr: Optional[float] = None,
        area_rng: str = "all",
        max_dets: int = 20,
    ) -> float:
        """Extract the score according the metric and category."""
        p = self.params
        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == area_rng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == max_dets]
        s = self.eval[metric]
        cat_ids: NDArrayF64 = np.array(p.catIds, dtype=np.int64)
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

    def summarize(self) -> PoseResult:
        """Compute summary metrics for evaluation results."""
        res_dict = {
            metric: [{OVERALL: get_score_func(None)}]
            for metric, get_score_func in self.get_score_funcs.items()
        }
        return PoseResult(**res_dict)


def evaluate_pose(
    ann_frames: List[Frame],
    pred_frames: List[Frame],
    config: Config,
    nproc: int = NPROC,
) -> PoseResult:
    """Evaluate pose estimation with Scalabel format.

    Args:
        ann_frames: the ground truth frames.
        pred_frames: the prediction frames.
        config: Metadata config.
        nproc: the number of process.

    Returns:
        PoseResult: evaluation results.
    """
    # Convert the annotation file to COCO format
    ann_frames = sorted(ann_frames, key=lambda frame: frame.name)
    ann_coco = scalabel2coco_pose(ann_frames, config)
    coco_gt = COCOV2(None, ann_coco)

    # Load results and convert the predictions
    pred_frames = reorder_preds(ann_frames, pred_frames)
    pred_res = scalabel2coco_pose(pred_frames, config)["annotations"]
    if not pred_res:
        return PoseResult.empty(coco_gt)
    coco_dt = coco_gt.loadRes(pred_res)

    cat_ids = coco_dt.getCatIds()
    cat_names = [cat["name"] for cat in coco_dt.loadCats(cat_ids)]

    img_ids = sorted(coco_gt.getImgIds())
    ann_type = "keypoints"
    coco_eval = COCOevalV2(
        cat_names, coco_gt, coco_dt, ann_type, config.poseSigmas, nproc
    )
    coco_eval.params.imgIds = img_ids

    logger.info("evaluating...")
    coco_eval.evaluate()
    logger.info("accumulating...")
    coco_eval.accumulate()
    result = coco_eval.summarize()
    return result


def parse_arguments() -> argparse.Namespace:
    """Parse the arguments."""
    parser = argparse.ArgumentParser(description="Pose evaluation.")
    parser.add_argument(
        "--gt", "-g", required=True, help="path to pose ground truth"
    )
    parser.add_argument(
        "--result", "-r", required=True, help="path to pose results"
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
        help="Output file for pose evaluation results.",
    )
    parser.add_argument(
        "--nproc",
        "-p",
        type=int,
        default=NPROC,
        help="number of processes for pose evaluation",
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
    eval_result = evaluate_pose(gts, preds, cfg, args.nproc)
    logger.info(eval_result)
    logger.info(eval_result.summary())
    if args.out_file:
        with open_write_text(args.out_file) as fp:
            json.dump(eval_result.dict(), fp, indent=2)
