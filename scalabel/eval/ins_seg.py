"""Instance segmentation evaluation code.

The prediction and ground truth are expected in scalabel format. The evaluation
results are from the COCO toolkit.
"""
import json
from typing import List

from scalabel.common.parallel import NPROC
from scalabel.eval.detect import COCOV2, COCOevalV2, DetResult, parse_arguments
from scalabel.label.to_coco import scalabel2coco_ins_seg
from scalabel.label.typing import Config

from ..common.io import open_write_text
from ..common.logger import logger
from ..label.io import load, load_label_config
from ..label.typing import Config, Frame
from .utils import check_overlap, reorder_preds


def evaluate_ins_seg(
    ann_frames: List[Frame],
    pred_frames: List[Frame],
    config: Config,
    nproc: int = NPROC,
) -> DetResult:
    """Evaluate instance segmentation with Scalabel format.

    Args:
        ann_frames: the ground truth frames.
        pred_frames: the prediction frames.
        config: Metadata config.
        nproc: the number of process.

    Returns:
        DetResult: evaluation results.
    """
    # convert the annotation file to COCO format
    ann_frames = sorted(ann_frames, key=lambda frame: frame.name)
    ann_coco = scalabel2coco_ins_seg(ann_frames, config)
    ann_coco["annotations"] = [
        ann for ann in ann_coco["annotations"] if "segmentation" in ann
    ]
    coco_gt = COCOV2(None, ann_coco)

    # load results and convert the predictions
    pred_frames = reorder_preds(ann_frames, pred_frames)
    # check overlap of masks
    logger.info("checking for overlap of masks...")
    if check_overlap(pred_frames, config, nproc):
        logger.critical(
            "Found overlap in prediction bitmasks, but instance segmentation "
            "evaluation does not allow overlaps. Removing such predictions."
        )
    logger.info("converting predictions to COCO format...")
    pred_res = scalabel2coco_ins_seg(pred_frames, config)["annotations"]
    # handle empty predictions
    if not pred_res:
        logger.critical("No predictions found. Skipping evaluation.")
        return DetResult.empty(coco_gt)
    # removing bbox so pycocotools will use mask to compute area
    for ann in pred_res:
        if "bbox" in ann:
            ann.pop("bbox")
    coco_dt = coco_gt.loadRes(pred_res)

    cat_ids = coco_dt.getCatIds()
    cat_names = [cat["name"] for cat in coco_dt.loadCats(cat_ids)]

    img_ids = sorted(coco_gt.getImgIds())
    ann_type = "segm"
    coco_eval = COCOevalV2(cat_names, coco_gt, coco_dt, ann_type, nproc)
    coco_eval.params.imgIds = img_ids

    logger.info("evaluating...")
    coco_eval.evaluate()
    logger.info("accumulating...")
    coco_eval.accumulate()
    result = coco_eval.summarize()
    return result


if __name__ == "__main__":
    args = parse_arguments()
    dataset = load(args.gt, args.nproc)
    gts, cfg = dataset.frames, dataset.config
    preds = load(args.result).frames
    if args.config is not None:
        cfg = load_label_config(args.config)
    assert cfg is not None
    eval_result = evaluate_ins_seg(gts, preds, cfg, args.nproc)
    logger.info(eval_result)
    logger.info(eval_result.summary())
    if args.out_file:
        with open_write_text(args.out_file) as fp:
            json.dump(eval_result.dict(), fp, indent=2)
