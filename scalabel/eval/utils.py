"""Utility functions for eval."""
import copy
from functools import partial
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

from scalabel.common.logger import logger
from scalabel.common.parallel import NPROC
from scalabel.common.typing import NDArrayI32, NDArrayU8
from scalabel.label.transforms import mask_to_rle, poly2ds_to_mask, rle_to_mask
from scalabel.label.typing import (
    RLE,
    Category,
    Config,
    Frame,
    ImageSize,
    Label,
)
from scalabel.label.utils import (
    check_crowd,
    check_ignored,
    get_leaf_categories,
)

RLEDict = Dict[str, Union[str, Tuple[int, int]]]


def label_ids_to_int(frames: List[Frame]) -> None:
    """Converts any type of label index to a string representing an integer."""
    assert len(frames) > 0
    # if label ids are strings not representing integers, convert them
    ids = []
    for frame in frames:
        if frame.labels is not None:
            ids.extend([l.id for l in frame.labels])

    if any(not id.isdigit() for id in ids):
        ids_to_int = {y: x + 1 for x, y in enumerate(sorted(set(ids)))}

        for frame in frames:
            if frame.labels is not None:
                for label in frame.labels:
                    label.id = str(ids_to_int[label.id])


def check_overlap_frame(
    frame: Frame, categories: List[str], image_size: Optional[ImageSize] = None
) -> bool:
    """Check overlap of segmentation masks for a single frame."""
    if frame.labels is None:
        return False
    overlap_mask: NDArrayU8 = np.zeros((0), dtype=np.uint8)
    for label in frame.labels:
        if label.category not in categories:
            continue
        if label.rle is None and label.poly2d is None:
            continue
        if label.rle is not None:
            mask = rle_to_mask(label.rle)
        elif label.poly2d is not None:
            assert (
                image_size is not None
            ), "Requires ImageSize for Poly2D conversion to RLE"
            mask = poly2ds_to_mask(image_size, label.poly2d)
        if len(overlap_mask) == 0:
            overlap_mask = mask
        else:
            if np.logical_and(overlap_mask, mask).any():
                # found overlap
                return True
            overlap_mask += mask
    return False


def check_overlap(
    frames: List[Frame], config: Config, nproc: int = NPROC
) -> bool:
    """Check overlap of segmentation masks.

    Returns True if overlap found in masks of any frame.
    """
    categories = get_leaf_categories(config.categories)
    category_names = [category.name for category in categories]
    if nproc > 1:
        with Pool(nproc) as pool:
            overlaps = pool.map(
                partial(
                    check_overlap_frame,
                    categories=category_names,
                    image_size=config.imageSize,
                ),
                tqdm(frames),
            )
    else:
        overlaps = [
            check_overlap_frame(frame, category_names, config.imageSize)
            for frame in tqdm(frames)
        ]
    for is_overlap, frame in zip(overlaps, frames):
        if is_overlap:
            # remove predictions with overlap
            frame.labels = None
    return any(overlaps)


def handle_inconsistent_length(
    gts: List[List[Frame]],
    results: List[List[Frame]],
) -> List[List[Frame]]:
    """Check the video results and give feedbacks about inconsistency.

    Args:
        gts: the ground truth annotations in Scalabel format
        results: the prediction results in Scalabel format.

    Returns:
      results: the processed results in Scalabel format.

    Raises:
        ValueError: if test results have videos that not present in gts.
    """
    # check the missing video sequences
    if len(results) < len(gts):
        # build gt_index and gt_videoName mapping
        gt_video_name_index_mapping = {}
        for i, gt in enumerate(gts):
            gt_video_name_index_mapping[gt[0].videoName] = i

        gt_video_names = set(gt_video_name_index_mapping.keys())
        res_video_names = {res[0].videoName for res in results}
        missing_video_names = gt_video_names - res_video_names
        outlier_results = res_video_names - gt_video_names

        if outlier_results:
            logger.critical(
                "You have videos not in the test set: " "%s",
                str(outlier_results),
            )
            raise ValueError(
                f"You have videos not in the test set: "
                f"{str(outlier_results)}"
            )

        if missing_video_names:
            logger.critical(
                "The results are missing for "
                "following video sequences: "
                "%s",
                str(missing_video_names),
            )

            # add empty list for those missing results
            for name in missing_video_names:
                gt_idx = gt_video_name_index_mapping[name]
                gt = gts[gt_idx]
                gt_without_labels = []

                for f in gt:
                    f_copy = copy.deepcopy(f)
                    f_copy.labels = []
                    gt_without_labels.append(f_copy)

                results.append(gt_without_labels)

            results = sorted(
                results, key=lambda frames: str(frames[0].videoName)
            )
    elif len(results) > len(gts):
        raise ValueError("You have videos not in the test set.")

    return results


def combine_stuff_masks(
    rles: List[RLE],
    class_ids: List[int],
    inst_ids: List[int],
    classes: List[Category],
) -> Tuple[List[RLE], List[int], List[int]]:
    """For each stuff class, combine masks into a single mask."""
    combine_rles: List[RLE] = []
    combine_cids: List[int] = []
    combine_iids: List[int] = []
    for class_id in sorted(set(class_ids)):
        category = classes[class_id]
        rles_c = [
            rle for rle, c_id in zip(rles, class_ids) if c_id == class_id
        ]
        iids_c = [
            iid for iid, c_id in zip(inst_ids, class_ids) if c_id == class_id
        ]
        if category.isThing is None or category.isThing:
            combine_rles.extend(rles_c)
            combine_cids.extend([class_id] * len(rles_c))
            combine_iids.extend(iids_c)
        else:
            combine_mask: NDArrayU8 = sum(  # type: ignore
                rle_to_mask(rle) for rle in rles_c
            )
            combine_rles.append(mask_to_rle(combine_mask))
            combine_cids.append(class_id)
            combine_iids.append(iids_c[0])
    return combine_rles, combine_cids, combine_iids


def parse_seg_objects(
    objects: List[Label],
    classes: List[Category],
    ignore_unknown_cats: bool = False,
    image_size: Optional[ImageSize] = None,
) -> Tuple[List[RLEDict], NDArrayI32, NDArrayI32, List[RLEDict]]:
    """Parse segmentation objects under Scalabel formats."""
    rles, labels, ids, ignore_rles = [], [], [], []
    class_names = [category.name for category in classes]
    for obj in objects:
        if obj.rle is not None:
            rle = obj.rle
        elif obj.poly2d is not None:
            assert (
                image_size is not None
            ), "Requires ImageSize for Poly2D conversion to RLE"
            rle = mask_to_rle(poly2ds_to_mask(image_size, obj.poly2d))
        else:
            continue
        category = obj.category
        if category in class_names:
            if check_crowd(obj) or check_ignored(obj):
                ignore_rles.append(rle)
            else:
                rles.append(rle)
                labels.append(class_names.index(category))
                ids.append(int(obj.id))
        else:
            if not ignore_unknown_cats:
                raise KeyError(f"Unknown category: {category}")
    if any(
        category.isThing is not None and not category.isThing
        for category in classes
    ):
        rles, labels, ids = combine_stuff_masks(rles, labels, ids, classes)
    rles_dict = [rle.dict() for rle in rles]
    ignore_rles_dict = [rle.dict() for rle in ignore_rles]
    labels_arr: NDArrayI32 = np.array(labels, dtype=np.int32)
    ids_arr: NDArrayI32 = np.array(ids, dtype=np.int32)
    return (rles_dict, labels_arr, ids_arr, ignore_rles_dict)


def reorder_preds(
    ann_frames: List[Frame], pred_frames: List[Frame]
) -> List[Frame]:
    """Reorder predictions and add empty frames for missing predictions."""
    pred_names = [f.name for f in pred_frames]
    use_video = False
    if len(pred_names) != len(set(pred_names)):
        # handling non-unique prediction frames names with videoName
        use_video = all(f.videoName for f in pred_frames) and all(
            f.videoName for f in ann_frames
        )
        if not use_video:
            logger.critical(
                "Prediction frames names are not unique, but videoName is not "
                "specified for all frames."
            )
    pred_map: Dict[str, Frame] = {}
    for pred_frame in pred_frames:
        name = pred_frame.name
        if use_video:
            name = f"{pred_frame.videoName}/{name}"
        pred_map[name] = pred_frame
    order_results: List[Frame] = []
    miss_num = 0
    for gt_frame in ann_frames:
        gt_name = gt_frame.name
        if use_video:
            gt_name = f"{gt_frame.videoName}/{gt_name}"
        if gt_name in pred_map:
            order_results.append(pred_map[gt_name])
        else:
            # add empty frame
            gt_frame_copy = copy.deepcopy(gt_frame)
            gt_frame_copy.labels = None
            order_results.append(gt_frame_copy)
            miss_num += 1
    if miss_num > 0:
        logger.critical("%s images are missed in the prediction!", miss_num)
    return order_results


def filter_labels(frames: List[Frame], cats: List[Category]) -> List[Frame]:
    """Filter frame labels by categories."""
    cats = get_leaf_categories(cats)
    cat_names = [c.name for c in cats]
    filt_frames = []
    for f in frames:
        if f.labels is not None:
            f.labels = [l for l in f.labels if l.category in cat_names]
        filt_frames.append(f)
    return filt_frames
