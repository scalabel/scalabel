"""Utility functions for eval."""
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from scalabel.common.logger import logger
from scalabel.common.typing import NDArrayI32, NDArrayU8
from scalabel.label.transforms import mask_to_rle, poly2ds_to_mask, rle_to_mask
from scalabel.label.typing import Category, ImageSize, Label, Frame, RLE
from scalabel.label.utils import check_crowd, check_ignored

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


def combine_stuff_masks(
    rles: List[RLE], class_ids: List[int], classes: List[Category]
) -> Tuple[List[RLE], List[int]]:
    """For each stuff class, combine masks into a single mask."""
    combine_rles: List[RLE] = []
    combine_cids: List[int] = []
    for class_id in sorted(set(class_ids)):
        category = classes[class_id]
        rles_c = [
            rle for rle, c_id in zip(rles, class_ids) if c_id == class_id
        ]
        if category.isThing is None or category.isThing:
            combine_rles.extend(rles_c)
            combine_cids.extend([class_id] * len(rles_c))
        else:
            combine_rle: NDArrayU8 = sum(  # type: ignore
                rle_to_mask(rle) for rle in rles_c
            )
            combine_rles.append(mask_to_rle(combine_rle))
            combine_cids.append(class_id)
    return combine_rles, combine_cids


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
                ids.append(obj.id)
        else:
            if not ignore_unknown_cats:
                raise KeyError(f"Unknown category: {category}")
    if any(
        category.isThing is not None and not category.isThing
        for category in classes
    ):
        rles, labels = combine_stuff_masks(rles, labels, classes)
    rles_dict = [rle.dict() for rle in rles]
    ignore_rles_dict = [rle.dict() for rle in ignore_rles]
    labels_arr = np.array(labels, dtype=np.int32)
    ids_arr = np.array(ids, dtype=np.int32)
    return (rles_dict, labels_arr, ids_arr, ignore_rles_dict)


def reorder_preds(
    ann_frames: List[Frame], pred_frames: List[Frame]
) -> List[Frame]:
    """Sort predictions and add empty frames for missing predictions."""
    pred_map: Dict[str, Frame] = {
        pred_frame.name: pred_frame for pred_frame in pred_frames
    }
    sorted_results: List[Frame] = []
    miss_num = 0
    for gt_frame in ann_frames:
        if gt_frame.name in pred_map:
            sorted_results.append(pred_map[gt_frame.name])
        else:
            sorted_results.append(Frame(name=gt_frame.name))
            miss_num += 1
    logger.info("%s images are missed in the prediction.", miss_num)
    return sorted_results
