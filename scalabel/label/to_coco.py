"""Convert Scalabel to COCO format."""

import argparse
import json
import os.path as osp
from functools import partial
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

import numpy as np
from pycocotools import mask as mask_utils  # type: ignore
from tqdm import tqdm

from ..common.io import load_config
from ..common.logger import logger
from .coco_typing import (
    AnnType,
    CatType,
    GtType,
    ImgType,
    PolygonType,
    RLEType,
    VidType,
)
from .io import group_and_sort, load
from .transforms import (
    box2d_to_bbox,
    mask_to_bbox,
    mask_to_polygon,
    poly2ds_to_mask,
)
from .typing import Frame, Label, Poly2D

DEFAULT_COCO_CONFIG = osp.join(
    osp.dirname(osp.abspath(__file__)), "configs.toml"
)


def parse_arguments() -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="Scalabel to COCO format")
    parser.add_argument(
        "-l",
        "--label",
        help=(
            "root directory of Scalabel label Json files or path to a label "
            "json file"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        help="path to save coco formatted label file",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Height of images",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Height of images",
    )
    parser.add_argument(
        "-m",
        "--mode",
        default="det",
        choices=["det", "ins_seg", "box_track", "seg_track"],
        help="conversion mode: detection or tracking.",
    )
    parser.add_argument(
        "-ri",
        "--remove-ignore",
        action="store_true",
        help="Remove the ignored annotations from the label file.",
    )
    parser.add_argument(
        "-ic",
        "--ignore-as-class",
        action="store_true",
        help="Put the ignored annotations to the `ignored` category.",
    )
    parser.add_argument(
        "-mm",
        "--mask-mode",
        default="rle",
        choices=["rle", "polygon"],
        help="conversion mode: rle or polygon.",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=4,
        help="number of processes for conversion",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_COCO_CONFIG,
        help="Configuration for COCO categories",
    )
    return parser.parse_args()


def process_category(
    category_name: str,
    categories: List[CatType],
    name_mapping: Optional[Dict[str, str]] = None,
    ignore_mapping: Optional[Dict[str, str]] = None,
    ignore_as_class: bool = False,
) -> Tuple[bool, int]:
    """Check whether the category should be ignored and get its ID."""
    cat_name2id: Dict[str, int] = {
        category["name"]: category["id"] for category in categories
    }
    if name_mapping is not None:
        category_name = name_mapping.get(category_name, category_name)
    if category_name not in cat_name2id:
        if ignore_as_class:
            category_name = "ignored"
            category_ignored = False
        else:
            assert ignore_mapping is not None
            assert category_name in ignore_mapping, "%s" % category_name
            category_name = ignore_mapping[category_name]
            category_ignored = True
    else:
        category_ignored = False
    category_id = cat_name2id[category_name]
    return category_ignored, category_id


def get_object_attributes(label: Label, ignore: bool) -> Tuple[int, int]:
    """Set attributes for the ann dict."""
    attributes = label.attributes
    if attributes is None:
        return False, int(ignore)
    iscrowd = int(bool(attributes.get("crowd", False)) or ignore)
    return iscrowd, int(ignore)


def set_box_object_geometry(annotation: AnnType, label: Label) -> AnnType:
    """Parsing bbox, area, polygon for bbox ann."""
    box_2d = label.box_2d
    if box_2d is None:
        return annotation
    bbox = box2d_to_bbox(box_2d)
    annotation.update(dict(bbox=bbox, area=float(bbox[2] * bbox[3])))
    return annotation


def set_seg_object_geometry(
    annotation: AnnType, mask: np.ndarray, mask_mode: str = "rle"
) -> AnnType:
    """Parsing bbox, area, polygon from seg ann."""
    if not mask.sum():
        return annotation

    if mask_mode == "polygon":
        bbox = mask_to_bbox(mask)
        area = np.sum(mask).tolist()
        x, y, w, h = bbox
        x, y, w, h = int(x), int(y), int(w), int(h)
        mask = mask[y : y + h, x : x + w]
        polygon: PolygonType = mask_to_polygon(mask, x, y)
        annotation.update(dict(segmentation=polygon))
    elif mask_mode == "rle":
        rle: RLEType = mask_utils.encode(
            np.array(mask[:, :, None], order="F", dtype="uint8")
        )[0]
        rle["counts"] = rle["counts"].decode("utf-8")  # type: ignore
        bbox = mask_utils.toBbox(rle).tolist()
        area = mask_utils.area(rle).tolist()
        annotation.update(dict(segmentation=rle))

    annotation.update(dict(bbox=bbox, area=area))
    return annotation


def poly2ds_to_coco(
    annotation: AnnType,
    poly2d: List[Poly2D],
    shape: Tuple[int, int],
    mask_mode: str,
) -> AnnType:
    """Converting Poly2D to coco format."""
    mask = poly2ds_to_mask(shape, poly2d)
    set_seg_object_geometry(annotation, mask, mask_mode)
    return annotation


def pol2ds_list_to_coco(
    shape: Tuple[int, int],
    annotations: List[AnnType],
    poly_2ds: List[List[Poly2D]],
    mask_mode: str,
    nproc: int,
) -> List[AnnType]:
    """Execute the Poly2D to coco conversion in parallel."""
    with Pool(nproc) as pool:
        annotations = pool.starmap(
            partial(poly2ds_to_coco, shape=shape, mask_mode=mask_mode),
            tqdm(
                zip(annotations, poly_2ds),
                total=len(annotations),
            ),
        )

    sorted(annotations, key=lambda ann: ann["id"])
    return annotations


def scalabel2coco_detection(
    shape: Tuple[int, int],
    frames: List[Frame],
    categories: List[CatType],
    name_mapping: Optional[Dict[str, str]] = None,
    ignore_mapping: Optional[Dict[str, str]] = None,
    ignore_as_class: bool = False,
    remove_ignore: bool = False,
) -> GtType:
    """Convert Scalabel format to COCO detection."""
    image_id, ann_id = 0, 0
    images: List[ImgType] = []
    annotations: List[AnnType] = []

    for image_anns in tqdm(frames):
        image_id += 1
        image = ImgType(
            file_name=image_anns.name,
            height=shape[0],
            width=shape[1],
            id=image_id,
        )
        if image_anns.url is not None:
            image["coco_url"] = image_anns.url
        images.append(image)

        if image_anns.labels is None:
            continue

        for label in image_anns.labels:
            if label.box_2d is None:
                continue

            category_ignored, category_id = process_category(
                label.category,
                categories,
                name_mapping,
                ignore_mapping,
                ignore_as_class,
            )
            if remove_ignore and category_ignored:
                continue

            iscrowd, ignore = get_object_attributes(label, category_ignored)
            ann_id += 1
            annotation = AnnType(
                id=ann_id,
                image_id=image_id,
                category_id=category_id,
                scalabel_id=label.id,
                iscrowd=iscrowd,
                ignore=ignore,
            )
            if label.score is not None:
                annotation["score"] = label.score
            annotation = set_box_object_geometry(annotation, label)
            annotations.append(annotation)

    return GtType(
        type="instance",
        categories=categories,
        images=images,
        annotations=annotations,
    )


def scalabel2coco_ins_seg(
    shape: Tuple[int, int],
    frames: List[Frame],
    categories: List[CatType],
    name_mapping: Optional[Dict[str, str]] = None,
    ignore_mapping: Optional[Dict[str, str]] = None,
    ignore_as_class: bool = False,
    remove_ignore: bool = False,
    mask_mode: str = "rle",
    nproc: int = 4,
) -> GtType:
    """Convert Scalabel format to COCO instance segmentation."""
    image_id, ann_id = 0, 0
    images: List[ImgType] = []
    annotations: List[AnnType] = []
    poly_2ds: List[List[Poly2D]] = []

    for image_anns in tqdm(frames):
        image_id += 1
        image = ImgType(
            file_name=image_anns.name,
            height=shape[0],
            width=shape[1],
            id=image_id,
        )
        if image_anns.url is not None:
            image["coco_url"] = image_anns.url
        images.append(image)

        if image_anns.labels is None:
            continue

        for label in image_anns.labels:
            if label.poly_2d is None:
                continue

            category_ignored, category_id = process_category(
                label.category,
                categories,
                name_mapping,
                ignore_mapping,
                ignore_as_class,
            )
            if remove_ignore and category_ignored:
                continue

            iscrowd, ignore = get_object_attributes(label, category_ignored)
            ann_id += 1
            annotation = AnnType(
                id=ann_id,
                image_id=image_id,
                category_id=category_id,
                scalabel_id=label.id,
                iscrowd=iscrowd,
                ignore=ignore,
            )
            if label.score is not None:
                annotation["score"] = label.score
            annotations.append(annotation)
            poly_2ds.append(label.poly_2d)

    annotations = pol2ds_list_to_coco(
        shape, annotations, poly_2ds, mask_mode, nproc
    )
    return GtType(
        type="instance",
        categories=categories,
        images=images,
        annotations=annotations,
    )


def get_instance_id(
    instance_id_maps: Dict[str, int], global_instance_id: int, scalabel_id: str
) -> Tuple[int, int]:
    """Get instance id given its corresponding Scalabel id."""
    if scalabel_id in instance_id_maps.keys():
        instance_id = instance_id_maps[scalabel_id]
    else:
        instance_id = global_instance_id
        global_instance_id += 1
        instance_id_maps[scalabel_id] = instance_id
    return instance_id, global_instance_id


def scalabel2coco_box_track(
    shape: Tuple[int, int],
    frames: List[Frame],
    categories: List[CatType],
    name_mapping: Optional[Dict[str, str]] = None,
    ignore_mapping: Optional[Dict[str, str]] = None,
    ignore_as_class: bool = False,
    remove_ignore: bool = False,
) -> GtType:
    """Converting Scalabel Box Tracking Set to COCO format."""
    frames_list = group_and_sort(frames)
    video_id, image_id, ann_id = 0, 0, 0
    videos: List[VidType] = []
    images: List[ImgType] = []
    annotations: List[AnnType] = []

    for video_anns in tqdm(frames_list):
        global_instance_id: int = 1
        instance_id_maps: Dict[str, int] = dict()

        video_id += 1
        video_name = video_anns[0].video_name
        video = VidType(id=video_id, name=video_name)
        videos.append(video)

        for image_anns in video_anns:
            image_id += 1
            image = ImgType(
                video_id=video_id,
                frame_id=image_anns.frame_index,
                file_name=osp.join(video_name, image_anns.name),
                height=shape[0],
                width=shape[1],
                id=image_id,
            )
            if image_anns.url is not None:
                image["coco_url"] = image_anns.url
            images.append(image)

            if image_anns.labels is None:
                continue

            for label in image_anns.labels:
                if label.box_2d is None:
                    continue

                category_ignored, category_id = process_category(
                    label.category,
                    categories,
                    name_mapping,
                    ignore_mapping,
                    ignore_as_class,
                )
                if remove_ignore and category_ignored:
                    continue

                instance_id, global_instance_id = get_instance_id(
                    instance_id_maps, global_instance_id, label.id
                )
                iscrowd, ignore = get_object_attributes(
                    label, category_ignored
                )

                ann_id += 1
                annotation = AnnType(
                    id=ann_id,
                    image_id=image_id,
                    category_id=category_id,
                    instance_id=instance_id,
                    scalabel_id=label.id,
                    iscrowd=iscrowd,
                    ignore=ignore,
                )
                if label.score is not None:
                    annotation["score"] = label.score
                annotation = set_box_object_geometry(annotation, label)
                annotations.append(annotation)

    return GtType(
        categories=categories,
        videos=videos,
        images=images,
        annotations=annotations,
    )


def scalabel2coco_seg_track(
    shape: Tuple[int, int],
    frames: List[Frame],
    categories: List[CatType],
    name_mapping: Optional[Dict[str, str]] = None,
    ignore_mapping: Optional[Dict[str, str]] = None,
    ignore_as_class: bool = False,
    remove_ignore: bool = False,
    mask_mode: str = "rle",
    nproc: int = 4,
) -> GtType:
    """Convert Scalabel format to COCO instance segmentation."""
    frames_list = group_and_sort(frames)
    video_id, image_id, ann_id = 0, 0, 0
    videos: List[VidType] = []
    images: List[ImgType] = []
    annotations: List[AnnType] = []
    poly_2ds: List[List[Poly2D]] = []

    for video_anns in tqdm(frames_list):
        global_instance_id: int = 1
        instance_id_maps: Dict[str, int] = dict()

        video_id += 1
        video_name = video_anns[0].video_name
        video = VidType(id=video_id, name=video_name)
        videos.append(video)

        for image_anns in frames:
            image_id += 1
            image = ImgType(
                file_name=image_anns.name,
                height=shape[0],
                width=shape[1],
                id=image_id,
            )
            if image_anns.url is not None:
                image["coco_url"] = image_anns.url
            images.append(image)

            if image_anns.labels is None:
                continue

            for label in image_anns.labels:
                if label.poly_2d is None:
                    continue

                category_ignored, category_id = process_category(
                    label.category
                    if label.category is not None
                    else "ignored",
                    categories,
                    name_mapping,
                    ignore_mapping,
                    ignore_as_class,
                )
                if remove_ignore and category_ignored:
                    continue

                instance_id, global_instance_id = get_instance_id(
                    instance_id_maps, global_instance_id, label.id
                )
                iscrowd, ignore = get_object_attributes(
                    label, category_ignored
                )

                ann_id += 1
                annotation = AnnType(
                    id=ann_id,
                    image_id=image_id,
                    category_id=category_id,
                    instance_id=instance_id,
                    scalabel_id=label.id,
                    iscrowd=iscrowd,
                    ignore=ignore,
                )
                if label.score is not None:
                    annotation["score"] = label.score
                annotations.append(annotation)
                poly_2ds.append(label.poly_2d)

    annotations = pol2ds_list_to_coco(
        shape, annotations, poly_2ds, mask_mode, nproc
    )
    return GtType(
        categories=categories,
        videos=videos,
        images=images,
        annotations=annotations,
    )


def load_coco_config(
    mode: str, filepath: str, ignore_as_class: bool = False
) -> Tuple[List[CatType], Dict[str, str], Dict[str, str]]:
    """Load default configs from a config file."""
    cfgs = load_config(filepath)

    categories, cat_extensions = cfgs["categories"], cfgs["cat_extensions"]
    name_mapping, ignore_mapping = cfgs["name_mapping"], cfgs["ignore_mapping"]
    if mode == "det":
        categories += cat_extensions

    if ignore_as_class:
        categories.append(
            CatType(
                supercategory="none", id=len(categories) + 1, name="ignored"
            )
        )

    return categories, name_mapping, ignore_mapping


def run(args: argparse.Namespace) -> None:
    """Run."""
    categories, name_mapping, ignore_mapping = load_coco_config(
        args.mode, args.config, args.ignore_as_class
    )

    logger.info("Loading Scalabel jsons...")
    frames = load(args.label, args.nproc)

    logger.info("Start format converting...")
    if args.mode in ["det", "box_track"]:
        convert_func = dict(
            det=scalabel2coco_detection,
            box_track=scalabel2coco_box_track,
        )[args.mode]
    else:
        convert_func = partial(
            dict(
                ins_seg=scalabel2coco_ins_seg,
                seg_track=scalabel2coco_seg_track,
            )[args.mode],
            mask_mode=args.mask_mode,
            nproc=args.nproc,
        )
    shape = (args.height, args.width)
    coco = convert_func(
        shape,
        frames,
        categories,
        name_mapping,
        ignore_mapping,
        args.ignore_as_class,
        args.remove_ignore,
    )

    logger.info("Saving converted annotations...")
    with open(args.output, "w") as f:
        json.dump(coco, f)


if __name__ == "__main__":
    run(parse_arguments())
