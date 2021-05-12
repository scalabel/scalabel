"""Convert Scalabel to COCO format."""

import argparse
import json
import os.path as osp
from functools import partial
from multiprocessing import Pool
from typing import Dict, List, Tuple

import numpy as np
from pycocotools import mask as mask_utils  # type: ignore
from tqdm import tqdm

from scalabel.label.io import load_label_config
from scalabel.label.typing import MetaConfig

from ..common.logger import logger
from .coco_typing import (
    AnnType,
    GtType,
    ImgType,
    PolygonType,
    RLEType,
    VidType,
)
from .io import DEFAULT_LABEL_CONFIG, group_and_sort, load
from .transforms import (
    box2d_to_bbox,
    get_coco_categories,
    mask_to_bbox,
    mask_to_polygon,
    poly2ds_to_mask,
)
from .typing import Frame, Label, Poly2D
from .utils import get_category_id


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
        "-m",
        "--mode",
        default="det",
        choices=["det", "ins_seg", "box_track", "seg_track"],
        help="conversion mode: detection or tracking.",
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
        default=DEFAULT_LABEL_CONFIG,
        help="Configuration for COCO categories",
    )
    return parser.parse_args()


def get_object_attributes(label: Label) -> int:
    """Set attributes for the ann dict."""
    attributes = label.attributes
    if attributes is None:
        return 0
    iscrowd = int(bool(attributes.get("crowd", False)))
    return iscrowd


def set_box_object_geometry(annotation: AnnType, label: Label) -> AnnType:
    """Parsing bbox, area, polygon for bbox ann."""
    box2d = label.box2d
    if box2d is None:
        return annotation
    bbox = box2d_to_bbox(box2d)
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


def poly2ds_list_to_coco(
    shape: List[Tuple[int, int]],
    annotations: List[AnnType],
    poly2ds: List[List[Poly2D]],
    mask_mode: str,
    nproc: int,
) -> List[AnnType]:
    """Execute the Poly2D to coco conversion in parallel."""
    with Pool(nproc) as pool:
        annotations = pool.starmap(
            partial(poly2ds_to_coco, mask_mode=mask_mode),
            tqdm(
                zip(annotations, poly2ds, shape),
                total=len(annotations),
            ),
        )

    sorted(annotations, key=lambda ann: ann["id"])
    return annotations


def scalabel2coco_detection(
    frames: List[Frame], metadata_cfg: MetaConfig
) -> GtType:
    """Convert Scalabel format to COCO detection."""
    image_id, ann_id = 0, 0
    images: List[ImgType] = []
    annotations: List[AnnType] = []

    for image_anns in tqdm(frames):
        image_id += 1
        img_shape = metadata_cfg.resolution
        if img_shape is None:
            if image_anns.size is not None:
                img_shape = (image_anns.size[1], image_anns.size[0])
            else:
                raise ValueError("Image shape not defined!")

        image = ImgType(
            file_name=image_anns.name,
            height=img_shape[0],
            width=img_shape[1],
            id=image_id,
        )
        if image_anns.url is not None:
            image["coco_url"] = image_anns.url
        images.append(image)

        if image_anns.labels is None:
            continue

        for label in image_anns.labels:
            if label.box2d is None:
                continue

            iscrowd = get_object_attributes(label)
            category_id = get_category_id(label.category, metadata_cfg)
            ann_id += 1
            annotation = AnnType(
                id=ann_id,
                image_id=image_id,
                category_id=category_id,
                scalabel_id=label.id,
                iscrowd=iscrowd,
                ignore=False,
            )
            if label.score is not None:
                annotation["score"] = label.score
            annotation = set_box_object_geometry(annotation, label)
            annotations.append(annotation)

    return GtType(
        type="instance",
        categories=get_coco_categories(metadata_cfg),
        images=images,
        annotations=annotations,
    )


def scalabel2coco_ins_seg(
    frames: List[Frame],
    metadata_cfg: MetaConfig,
    mask_mode: str = "rle",
    nproc: int = 4,
) -> GtType:
    """Convert Scalabel format to COCO instance segmentation."""
    image_id, ann_id = 0, 0
    images: List[ImgType] = []
    annotations: List[AnnType] = []
    poly2ds: List[List[Poly2D]] = []

    shapes = []
    for image_anns in tqdm(frames):
        image_id += 1
        img_shape = metadata_cfg.resolution
        if img_shape is None:
            if image_anns.size is not None:
                img_shape = (image_anns.size[1], image_anns.size[0])
            else:
                raise ValueError("Image shape not defined!")
        shapes.append(img_shape)

        image = ImgType(
            file_name=image_anns.name,
            height=img_shape[0],
            width=img_shape[1],
            id=image_id,
        )
        if image_anns.url is not None:
            image["coco_url"] = image_anns.url
        images.append(image)

        if image_anns.labels is None:
            continue

        for label in image_anns.labels:
            if label.poly2d is None:
                continue

            iscrowd = get_object_attributes(label)
            category_id = get_category_id(label.category, metadata_cfg)
            ann_id += 1
            annotation = AnnType(
                id=ann_id,
                image_id=image_id,
                category_id=category_id,
                scalabel_id=label.id,
                iscrowd=iscrowd,
                ignore=False,
            )
            if label.score is not None:
                annotation["score"] = label.score
            annotations.append(annotation)
            poly2ds.append(label.poly2d)

    annotations = poly2ds_list_to_coco(
        shapes, annotations, poly2ds, mask_mode, nproc
    )
    return GtType(
        type="instance",
        categories=get_coco_categories(metadata_cfg),
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
    frames: List[Frame],
    metadata_cfg: MetaConfig,
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
            img_shape = metadata_cfg.resolution
            if img_shape is None:
                if image_anns.size is not None:
                    img_shape = (image_anns.size[1], image_anns.size[0])
                else:
                    raise ValueError("Image shape not defined!")

            image = ImgType(
                video_id=video_id,
                frame_id=image_anns.frame_index,
                file_name=osp.join(video_name, image_anns.name),
                height=img_shape[0],
                width=img_shape[1],
                id=image_id,
            )
            if image_anns.url is not None:
                image["coco_url"] = image_anns.url
            images.append(image)

            if image_anns.labels is None:
                continue

            for label in image_anns.labels:
                if label.box2d is None:
                    continue

                instance_id, global_instance_id = get_instance_id(
                    instance_id_maps, global_instance_id, label.id
                )
                iscrowd = get_object_attributes(label)
                category_id = get_category_id(label.category, metadata_cfg)

                ann_id += 1
                annotation = AnnType(
                    id=ann_id,
                    image_id=image_id,
                    category_id=category_id,
                    instance_id=instance_id,
                    scalabel_id=label.id,
                    iscrowd=iscrowd,
                    ignore=False,
                )
                if label.score is not None:
                    annotation["score"] = label.score
                annotation = set_box_object_geometry(annotation, label)
                annotations.append(annotation)

    return GtType(
        categories=get_coco_categories(metadata_cfg),
        videos=videos,
        images=images,
        annotations=annotations,
    )


def scalabel2coco_seg_track(
    frames: List[Frame],
    metadata_cfg: MetaConfig,
    mask_mode: str = "rle",
    nproc: int = 4,
) -> GtType:
    """Convert Scalabel format to COCO instance segmentation."""
    frames_list = group_and_sort(frames)
    video_id, image_id, ann_id = 0, 0, 0
    videos: List[VidType] = []
    images: List[ImgType] = []
    annotations: List[AnnType] = []
    poly2ds: List[List[Poly2D]] = []

    shapes = []
    for video_anns in tqdm(frames_list):
        global_instance_id: int = 1
        instance_id_maps: Dict[str, int] = dict()

        video_id += 1
        video_name = video_anns[0].video_name
        video = VidType(id=video_id, name=video_name)
        videos.append(video)

        for image_anns in frames:
            image_id += 1
            img_shape = metadata_cfg.resolution
            if img_shape is None:
                if image_anns.size is not None:
                    img_shape = (image_anns.size[1], image_anns.size[0])
                else:
                    raise ValueError("Image shape not defined!")
            shapes.append(img_shape)

            image = ImgType(
                video_id=video_id,
                frame_id=image_anns.frame_index,
                file_name=image_anns.name,
                height=img_shape[0],
                width=img_shape[1],
                id=image_id,
            )
            if image_anns.url is not None:
                image["coco_url"] = image_anns.url
            images.append(image)

            if image_anns.labels is None:
                continue

            for label in image_anns.labels:
                if label.poly2d is None:
                    continue

                instance_id, global_instance_id = get_instance_id(
                    instance_id_maps, global_instance_id, label.id
                )
                iscrowd = get_object_attributes(label)
                assert label.category is not None
                category_id = get_category_id(label.category, metadata_cfg)

                ann_id += 1
                annotation = AnnType(
                    id=ann_id,
                    image_id=image_id,
                    category_id=category_id,
                    instance_id=instance_id,
                    scalabel_id=label.id,
                    iscrowd=iscrowd,
                    ignore=False,
                )
                if label.score is not None:
                    annotation["score"] = label.score
                annotations.append(annotation)
                poly2ds.append(label.poly2d)

    annotations = poly2ds_list_to_coco(
        shapes, annotations, poly2ds, mask_mode, nproc
    )
    return GtType(
        categories=get_coco_categories(metadata_cfg),
        videos=videos,
        images=images,
        annotations=annotations,
    )


def run(args: argparse.Namespace) -> None:
    """Run."""
    logger.info("Loading Scalabel jsons...")
    frames, metadata_cfg = load(args.label, args.nproc)

    if args.cfg_path is not None:
        metadata_cfg = load_label_config(args.config)
    assert metadata_cfg is not None

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
    coco = convert_func(frames, metadata_cfg)

    logger.info("Saving converted annotations...")
    with open(args.output, "w") as f:
        json.dump(coco, f)


if __name__ == "__main__":
    run(parse_arguments())
