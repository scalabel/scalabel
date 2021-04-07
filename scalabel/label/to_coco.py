"""Convert Scalabel to COCO format."""

import argparse
import glob
import json
import os.path as osp
from functools import partial
from itertools import groupby
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import toml
from matplotlib.path import Path
from pycocotools import mask as mask_utils  # type: ignore
from skimage import measure
from tqdm import tqdm

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
from .io import load
from .typing import Frame, Label, Poly2D


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
        "-h",
        "--height",
        type=int,
        default=720,
        help="Height of images",
    )
    parser.add_argument(
        "-w",
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
        help="number of processes for mot evaluation",
    )
    return parser.parse_args()


def group_and_sort(inputs: List[Frame]) -> List[List[Frame]]:
    """Group frames by video_name and sort."""
    for frame in inputs:
        assert frame.video_name is not None
        assert frame.frame_index is not None
    frames_list: List[List[Frame]] = []
    for _, frame_iter in groupby(inputs, lambda frame: frame.video_name):
        frames = sorted(
            list(frame_iter),
            key=lambda frame: frame.frame_index if frame.frame_index else 0,
        )
        frames_list.append(frames)
    frames_list = sorted(
        frames_list, key=lambda frames: str(frames[0].video_name)
    )
    return frames_list


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
    x1 = box_2d.x1
    y1 = box_2d.y1
    x2 = box_2d.x2
    y2 = box_2d.y2

    annotation.update(
        dict(
            bbox=[x1, y1, x2 - x1 + 1, y2 - y1 + 1],
            area=float((x2 - x1 + 1) * (y2 - y1 + 1)),
        )
    )
    return annotation


def poly_to_patch(
    vertices: List[Tuple[float, float]],
    types: str,
    color: Tuple[float, float, float],
    closed: bool,
) -> mpatches.PathPatch:
    """Draw polygons using the Bezier curve."""
    moves = {"L": Path.LINETO, "C": Path.CURVE4}
    points = list(vertices)
    codes = [moves[t] for t in types]
    codes[0] = Path.MOVETO

    if closed:
        points.append(points[0])
        codes.append(Path.LINETO)

    return mpatches.PathPatch(
        Path(points, codes),
        facecolor=color if closed else "none",
        edgecolor=color,
        lw=0 if closed else 1,
        alpha=1,
        antialiased=False,
        snap=True,
    )


def poly2ds_to_mask(
    shape: Tuple[int, int], poly2d: List[Poly2D]
) -> np.ndarray:
    """Converting Poly2D to mask."""
    fig = plt.figure(facecolor="0")
    fig.set_size_inches(shape[1] / fig.get_dpi(), shape[0] / fig.get_dpi())
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, shape[1])
    ax.set_ylim(0, shape[0])
    ax.set_facecolor((0, 0, 0, 0))
    ax.invert_yaxis()

    for poly in poly2d:
        ax.add_patch(
            poly_to_patch(
                poly.vertices,
                poly.types,
                color=(1, 1, 1),
                closed=True,
            )
        )

    fig.canvas.draw()
    mask: np.ndarray = np.frombuffer(fig.canvas.tostring_rgb(), np.uint8)
    mask = mask.reshape((*shape, -1))[..., 0]
    plt.close()
    return mask


def close_contour(contour: np.ndarray) -> np.ndarray:
    """Explicitly close the contour."""
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def mask_to_polygon(
    binary_mask: np.ndarray, x_1: int, y_1: int, tolerance: float = 0.5
) -> List[List[float]]:
    """Convert BitMask to polygon."""
    polygons = []
    padded_binary_mask = np.pad(
        binary_mask, pad_width=1, mode="constant", constant_values=0
    )
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        segmentation = [0 if i < 0 else i for i in segmentation]
        for i, _ in enumerate(segmentation):
            if i % 2 == 0:
                segmentation[i] = float(segmentation[i] + x_1)
            else:
                segmentation[i] = float(segmentation[i] + y_1)

        polygons.append(segmentation)

    return polygons


def set_seg_object_geometry(
    annotation: AnnType, mask: np.ndarray, mask_mode: str = "rle"
) -> AnnType:
    """Parsing bbox, area, polygon from seg ann."""
    if not mask.sum():
        return annotation

    if mask_mode == "polygon":
        x_inds = np.nonzero(np.sum(mask, axis=0))[0]
        y_inds = np.nonzero(np.sum(mask, axis=1))[0]
        x1, x2 = np.min(x_inds), np.max(x_inds)
        y1, y2 = np.min(y_inds), np.max(y_inds)
        mask = mask[y1 : y2 + 1, x1 : x2 + 1]
        polygon: PolygonType = mask_to_polygon(mask, x1, y1)
        bbox = np.array([x1, y1, x2 - x1 + 1, y2 - y1 + 1]).tolist()
        area = np.sum(mask).tolist()
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
    shape: Tuple[int, int],
    annotation: AnnType,
    poly2d: List[Poly2D],
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
    pool = Pool(nproc)
    annotations = pool.starmap(
        partial(poly2ds_to_coco, shape=shape, mask_mode=mask_mode),
        tqdm(
            zip(annotations, poly_2ds),
            total=len(annotations),
        ),
    )
    pool.close()

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
    image_id, ann_id = 1, 1
    images: List[ImgType] = []
    annotations: List[AnnType] = []

    for image_anns in tqdm(frames):
        image = ImgType(
            file_name=image_anns.name,
            height=shape[0],
            width=shape[1],
            id=image_id,
        )
        if image_anns.url is not None:
            image["coco_url"] = image_anns.url
        images.append(image)

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
            annotation = AnnType(
                id=ann_id,
                image_id=image_id,
                category_id=category_id,
                scalabel_id=label.id,
                iscrowd=iscrowd,
                ignore=ignore,
            )
            annotation = set_box_object_geometry(annotation, label)
            annotations.append(annotation)

            ann_id += 1
        image_id += 1

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
    image_id, ann_id = 1, 1
    images: List[ImgType] = []
    annotations: List[AnnType] = []
    poly_2ds: List[List[Poly2D]] = []

    for image_anns in tqdm(frames):
        image = ImgType(
            file_name=image_anns.name,
            height=shape[0],
            width=shape[1],
            id=image_id,
        )
        if image_anns.url is not None:
            image["coco_url"] = image_anns.url
        images.append(image)

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
            annotation = AnnType(
                id=ann_id,
                image_id=image_id,
                category_id=category_id,
                scalabel_id=label.id,
                iscrowd=iscrowd,
                ignore=ignore,
            )
            annotations.append(annotation)
            poly_2ds.append(label.poly_2d)

            ann_id += 1
        image_id += 1

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
    video_id, image_id, ann_id = 1, 1, 1
    videos: List[VidType] = []
    images: List[ImgType] = []
    annotations: List[AnnType] = []

    for video_anns in tqdm(frames_list):
        global_instance_id: int = 1
        instance_id_maps: Dict[str, int] = dict()

        video_name = video_anns[0].video_name
        video = VidType(id=video_id, name=video_name)
        videos.append(video)

        for image_anns in video_anns:
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
                annotation = AnnType(
                    id=ann_id,
                    image_id=image_id,
                    category_id=category_id,
                    instance_id=instance_id,
                    scalabel_id=label.id,
                    iscrowd=iscrowd,
                    ignore=ignore,
                )
                annotation = set_box_object_geometry(annotation, label)
                annotations.append(annotation)

                ann_id += 1
            image_id += 1
        video_id += 1

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
    video_id, image_id, ann_id = 1, 1, 1
    videos: List[VidType] = []
    images: List[ImgType] = []
    annotations: List[AnnType] = []
    poly_2ds: List[List[Poly2D]] = []

    for video_anns in tqdm(frames_list):
        global_instance_id: int = 1
        instance_id_maps: Dict[str, int] = dict()

        # videos
        video_name = video_anns[0].video_name
        video = VidType(id=video_id, name=video_name)
        videos.append(video)

        for image_anns in tqdm(frames):
            image = ImgType(
                file_name=image_anns.name,
                height=shape[0],
                width=shape[1],
                id=image_id,
            )
            if image_anns.url is not None:
                image["coco_url"] = image_anns.url
            images.append(image)

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

                instance_id, global_instance_id = get_instance_id(
                    instance_id_maps, global_instance_id, label.id
                )
                iscrowd, ignore = get_object_attributes(
                    label, category_ignored
                )
                annotation = AnnType(
                    id=ann_id,
                    image_id=image_id,
                    category_id=category_id,
                    instance_id=instance_id,
                    scalabel_id=label.id,
                    iscrowd=iscrowd,
                    ignore=ignore,
                )
                annotations.append(annotation)
                poly_2ds.append(label.poly_2d)

                ann_id += 1
            image_id += 1
        video_id += 1

    annotations = pol2ds_list_to_coco(
        shape, annotations, poly_2ds, mask_mode, nproc
    )
    return GtType(
        categories=categories,
        videos=videos,
        images=images,
        annotations=annotations,
    )


def read(inputs: str) -> List[Frame]:
    """Read annotations from file/files."""
    outputs: List[Frame] = []
    if osp.isdir(inputs):
        files = glob.glob(osp.join(inputs, "*.json"))
        for file_ in files:
            outputs.extend(load(file_))
    elif osp.isfile(inputs) and inputs.endswith("json"):
        outputs.extend(load(inputs))
    else:
        raise TypeError("Inputs must be a folder or a JSON file.")

    return outputs


def load_default_cfgs(
    mode: str, ignore_as_class: bool = False
) -> Tuple[List[CatType], Dict[str, str], Dict[str, str]]:
    """Load default configs from the toml file."""
    cur_dir = osp.dirname(osp.abspath(__file__))
    cfg_file = osp.join(cur_dir, "configs.toml")
    cfgs = toml.load(cfg_file)

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


def main() -> None:
    """Main."""
    args = parse_arguments()
    categories, name_mapping, ignore_mapping = load_default_cfgs(
        args.mode, args.ignore_as_class
    )

    logger.info("Loading Scalabel jsons...")
    frames = read(args.label)

    logger.info("Start format converting...")
    if args.mode in ["det", "box_track"]:
        convert_func = dict(
            det=scalabel2coco_detection,
            box_track=scalabel2coco_box_track,
        )[args.mode]
    else:
        convert_func = partial(
            dict(
                det=scalabel2coco_detection,
                box_track=scalabel2coco_box_track,
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
    main()
