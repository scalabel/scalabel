"""Convert coco to Scalabel format."""

import argparse
import json
import os
from itertools import groupby
from multiprocessing import Pool
from typing import Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

from ..common.parallel import NPROC
from .coco_typing import AnnType, GtType, ImgType
from .io import group_and_sort, save
from .transforms import bbox_to_box2d, polygon_to_poly2ds
from .typing import Category, Config, Frame, ImageSize, Label


def parse_arguments() -> argparse.Namespace:
    """Parse the arguments."""
    parser = argparse.ArgumentParser(description="coco to scalabel")
    parser.add_argument(
        "--input",
        "-i",
        help="path to the input coco label file",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="path to save scalabel format label file",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=NPROC,
        help="number of processes for conversion",
    )
    return parser.parse_args()


def coco_to_scalabel(coco: GtType) -> Tuple[List[Frame], Config]:
    """Transform COCO object to scalabel format."""
    vid_id2name: Optional[Dict[int, str]] = None
    if "videos" in coco:
        vid_id2name = {
            video["id"]: video["name"]
            for video in coco["videos"]  # type: ignore
        }
    img_id2img: Dict[int, ImgType] = {img["id"]: img for img in coco["images"]}

    cats = [Category() for _ in range(len(coco["categories"]))]
    cat_id2name = {}
    for category in coco["categories"]:
        assert 0 < int(category["id"]) <= len(coco["categories"])
        cat_id2name[category["id"]] = category["name"]
        cats[int(category["id"]) - 1] = Category(name=category["name"])
    assert None not in cats
    config = Config(categories=cats)

    img_id2anns: Dict[int, Iterable[AnnType]] = {
        img_id: list(anns)
        for img_id, anns in groupby(
            coco["annotations"], lambda ann: ann["image_id"]
        )
    }

    scalabel: List[Frame] = []
    img_ids = sorted(img_id2img.keys())
    for img_id in tqdm(img_ids):
        img = img_id2img[img_id]
        frame = Frame(name=os.path.split(img["file_name"])[-1])
        frame.size = ImageSize(width=img["width"], height=img["height"])
        scalabel.append(frame)

        if "coco_url" in img:
            frame.url = img["coco_url"]
        if (
            vid_id2name is not None
            and "video_id" in img
            and img["video_id"] is not None
        ):
            frame.videoName = vid_id2name[  # pylint: disable=invalid-name
                img["video_id"]
            ]
        if "frame_id" in img:
            frame.frameIndex = img["frame_id"]  # pylint: disable=invalid-name

        if img_id not in img_id2anns:
            continue

        frame.labels = []
        anns = sorted(img_id2anns[img_id], key=lambda ann: ann["id"])
        for i, ann in enumerate(anns):
            label = Label(
                id=ann.get(
                    "scalabel_id", str(ann.get("instance_id", ann["id"]))
                ),
                index=i + 1,
                attributes=dict(
                    crowd=bool(ann["iscrowd"]), ignored=bool(ann["ignore"])
                ),
                category=cat_id2name[ann["category_id"]],
            )
            if "score" in ann:
                label.score = ann["score"]
            if "bbox" in ann and ann["bbox"] is not None:
                label.box2d = bbox_to_box2d(ann["bbox"])
            if "segmentation" in ann:
                # Currently only support conversion from polygon.
                assert isinstance(ann["segmentation"], list)
                label.poly2d = polygon_to_poly2ds(ann["segmentation"])
            frame.labels.append(label)

    return scalabel, config


def run(args: argparse.Namespace) -> None:
    """Run."""
    with open(args.input) as fp:
        coco: GtType = json.load(fp)
    scalabel, vid_id2name = coco_to_scalabel(coco)

    if vid_id2name is None:
        assert args.output.endswith(".json"), "output should be a json file"
        save(args.output, scalabel, args.nproc)
    else:
        scalabels = group_and_sort(scalabel)
        if not os.path.isdir(args.output):
            os.makedirs(args.output)
        save_paths = [
            os.path.join(args.output, str(video_anns[0].videoName) + ".json")
            for video_anns in scalabels
        ]
        with Pool(args.nproc) as pool:
            pool.starmap(
                save,
                tqdm(zip(save_paths, scalabels), total=len(scalabels)),
            )


if __name__ == "__main__":
    run(parse_arguments())
