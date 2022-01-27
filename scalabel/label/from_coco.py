"""Convert coco to Scalabel format."""

import argparse
import json
import os
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

from ..common.io import open_read_text
from ..common.parallel import NPROC
from ..common.tqdm import tqdm
from .coco_typing import AnnType, GtType, ImgType
from .io import group_and_sort, save
from .transforms import bbox_to_box2d, coco_rle_to_rle, polygon_to_poly2ds
from .typing import Category, Config, Dataset, Frame, ImageSize, Label


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

    cats: List[Optional[Category]] = [
        None for _ in range(len(coco["categories"]))
    ]
    cat_id2name = {}
    uniq_catids = set(category["id"] for category in coco["categories"])
    assert len(uniq_catids) == len(coco["categories"])
    uniq_catids_list = sorted(list(uniq_catids))
    for category in coco["categories"]:
        cat_id2name[category["id"]] = category["name"]
        cats[uniq_catids_list.index(category["id"])] = Category(
            name=category["name"]
        )
    assert None not in cats
    config = Config(categories=cats)

    img_id2anns: Dict[int, List[AnnType]] = {
        img_id: [] for img_id in img_id2img
    }
    for ann in coco["annotations"]:
        if ann["image_id"] not in img_id2anns:
            continue
        img_id2anns[ann["image_id"]].append(ann)

    scalabel: List[Frame] = []
    img_ids = sorted(img_id2img.keys())
    for img_id in tqdm(img_ids):
        img = img_id2img[img_id]
        size = ImageSize(width=img["width"], height=img["height"])

        if "file_name" in img:
            url: Optional[str] = img["file_name"]
        else:
            url = None

        if (
            vid_id2name is not None
            and "video_id" in img
            and img["video_id"] is not None
        ):
            video_name: Optional[str] = vid_id2name[img["video_id"]]
        else:
            video_name = None
        if "frame_id" in img:
            frame_index = img["frame_id"]
        else:
            frame_index = None

        labels: Optional[List[Label]] = None
        if img_id in img_id2anns:
            labels = []
            anns = sorted(img_id2anns[img_id], key=lambda ann: ann["id"])
            for i, ann in enumerate(anns):
                label = Label(
                    id=ann.get(
                        "scalabel_id", str(ann.get("instance_id", ann["id"]))
                    ),
                    index=i + 1,
                    attributes=dict(
                        crowd=bool(ann.get("iscrowd", None)),
                        ignored=bool(ann.get("ignore", None)),
                    ),
                    category=cat_id2name[ann["category_id"]],
                )
                if "score" in ann:
                    label.score = ann["score"]
                if "bbox" in ann and ann["bbox"] is not None:
                    label.box2d = bbox_to_box2d(ann["bbox"])
                if "segmentation" in ann:
                    # Currently only support conversion from polygon.
                    assert ann["segmentation"] is not None
                    if isinstance(ann["segmentation"], list):
                        label.poly2d = polygon_to_poly2ds(ann["segmentation"])
                    else:
                        label.rle = coco_rle_to_rle(ann["segmentation"])
                labels.append(label)

        scalabel.append(
            Frame(
                name=os.path.split(img["file_name"])[-1],
                url=url,
                size=size,
                videoName=video_name,
                frameIndex=frame_index,
                labels=labels,
            )
        )

    return scalabel, config


def run(args: argparse.Namespace) -> None:
    """Run."""
    with open_read_text(args.input) as fp:
        coco: GtType = json.load(fp)
    scalabel, config = coco_to_scalabel(coco)

    has_videos = all(frame.videoName is not None for frame in scalabel)
    if not has_videos:
        assert args.output.endswith(".json"), "output should be a json file"
        save(args.output, Dataset(frames=scalabel, config=config))
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
