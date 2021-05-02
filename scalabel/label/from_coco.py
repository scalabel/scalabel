"""Convert coco to Scalabel format."""
import argparse
import json
import os
from itertools import groupby
from multiprocessing import Pool
from typing import Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

from .coco_typing import AnnType, GtType, ImgType
from .io import group_and_sort, save
from .transforms import bbox_to_box2d, polygon_to_poly2ds
from .typing import Frame, Label


def parse_arguments() -> argparse.Namespace:
    """Parse the arguments."""
    parser = argparse.ArgumentParser(description="coco to scalabel")
    parser.add_argument(
        "--label",
        "-l",
        help="path to coco label file",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="path to save scalabel formatted label file",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=4,
        help="number of processes for conversion",
    )
    return parser.parse_args()


def coco_to_scalabel(
    coco: GtType,
) -> Tuple[List[Frame], Optional[Dict[int, str]]]:
    """Transform COCO object to scalabel format."""
    vid_id2name: Optional[Dict[int, str]] = None
    if "videos" in coco:
        vid_id2name = {
            video["id"]: video["name"]
            for video in coco["videos"]  # type: ignore
        }
    img_id2img: Dict[int, ImgType] = {img["id"]: img for img in coco["images"]}
    cat_id2name: Dict[int, str] = {
        category["id"]: category["name"] for category in coco["categories"]
    }

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
        if "coco_url" in img:
            frame.url = img["coco_url"]
        if vid_id2name is not None and "video_id" in img:
            frame.video_name = vid_id2name[img["video_id"]]  # type: ignore
        if "frame_id" in img:
            frame.frame_index = img["frame_id"]

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
                attributes=dict(),
                category=cat_id2name[ann["category_id"]],
            )
            if "score" in ann:
                label.score = ann["score"]
            if "bbox" in ann:
                label.box_2d = bbox_to_box2d(ann["bbox"])  # type: ignore
            if "segmentation" in ann:
                # Currently only support conversion from polygon.
                assert isinstance(ann["segmentation"], list)
                label.poly_2d = polygon_to_poly2ds(ann["segmentation"])
            frame.labels.append(label)
        scalabel.append(frame)

    return scalabel, vid_id2name


def run(args: argparse.Namespace) -> None:
    """Run."""
    with open(args.label) as fp:
        coco: GtType = json.load(fp)
    scalabel, vid_id2name = coco_to_scalabel(coco)
    print(args.nproc)

    if vid_id2name is None:
        assert args.output.endswith(".json"), "output should be a json file"
        save(args.output, scalabel, args.nproc)
    else:
        scalabels = group_and_sort(scalabel)
        if not os.path.isdir(args.output):
            os.makedirs(args.output)
        save_paths = [
            os.path.join(args.output, str(video_anns[0].video_name) + ".json")
            for video_anns in scalabels
        ]
        with Pool(args.nproc) as pool:
            pool.starmap(
                save,
                tqdm(zip(save_paths, scalabels), total=len(scalabels)),
            )


if __name__ == "__main__":
    run(parse_arguments())
