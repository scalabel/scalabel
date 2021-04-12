"""Convert coco to Scalabel format."""
import argparse
import json
import os
from itertools import groupby
from typing import Dict, Iterable, List, Optional, Tuple

from .coco_typing import AnnType, GtType, ImgType, PolygonType
from .io import save
from .to_coco import group_and_sort
from .typing import Box2D, Frame, Label, Poly2D


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
    return parser.parse_args()


def bbox_to_box2d(bbox: List[float]) -> Box2D:
    """Convert COCO bbox into Scalabel Box2D."""
    assert len(bbox) == 4
    x1, y1, width, height = bbox
    x2, y2 = x1 + width - 1, y1 + height - 1
    return Box2D(x1=x1, y1=y1, x2=x2, y2=y2)


def polygon_to_poly2ds(polygon: PolygonType) -> List[Poly2D]:
    """Convert COCO polygon into Scalabel Box2Ds."""
    poly_2ds: List[Poly2D] = []
    for poly in polygon:
        point_num = len(poly) // 2
        assert 2 * point_num == len(poly)
        vertices = [[poly[2 * i], poly[2 * i + 1]] for i in range(point_num)]
        poly_2d = Poly2D(vertices=vertices, types="L" * point_num, closed=True)
        poly_2ds.append(poly_2d)
    return poly_2ds


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
    assert len(set(img_id2img.keys()) - set(img_id2anns.keys())) == 0

    scalabel: List[Frame] = []
    img_ids = sorted(img_id2img.keys())
    for img_id in img_ids:
        img = img_id2img[img_id]
        frame = Frame(name=os.path.split(img["file_name"])[-1])
        if "coco_url" in img:
            frame.url = img["coco_url"]
        if vid_id2name is not None and "video_id" in img:
            frame.video_name = vid_id2name[img["video_id"]]  # type: ignore
        if "frame_id" in img:
            frame.frame_index = img["frame_id"]

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

    if vid_id2name is None:
        assert args.output.endswith(".json"), "output should be a json file"
        save(args.output, scalabel)
    else:
        scalabels = group_and_sort(scalabel)
        if not os.path.isdir(args.output):
            os.makedirs(args.output)
        for video_anns in scalabels:
            assert video_anns[0].video_name is not None
            save_name = video_anns[0].video_name + ".json"
            save_path = os.path.join(args.output, save_name)
            save(save_path, video_anns)


if __name__ == "__main__":
    run(parse_arguments())
