"""Convert coco to scalabel format."""
import argparse
import json
from typing import Any, Dict, List

from pycocotools.coco import COCO

DictAny = Dict[str, Any]  # type: ignore[misc]


def parse_arguments() -> argparse.Namespace:
    """Parse the arguments."""
    parser = argparse.ArgumentParser(description="coco to scalabel")
    parser.add_argument(
        "--annFile",
        "-a",
        default="/path/to/coco/label/file",
        help="path to coco label file",
    )
    parser.add_argument(
        "--save_path",
        "-s",
        default="/save/path",
        help="path to save scalabel formatted label file",
    )
    return parser.parse_args()


def transform(label_file: str) -> List[DictAny]:
    """Transform to scalabel format."""
    coco = COCO(label_file)
    img_ids = coco.getImgIds()
    img_ids = sorted(img_ids)
    cat_ids = coco.getCatIds()
    cats = coco.loadCats(cat_ids)
    nms = [cat["name"] for cat in cats]
    cat_map = dict(zip(coco.getCatIds(), nms))
    scalabel_label = []
    for img_id in img_ids:
        img = coco.loadImgs(img_id)[0]
        ann_ids = coco.getAnnIds(imgIds=img["id"])
        anns = coco.loadAnns(ann_ids)
        det_dict = {}
        det_dict["name"] = img["file_name"]
        det_dict["url"] = img["coco_url"]
        det_dict["attributes"] = {
            "weather": "undefined",
            "scene": "undefined",
            "timeofday": "undefined",
        }
        det_dict["labels"] = []
        for ann in anns:
            label = {
                "id": ann["id"],
                "category": cat_map[ann["category_id"]],
                "manualShape": True,
                "manualAttributes": True,
                "box2d": {
                    "x1": ann["bbox"][0],
                    "y1": ann["bbox"][1],
                    "x2": ann["bbox"][0] + ann["bbox"][2] - 1,
                    "y2": ann["bbox"][1] + ann["bbox"][3] - 1,
                },
            }
            det_dict["labels"].append(label)
        scalabel_label.append(det_dict)
    return scalabel_label


def main() -> None:
    """Main."""
    args = parse_arguments()
    scalabel_label = transform(args.annFile)
    with open(args.save_path, "w") as outfile:
        json.dump(scalabel_label, outfile)


if __name__ == "__main__":
    main()
