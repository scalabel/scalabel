"""Convert coco to Scalabel format."""
import argparse
from typing import List

from pycocotools.coco import COCO

from .io import save as save_labels
from .typing import Frame as LabeledFrame
from .typing import Label


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
        help="path to save bdd formatted label file",
    )
    return parser.parse_args()


def transform(label_file: str) -> List[LabeledFrame]:
    """Transform to Scalabel format."""
    coco = COCO(label_file)
    img_ids = coco.getImgIds()
    img_ids = sorted(img_ids)
    cat_ids = coco.getCatIds()
    cats = coco.loadCats(cat_ids)
    nms = [cat["name"] for cat in cats]
    cat_map = dict(zip(coco.getCatIds(), nms))
    labels = []
    for img_id in img_ids:
        img = coco.loadImgs(img_id)[0]
        ann_ids = coco.getAnnIds(imgIds=img["id"])
        anns = coco.loadAnns(ann_ids)
        det_dict = LabeledFrame()
        det_dict.name = img["file_name"]
        det_dict.url = img["coco_url"]
        det_dict.labels = []
        for i, ann in enumerate(anns):
            label = Label(
                **{
                    "id": ann["id"],
                    "index": i + 1,
                    "category": cat_map[ann["category_id"]],
                    "box_2d": {
                        "x1": ann["bbox"][0],
                        "y1": ann["bbox"][1],
                        "x2": ann["bbox"][0] + ann["bbox"][2] - 1,
                        "y2": ann["bbox"][1] + ann["bbox"][3] - 1,
                    },
                }
            )
            det_dict.labels.append(label)
        labels.append(det_dict)
    return labels


def run() -> None:
    """Run."""
    args = parse_arguments()
    labels = transform(args.label)
    save_labels(args.output, labels)


if __name__ == "__main__":
    run()
