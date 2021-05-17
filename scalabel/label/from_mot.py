"""Convert MOT Challenge format dataset to Scalabel."""
import argparse
import os
from collections import defaultdict
from typing import Dict, List, Union

from ..common.io import load_file_as_list
from .io import load_label_config, save
from .transforms import bbox_to_box2d
from .typing import Category, Config, Frame, Label
from .utils import get_leaf_categories

# Classes in MOT:
#   1: 'pedestrian'
#   2: 'person on vehicle'
#   3: 'car'
#   4: 'bicycle'
#   5: 'motorbike'
#   6: 'non motorized vehicle'
#   7: 'static person'
#   8: 'distractor'
#   9: 'occluder'
#   10: 'occluder on the ground',
#   11: 'occluder full'
#   12: 'reflection'

IGNORE = [
    "person on vehicle",
    "static person",
    "distractor",
    "reflection",
    "ignore",
]
NAME_MAPPING = {
    "1": "pedestrian",
    "2": "person on vehicle",
    "7": "static person",
    "8": "distractor",
    "12": "reflection",
    "13": "ignore",
}
DISCARD = ["3", "4", "5", "6", "9", "10", "11"]


def parse_arguments() -> argparse.Namespace:
    """Parse the arguments."""
    parser = argparse.ArgumentParser(description="motchallenge to scalabel")
    parser.add_argument(
        "--data-path",
        "-d",
        help="path to MOTChallenge data (images + annotations).",
    )
    parser.add_argument(
        "--cfg-path",
        "-c",
        default=None,
        help="Config path for converting the annotations. Contains metadata "
        "like available categories.",
    )
    parser.add_argument(
        "--out-dir",
        "-o",
        default=".'",
        help="Output path for Scalabel format annotations.",
    )
    return parser.parse_args()


def parse_annotations(
    ann_filepath: str, metadata_cfg: Config
) -> Dict[int, List[Label]]:
    """Parse annotation file into List of Scalabel Label type per frame."""
    outputs = defaultdict(list)
    cats = [cat.name for cat in get_leaf_categories(metadata_cfg.categories)]
    for line in load_file_as_list(ann_filepath):
        gt = line.strip().split(",")
        class_id = gt[7]
        class_id = NAME_MAPPING[class_id]
        if class_id not in cats:
            continue
        frame_id, ins_id = map(int, gt[:2])
        bbox = list(map(float, gt[2:6]))
        box2d = bbox_to_box2d(bbox)
        attrs = dict(
            visibility=float(gt[8])
        )  # type: Dict[str, Union[bool, float ,str]]
        if class_id in IGNORE:
            attrs["crowd"] = True
        else:
            attrs["crowd"] = False
        ann = Label(
            category=class_id,
            id=ins_id,
            box2d=box2d,
            attributes=attrs,
        )
        outputs[frame_id].append(ann)
    return outputs


def from_mot(data_path: str, metadata_cfg: Config) -> List[Frame]:
    """Function converting MOT annotations to Scalabel format."""
    frames = []
    for video in sorted(os.listdir(data_path)):
        img_names = sorted(os.listdir(os.path.join(data_path, video, "img1")))
        annotations = parse_annotations(
            os.path.join(data_path, video, "gt/gt.txt"), metadata_cfg
        )

        for i, img_name in enumerate(img_names):
            frame = Frame(
                name=os.path.join("img1", img_name),
                video_name=video,
                frame_index=i,
                labels=annotations[i] if i in annotations else None,
            )
            frames.append(frame)
    return frames


def run(args: argparse.Namespace) -> None:
    """Run conversion with command line arguments."""
    if args.cfg_path is not None:
        metadata_cfg = load_label_config(args.cfg_path)
    else:
        metadata_cfg = Config(
            categories=[
                Category(name="pedestrian"),
                Category(name="person on vehicle"),
                Category(name="static person"),
                Category(name="distractor"),
                Category(name="reflection"),
                Category(name="ignore"),
            ]
        )

    result = from_mot(args.data_path, metadata_cfg)
    save(os.path.join(args.out_dir, "scalabel_anns.json"), result)


if __name__ == "__main__":
    run(parse_arguments())
