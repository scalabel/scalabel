"""Convert MOT Challenge format dataset to Scalabel."""
import argparse
import os
from collections import defaultdict
from typing import Dict, List, Union

from ..common.io import load_file_as_list
from .io import save
from .transforms import bbox_to_box2d
from .typing import Frame, Label

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


def parse_arguments() -> argparse.Namespace:
    """Parse the arguments."""
    parser = argparse.ArgumentParser(description="motchallenge to scalabel")
    parser.add_argument(
        "--data-path",
        "-d",
        help="path to MOTChallenge data (images + annotations).",
    )
    parser.add_argument(
        "--out-dir",
        "-o",
        default=".'",
        help="Output path for Scalabel format annotations.",
    )
    return parser.parse_args()


def parse_annotations(ann_filepath: str) -> Dict[int, List[Label]]:
    """Parse annotation file into List of Scalabel Label type per frame."""
    outputs = defaultdict(list)
    for line in load_file_as_list(ann_filepath):
        gt = line.strip().split(",")
        class_id = gt[7]
        if class_id not in NAME_MAPPING:
            continue
        class_id = NAME_MAPPING[class_id]
        frame_id, ins_id = map(int, gt[:2])
        bbox = list(map(float, gt[2:6]))
        box2d = bbox_to_box2d(bbox)
        attrs = dict(
            visibility=float(gt[8])
        )  # type: Dict[str, Union[bool, float, str]]
        if class_id in IGNORE:
            attrs["crowd"] = True
            class_id = "pedestrian"
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


def from_mot(data_path: str) -> List[Frame]:
    """Function converting MOT annotations to Scalabel format."""
    frames = []
    for video in sorted(os.listdir(data_path)):
        img_names = sorted(os.listdir(os.path.join(data_path, video, "img1")))
        annotations = parse_annotations(
            os.path.join(data_path, video, "gt/gt.txt"),
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
    result = from_mot(args.data_path)
    save(os.path.join(args.out_dir, "scalabel_anns.json"), result)


if __name__ == "__main__":
    run(parse_arguments())
