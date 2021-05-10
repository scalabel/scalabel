"""Convert MOT Challenge format dataset to Scalabel."""
import argparse
import os
from collections import defaultdict
from typing import Dict, List, Optional, Union

from scalabel.common.io import load_file_as_list
from scalabel.label.io import load_label_config
from scalabel.label.typing import Frame, Label

from .io import save

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
    parser = argparse.ArgumentParser(description="coco to scalabel")
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
    parser.add_argument(
        "--discard-classes",
        "-dc",
        default=None,
        help="Classes that should be discarded separated by comma, e.g. 1,2",
    )
    return parser.parse_args()


def parse_annotations(
    ann_filepath: str,
    name_mapping: Dict[str, str],
    discard_classes: List[str],
    ignore_classes: List[str],
) -> Dict[int, List[Label]]:
    """Parse annotation file into List of Scalabel Label type per frame."""
    outputs = defaultdict(list)
    for line in load_file_as_list(ann_filepath):
        gt = line.strip().split(",")
        class_id = gt[7]
        if class_id in discard_classes:
            continue
        class_id = name_mapping[class_id]
        frame_id, ins_id = map(int, gt[:2])
        bbox = list(map(float, gt[2:6]))
        box2d = dict(
            x1=bbox[0], y1=bbox[1], x2=bbox[0] + bbox[2], y2=bbox[1] + bbox[3]
        )
        attrs = dict(
            visibility=float(gt[8])
        )  # type: Dict[str, Union[bool, float ,str]]
        if class_id in ignore_classes:
            attrs["ignore"] = True
        else:
            attrs["ignore"] = False
        ann = Label(
            category=class_id,
            id=ins_id,
            box2d=box2d,
            attributes=attrs,
        )
        outputs[frame_id].append(ann)
    return outputs


def from_mot(
    data_path: str,
    name_mapping: Optional[Dict[str, str]] = None,
    discard_classes: Optional[List[str]] = None,
    ignore_classes: Optional[List[str]] = None,
) -> List[Frame]:
    """Function converting MOT annotations to Scalabel format."""
    frames = []
    # if the mappings are None, use defaults
    if name_mapping is None:
        name_mapping = NAME_MAPPING
    if discard_classes is None:
        discard_classes = DISCARD
    if ignore_classes is None:
        ignore_classes = IGNORE

    for video in os.listdir(data_path):
        img_names = sorted(os.listdir(os.path.join(data_path, video, "img1")))
        annotations = parse_annotations(
            os.path.join(data_path, video, "gt/gt.txt"),
            name_mapping,
            discard_classes,
            ignore_classes,
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
    name_map = None
    ignore_cls = None
    if args.cfg_path is not None:
        _, _, name_map, ignore_map = load_label_config(filepath=args.cfg_path)
        ignore_cls = list(ignore_map.keys()) if ignore_map is not None else []

    discard_cls = (
        args.discard_classes.split(",")
        if args.discard_classes is not None
        else None
    )

    result = from_mot(args.data_path, name_map, discard_cls, ignore_cls)
    save(os.path.join(args.out_dir, "scalabel_anns.json"), result)


if __name__ == "__main__":
    run(parse_arguments())
