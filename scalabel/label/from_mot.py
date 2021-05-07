"""Convert MOT Challenge format dataset to Scalabel."""
import os
from collections import defaultdict
from typing import Dict, List, Optional, Union

from scalabel.common.io import load_file_as_list
from scalabel.label.typing import Frame, Label

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
        attrs = dict(visibility=gt[8])  # type: Dict[str, Union[bool ,str]]
        if class_id in ignore_classes:
            attrs["ignore"] = True
        ann = Label(
            category=class_id,
            id=ins_id,
            box_2d=box2d,
            attributes=attrs,
        )
        outputs[frame_id].append(ann)
    return outputs


def from_mot(
    annotation_path: str,
    image_root: str,
    name_mapping: Optional[Dict[str, str]] = None,
    discard_classes: Optional[List[str]] = None,
    ignore_classes: Optional[List[str]] = None,
) -> List[Frame]:
    """Function converting MOT dataset to Scalabel format (List[Frame])."""
    frames = []
    # if the mappings are None, use defaults
    if name_mapping is None:
        name_mapping = NAME_MAPPING
    if discard_classes is None:
        discard_classes = DISCARD
    if ignore_classes is None:
        ignore_classes = IGNORE

    for video in os.listdir(annotation_path):
        img_names = sorted(os.listdir(os.path.join(image_root, video, "img1")))
        annotations = parse_annotations(
            os.path.join(image_root, video, "gt/gt.txt"),
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
