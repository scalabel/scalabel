"""Convert MOT Challenge format dataset to Scalabel."""
import argparse
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Union

from ..common.io import load_file_as_list
from .io import save
from .transforms import bbox_to_box2d
from .typing import Category, Config, Dataset, Frame, Label

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
        "--input",
        "-i",
        help="path to MOTChallenge data (images + annotations).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=".'",
        help="Output path for Scalabel format annotations.",
    )
    parser.add_argument(
        "--split-val",
        action="store_true",
        help="Split each video into train and validation parts (50 / 50).",
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
        class_name = NAME_MAPPING[class_id]
        frame_id, ins_id = map(int, gt[:2])
        bbox = list(map(float, gt[2:6]))
        box2d = bbox_to_box2d(bbox)
        ignored = False
        if class_name in IGNORE:
            ignored = True
            class_name = "pedestrian"
        attrs = dict(
            visibility=float(gt[8]), ignored=ignored
        )  # type: Dict[str, Union[bool, float, str]]
        ann = Label(
            category=class_name,
            id=ins_id,
            box2d=box2d,
            attributes=attrs,
        )
        outputs[frame_id].append(ann)
    return outputs


def from_mot(
    data_path: str, split_val: bool = False
) -> Union[Dataset, Tuple[Dataset, Dataset]]:
    """Function converting MOT annotations to Scalabel format."""
    frames, val_frames = [], []
    for video in sorted(os.listdir(data_path)):
        if not os.path.isdir(os.path.join(data_path, video)):
            continue
        video_frames = []
        img_names = sorted(os.listdir(os.path.join(data_path, video, "img1")))
        annotations = parse_annotations(
            os.path.join(data_path, video, "gt/gt.txt"),
        )

        for i, img_name in enumerate(img_names):
            assert i + 1 == int(img_name.replace(".jpg", ""))
            relative_path = os.path.join(video, "img1", img_name)
            frame = Frame(
                name=img_name,
                videoName=video,
                url=relative_path,
                frameIndex=i,
                labels=annotations[i + 1] if i in annotations else None,
            )
            video_frames.append(frame)
        if split_val:
            split_frame = len(img_names) // 2 + 1
            frames.extend(video_frames[:split_frame])
            for val_frame in video_frames[split_frame:]:
                assert val_frame.frameIndex is not None
                val_frame.frameIndex -= split_frame
            val_frames.extend(video_frames[split_frame:])
        else:
            frames.extend(video_frames)

    cfg = Config(categories=[Category(name="pedestrian")])
    dataset = Dataset(frames=frames, config=cfg)
    if split_val:
        return dataset, Dataset(frames=val_frames, config=cfg)
    return dataset


def run(args: argparse.Namespace) -> None:
    """Run conversion with command line arguments."""
    result = from_mot(args.input, args.split_val)
    if isinstance(result, tuple):
        train, val = result
        save(os.path.join(args.output, "train_scalabel.json"), train)
        save(os.path.join(args.output, "val_scalabel.json"), val)
    else:
        save(os.path.join(args.output, "annotations_scalabel.json"), result)


if __name__ == "__main__":
    run(parse_arguments())
