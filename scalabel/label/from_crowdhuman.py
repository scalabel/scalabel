"""Convert CrowdHuman format dataset to Scalabel."""
import argparse
import json
import os
from typing import List

from ..common.typing import DictStrAny
from .io import save
from .typing import Box2D, Category, Config, Dataset, Frame, Label


def parse_arguments() -> argparse.Namespace:
    """Parse the arguments."""
    parser = argparse.ArgumentParser(description="crowdhuman to scalabel")
    parser.add_argument(
        "--input",
        "-i",
        help="path to Crowdhuman annotation file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="./annotations_scalabel.json",
        help="Output filename for Scalabel format annotations.",
    )
    return parser.parse_args()


def parse_annotations(annotations: List[DictStrAny]) -> List[Label]:
    """Parse annotations per frame."""
    labels = []
    for anno in annotations:
        if not anno["tag"] == "person":
            continue
        if anno["extra"].get("ignore", 0) == 1:
            continue

        x, y, w, h = anno["fbox"]
        box2d = Box2D(x1=x, y1=y, x2=x + w, y2=y + h)
        label = Label(
            id=anno["extra"]["box_id"],
            category="pedestrian",
            box2d=box2d,
        )
        labels.append(label)
    return labels


def from_crowdhuman(input_path: str, image_path: str = "./Images/") -> Dataset:
    """Function converting CrowdHuman annotations to Scalabel format."""
    frames = []
    with open(input_path, "r", encoding="utf-8") as anno_file:
        lines = anno_file.readlines()
        for line in lines:
            frame_json = json.loads(line)
            name = frame_json["ID"]
            frame = Frame(
                name=name,
                url=os.path.join(image_path, name + ".jpg"),
                labels=parse_annotations(frame_json["gtboxes"]),
            )
            frames.append(frame)
    dataset = Dataset(
        frames=frames, config=Config(categories=[Category(name="pedestrian")])
    )
    return dataset


def run(args: argparse.Namespace) -> None:
    """Run conversion with command line arguments."""
    result = from_crowdhuman(args.input)
    save(args.output, result)


if __name__ == "__main__":
    run(parse_arguments())
