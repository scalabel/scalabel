"""Edit the label and image list files in BDD100K format."""

import argparse
import json
from os.path import basename, join, splitext
from typing import Any, Callable, Dict, List

import yaml

from scalabel.common.logger import logger

LabelObject = Dict[str, Any]  # type: ignore


def parse_arguments() -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser(
        description="Edit the label and image list files in BDD100K format"
    )
    parser.add_argument(
        "--add-url",
        type=str,
        default="",
        help="add url based on the name field in each frame",
    )
    parser.add_argument(
        "--remove-name-dir",
        action="store_true",
        help="ignore the directory portion of the names when converting"
        + " name to url",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        nargs="+",
        help="path to the input bdd100k format files",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="path to the output file ",
    )

    args = parser.parse_args()
    return args


def add_url(frame: LabelObject, url_root: str, remove_name_dir: bool) -> None:
    """Add url root to the name and assign to the url frame."""
    name = frame["name"]
    if remove_name_dir:
        name = basename(name)
    frame["url"] = join(url_root, name)


def edit_frames(
    frames: List[LabelObject], url_root: str, remove_name_dir: bool
) -> None:
    """Edit frames based on the arguments."""
    processor: List[Callable[[LabelObject], None]] = []
    if url_root != "":
        processor.append(lambda f: add_url(f, url_root, remove_name_dir))
    for f in frames:
        for p in processor:
            p(f)


def read_input(filename: str) -> List[LabelObject]:
    """Read one input label file."""
    labels: List[LabelObject]
    ext = splitext(filename)[1]
    logger.info("Reading %s", filename)
    with open(filename, "r") as fp:
        if ext == ".json":
            labels = json.load(fp)
        elif ext in [".yml", ".yaml"]:
            labels = yaml.load(fp)
        else:
            raise ValueError("Unrecognized file extension {}".format(ext))
    return labels


def write_output(filename: str, labels: List[LabelObject]) -> None:
    """Write output file."""
    ext = splitext(filename)[1]
    logger.info("Writing %s", filename)
    with open(filename, "w") as fp:
        if ext == ".json":
            json.dump(labels, fp)
        elif ext in [".yml", ".yaml"]:
            yaml.dump(labels, fp)
        else:
            raise ValueError("Unrecognized file extension {}".format(ext))


def main() -> None:
    """Run."""
    args = parse_arguments()
    labels: List[LabelObject] = []
    for filename in args.input:
        labels.extend(read_input(filename))
    edit_frames(labels, args.add_url, args.remove_name_dir)
    print(labels[0]["name"])
    write_output(args.output, labels)


if __name__ == "__main__":
    main()
