"""Label io."""

import glob
import json
import os.path as osp
from itertools import groupby
from typing import Any, List, Union

import humps

from ..common.typing import DictStrAny
from .typing import Frame


def load(filepath: str) -> List[Frame]:
    """Load labels from a file."""
    return parse(json.load(open(filepath, "r")))


def parse(raw_frames: Union[str, List[DictStrAny], DictStrAny]) -> List[Frame]:
    """Load labels in Scalabel format."""
    if isinstance(raw_frames, str):
        raw_frames = json.loads(raw_frames)
    if isinstance(raw_frames, dict):
        raw_frames = [raw_frames]
    frames: List[Frame] = []
    for rf in raw_frames:
        f = humps.decamelize(rf)
        frames.append(Frame(**f))
    return frames


def read(inputs: str) -> List[Frame]:
    """Read annotations from file or files. More general than `load`."""
    outputs: List[Frame] = []
    if osp.isdir(inputs):
        files = glob.glob(osp.join(inputs, "*.json"))
        for file_ in files:
            outputs.extend(load(file_))
    elif osp.isfile(inputs) and inputs.endswith("json"):
        outputs.extend(load(inputs))
    else:
        raise TypeError("Inputs must be a folder or a JSON file.")

    outputs = sorted(outputs, key=lambda output: output.name)
    return outputs


def group_and_sort(inputs: List[Frame]) -> List[List[Frame]]:
    """Group frames by video_name and sort."""
    for frame in inputs:
        assert frame.video_name is not None
        assert frame.frame_index is not None
    frames_list: List[List[Frame]] = []
    for _, frame_iter in groupby(inputs, lambda frame: frame.video_name):
        frames = sorted(
            list(frame_iter),
            key=lambda frame: frame.frame_index if frame.frame_index else 0,
        )
        frames_list.append(frames)
    frames_list = sorted(
        frames_list, key=lambda frames: str(frames[0].video_name)
    )
    return frames_list


def remove_empty_elements(frame: DictStrAny) -> DictStrAny:
    """Recursively remove empty lists, empty dicts, or None elements."""

    def empty(element: Any) -> bool:  # type: ignore
        return element is None or element == {} or element == []

    if not isinstance(frame, (dict, list)):
        return frame
    if isinstance(frame, list):
        return [
            v
            for v in (remove_empty_elements(v) for v in frame)
            if not empty(v)
        ]
    return {
        k: v
        for k, v in ((k, remove_empty_elements(v)) for k, v in frame.items())
        if not empty(v)
    }


def save(filepath: str, frames: List[Frame]) -> None:
    """Save labels in Scalabel format."""
    labels = dump(frames)
    with open(filepath, "w") as fp:
        json.dump(labels, fp, indent=2)


def dump(frames: List[Frame]) -> List[DictStrAny]:
    """Dump labels into dictionaries."""
    return [humps.camelize(remove_empty_elements(f.dict())) for f in frames]
