"""Label io."""

import glob
import json
import os.path as osp
from itertools import groupby
from typing import Any, List

import humps

from ..common.parallel import pmap
from ..common.typing import DictStrAny
from .typing import Frame


def load(inputs: str, nprocs: int = 0) -> List[Frame]:
    """Load labels from a json file or a folder of json files."""
    raw_frames: List[DictStrAny] = []
    if osp.isfile(inputs) and inputs.endswith("json"):
        with open(inputs, "r") as fp:
            raw_frames.extend(json.load(fp))
    elif osp.isdir(inputs):
        files = glob.glob(osp.join(inputs, "*.json"))
        for file_ in files:
            with open(file_, "r") as fp:
                raw_frames.extend(json.load(fp))
    else:
        raise TypeError("Inputs must be a folder or a JSON file.")

    def parse_frame(raw_frame: DictStrAny) -> Frame:
        """Parse a single frame."""
        return Frame(**humps.decamelize(raw_frame))

    if nprocs > 0:
        return pmap(parse_frame, raw_frames, nprocs)
    return list(map(parse_frame, raw_frames))


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
