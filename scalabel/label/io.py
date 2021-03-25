"""Label io."""

import json
from typing import Any, List, Union

import humps

from .typing import DictStrAny, Frame


def load(filepath: str, use_obj_prarser: bool = False) -> List[Frame]:
    """Load labels from a file."""
    json_obj = json.load(open(filepath, "r"))
    if use_obj_prarser:
        return parse_obj(json_obj)
    else:
        return parse(json_obj)


def parse(raw_frames: Union[str, List[DictStrAny], DictStrAny]) -> List[Frame]:
    """Load labels in Scalabel format."""
    if isinstance(raw_frames, str):
        raw_frames = json.loads(raw_frames)
    if isinstance(raw_frames, dict):
        raw_frames = [raw_frames]
    # print(raw_frames[0]["labels"][0])
    frames: List[Frame] = []
    for rf in raw_frames:
        f = humps.decamelize(rf)
        # print(f)
        frames.append(Frame(**f))
    return frames


def parse_obj(raw_frames: List[DictStrAny]) -> List[Frame]:
    """Load labels in Scalabel format using Pydantic's own parser"""
    frames: List[Frame] = []
    for rf in raw_frames:
        frames.append(
            Frame.parse_obj(rf)
        )  # This will parse structures recursively.
    return frames


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
