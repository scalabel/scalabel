"""Label io."""

import glob
import json
import os.path as osp
from functools import partial
from itertools import groupby
from typing import Any, List, Optional, Union

import humps

from ..common.io import load_config
from ..common.parallel import pmap
from ..common.typing import DictStrAny
from .typing import Box2D, Box3D, Config, Dataset, Frame, Label, Poly2D


def parse(raw_frame: DictStrAny, validate_frames: bool = True) -> Frame:
    """Parse a single frame."""
    if not validate_frames:
        frame = Frame.construct(**humps.decamelize(raw_frame))
        if frame.labels is not None:
            labels = []
            for l in frame.labels:
                # ignore the construct arguments in mypy
                label = Label.construct(**l)  # type: ignore
                if label.box2d is not None:
                    label.box2d = Box2D.construct(**label.box2d)  # type: ignore # pylint: disable=line-too-long
                if label.box3d is not None:
                    label.box3d = Box3D.construct(**label.box3d)  # type: ignore # pylint: disable=line-too-long
                if label.poly2d is not None:
                    label.poly2d = [
                        Poly2D.construct(**p) for p in label.poly2d  # type: ignore # pylint: disable=line-too-long
                    ]
                labels.append(label)
            frame.labels = labels
        return frame
    return Frame(**humps.decamelize(raw_frame))


def load(
    inputs: str, nprocs: int = 0, validate_frames: bool = True
) -> Dataset:
    """Load labels from a json file or a folder of json files."""
    raw_frames: List[DictStrAny] = []
    if not osp.exists(inputs):
        raise FileNotFoundError(f"{inputs} does not exist.")

    def process_file(filepath: str) -> Optional[DictStrAny]:
        raw_cfg = None
        with open(filepath, "r") as fp:
            content = json.load(fp)
        if isinstance(content, dict):
            raw_frames.extend(content["frames"])
            raw_cfg = content["config"]
        elif isinstance(content, list):
            raw_frames.extend(content)
        else:
            raise TypeError("The input file contains neither dict nor list.")
        return raw_cfg

    cfg = None
    if osp.isfile(inputs) and inputs.endswith("json"):
        ret_cfg = process_file(inputs)
        if ret_cfg is not None:
            cfg = ret_cfg
    elif osp.isdir(inputs):
        files = glob.glob(osp.join(inputs, "*.json"))
        for file_ in files:
            ret_cfg = process_file(file_)
            if cfg is None and ret_cfg is not None:
                cfg = ret_cfg
    else:
        raise TypeError("Inputs must be a folder or a JSON file.")

    config = None
    if cfg is not None:
        config = Config(**cfg)

    parse_ = partial(parse, validate_frames=validate_frames)
    if nprocs > 1:
        return Dataset(frames=pmap(parse_, raw_frames, nprocs), config=config)
    return Dataset(frames=list(map(parse_, raw_frames)), config=config)


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


def save(
    filepath: str, dataset: Union[List[Frame], Dataset], nprocs: int = 0
) -> None:
    """Save labels in Scalabel format."""
    if not isinstance(dataset, Dataset):
        dataset = Dataset(frames=dataset, config=None)
    dataset_dict = dataset.dict()

    if nprocs > 1:
        dataset_dict["frames"] = pmap(dump, dataset_dict["frames"], nprocs)
    else:
        dataset_dict["frames"] = list(map(dump, dataset_dict["frames"]))

    with open(filepath, "w") as fp:
        json.dump(dataset_dict, fp, indent=2)


def dump(frame: DictStrAny) -> DictStrAny:
    """Dump labels into dictionaries."""
    frame_str: DictStrAny = humps.camelize(remove_empty_elements(frame))
    return frame_str


def load_label_config(filepath: str) -> Config:
    """Load label configuration from a config file (toml / yaml)."""
    cfg = load_config(filepath)
    config = Config(**cfg)
    return config
