"""Common io helper functions."""

import json
import os
from typing import List

import toml
import yaml

from .typing import DictStrAny


def load_config(filepath: str) -> DictStrAny:
    """Read config file.

    The config file can be in yaml, json or toml.
    toml is recommended for readability.
    """
    ext = os.path.splitext(filepath)[1]
    if ext == ".yaml":
        with open(filepath, "r") as fp:
            config_dict = yaml.load(fp.read(), Loader=yaml.Loader)
    elif ext == ".toml":
        config_dict = toml.load(filepath)
    elif ext == ".json":
        with open(filepath, "r") as fp:
            config_dict = json.load(fp)  # type: ignore
    else:
        raise NotImplementedError(f"Config extention {ext} not supported")
    assert isinstance(config_dict, dict)
    return config_dict


def load_file_as_list(filepath: str) -> List[str]:
    """Get contents of a text file as list."""
    with open(filepath, "r") as f:
        contents = f.readlines()
    return contents
