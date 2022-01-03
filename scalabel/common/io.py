"""Common io helper functions."""

import json
import os
from typing import List, TextIO

import toml
import yaml

from .typing import DictStrAny


def open_read_text(filepath: str) -> TextIO:
    """Open a text file for reading and return a file object."""
    return open(filepath, mode="r", encoding="utf-8")


def open_write_text(filepath: str) -> TextIO:
    """Open a text file for writing and return a file object."""
    return open(filepath, mode="w", encoding="utf-8")


def load_config(filepath: str) -> DictStrAny:
    """Read config file.

    The config file can be in yaml, json or toml.
    toml is recommended for readability.
    """
    ext = os.path.splitext(filepath)[1]
    if ext in [".yaml", ".yml"]:
        with open_read_text(filepath) as fp:
            config_dict = yaml.load(fp.read(), Loader=yaml.Loader)
    elif ext == ".toml":
        config_dict = toml.load(filepath)
    elif ext == ".json":
        with open_read_text(filepath) as fp:
            config_dict = json.load(fp)
    else:
        raise NotImplementedError(f"Config extension {ext} not supported")
    assert isinstance(config_dict, dict)
    return config_dict


def load_file_as_list(filepath: str) -> List[str]:
    """Get contents of a text file as list."""
    with open_read_text(filepath) as f:
        contents = f.readlines()
    return contents
