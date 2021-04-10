"""Test io."""

from ..unittest.util import get_test_file
from . import io


def test_load_yaml_config() -> None:
    """Test load yaml config."""
    config = io.load_config(get_test_file("config.yaml"))
    assert config["hello"] == "world"


def test_load_json_config() -> None:
    """Test load yaml config."""
    config = io.load_config(get_test_file("config.json"))
    assert config["hello"] == "world"


def test_load_toml_config() -> None:
    """Test load toml config."""
    config = io.load_config(get_test_file("config.toml"))
    assert config["hello"] == "world"
