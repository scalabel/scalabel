"""Test parallel."""
from typing import Tuple

from .parallel import pmap


def add(num: Tuple[int, int]) -> int:
    """Add two numbers."""
    return num[0] + num[1]


def test_pmap() -> None:
    """Test pmap."""
    res = pmap(add, zip(range(10), range(10, 20)), 2)
    assert len(res) == 10
    assert res[0] == 10
