"""Test parallel."""

from .parallel import pmap


def add(aa: int, bb: int) -> int:
    """Add two numbers."""
    return aa + bb


def test_pmap() -> None:
    """Test pmap."""
    res = pmap(add, zip(range(10), range(10, 20)), 2)
    assert len(res) == 10
    assert res[0] == 10
