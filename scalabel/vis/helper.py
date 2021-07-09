"""Helper functions used by the visualizer."""
import numpy as np

from ..common.typing import NDArrayF64


def random_color() -> NDArrayF64:
    """Generate a random color (RGB)."""
    return np.array(np.random.rand(3))
