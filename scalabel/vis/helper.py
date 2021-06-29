"""Helper functions used by the visualizer."""
import numpy as np

from ..common.typing import NDArrayF32


def random_color() -> NDArrayF32:
    """Generate a random color (RGB)."""
    return np.array(np.random.rand(3))
