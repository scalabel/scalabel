"""Helper functions used by the visualizer."""
import numpy as np

from ..common.typing import FloatArray


def random_color() -> FloatArray:
    """Generate a random color (RGB)."""
    return np.array(np.random.rand(3))
