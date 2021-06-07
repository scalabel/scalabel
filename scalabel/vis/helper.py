"""Helper functions used by the visualizer."""
import numpy as np


def random_color() -> np.ndarray:
    """Generate a random color (RGB)."""
    return np.array(np.random.rand(3))
