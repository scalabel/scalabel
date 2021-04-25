"""Helper functions used by the visualizer."""

import numpy as np

from scalabel.label.typing import Intrinsics


def get_intrinsic_matrix(intrinsics: Intrinsics) -> np.ndarray:
    """Get the camera intrinsic matrix."""
    calibration = np.identity(3)
    calibration[0, 2] = intrinsics.center[0]
    calibration[1, 2] = intrinsics.center[1]
    calibration[0, 0] = intrinsics.focal[0]
    calibration[1, 1] = intrinsics.focal[1]
    return calibration


def random_color() -> np.ndarray:
    """Generate a random color (RGB)."""
    return np.array(np.random.rand(3))
