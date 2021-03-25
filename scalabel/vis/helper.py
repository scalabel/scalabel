"""
Helper functions used by the visualizer.
"""

import os
import sys
import numpy as np

from scalabel.label.typing import Intrinsics


def get_intrinsic_matrix(intrinsics: Intrinsics) -> np.ndarray:
    """Get the camera intrinsic matrix"""
    calibration = np.identity(3)
    calibration[0, 2] = intrinsics.center[0]
    calibration[1, 2] = intrinsics.center[1]
    calibration[0, 0] = intrinsics.focal[0]
    calibration[1, 1] = intrinsics.focal[1]
    return calibration


def random_color() -> np.ndarray:
    """Generate a random color (RGB)"""
    return np.random.rand(3)


def is_valid_file(parser, file_name):
    """Ensure that the file exists."""

    if not os.path.exists(file_name):
        parser.error(
            "The corresponding bounding box annotation '{}' does "
            "not exist!".format(file_name)
        )
        sys.exit(1)
