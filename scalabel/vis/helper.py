import numpy as np

from scalabel.label.typing import Box2D, Box3D, Intrinsics


def get_intrinsic_matrix(intrinsics: Intrinsics):
    calibration = np.identity(3)
    calibration[0, 2] = intrinsics.center[0]
    calibration[1, 2] = intrinsics.center[1]
    calibration[0, 0] = intrinsics.focal[0]
    calibration[1, 1] = intrinsics.focal[1]
    return calibration


def random_color():
    return np.random.rand(3)
