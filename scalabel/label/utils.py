"""Utility functions for label."""
from typing import Dict, List

import numpy as np
from scipy.spatial.transform import Rotation

from .typing import Category, Extrinsics, Intrinsics, Label


def get_intrinsics_from_matrix(matrix: np.ndarray) -> Intrinsics:
    """Get intrinsics data structure from 3x3 matrix."""
    intrinsics = Intrinsics(
        focal=(matrix[0, 0], matrix[1, 1]),
        center=(matrix[0, 2], matrix[1, 2]),
        skew=matrix[0, 1],
    )
    return intrinsics


def get_matrix_from_intrinsics(intrinsics: Intrinsics) -> np.ndarray:
    """Get the camera intrinsic matrix."""
    calibration = np.identity(3)
    calibration[0, 2] = intrinsics.center[0]
    calibration[1, 2] = intrinsics.center[1]
    calibration[0, 0] = intrinsics.focal[0]
    calibration[1, 1] = intrinsics.focal[1]
    calibration[0, 1] = intrinsics.skew
    return calibration


def get_extrinsics_from_matrix(matrix: np.ndarray) -> Extrinsics:
    """Get extrinsics data structure from 4x4 matrix."""
    extrinsics = Extrinsics(
        location=(matrix[0, -1], matrix[1, -1], matrix[2, -1]),
        rotation=tuple(
            Rotation.from_matrix(matrix[:3, :3]).as_euler("xyz").tolist()
        ),
    )
    return extrinsics


def get_matrix_from_extrinsics(extrinsics: Extrinsics) -> np.ndarray:
    """Convert Extrinsics class object to rotation matrix."""
    rot_mat = Rotation.from_euler("xyz", extrinsics.rotation).as_matrix()
    translation = np.array(extrinsics.location)
    extrinsics_mat = np.identity(4)
    extrinsics_mat[:3, :3] = rot_mat
    extrinsics_mat[:3, -1] = translation
    return extrinsics_mat


def get_leaf_categories(parent_categories: List[Category]) -> List[Category]:
    """Get the leaf categories in the category tree structure."""
    result = []
    for category in parent_categories:
        if category.subcategories is None:
            result.append(category)
        else:
            result.extend(get_leaf_categories(category.subcategories))

    return result


def get_parent_categories(
    parent_categories: List[Category],
) -> Dict[str, List[Category]]:
    """Get all parent categories and their associated leaf categories."""
    result = {}
    for category in parent_categories:
        if category.subcategories is not None:
            result.update(get_parent_categories(category.subcategories))
            result[category.name] = get_leaf_categories([category])
        else:
            return {}
    return result


def check_crowd(label: Label) -> bool:
    """Check crowd attribute."""
    if label.attributes is not None:
        crowd = bool(label.attributes.get("crowd", False))
    else:
        crowd = False
    return crowd


def check_ignored(label: Label) -> bool:
    """Check ignored attribute."""
    if label.attributes is not None:
        ignored = bool(label.attributes.get("ignored", False))
    else:
        ignored = False
    return ignored
