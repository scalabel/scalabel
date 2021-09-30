"""Utility functions for label."""
import math
from typing import Dict, List, Tuple

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ..common.typing import NDArrayF64
from .typing import Category, Extrinsics, Frame, Intrinsics, Label


def get_intrinsics_from_matrix(matrix: NDArrayF64) -> Intrinsics:
    """Get intrinsics data structure from 3x3 matrix."""
    intrinsics = Intrinsics(
        focal=(matrix[0, 0], matrix[1, 1]),
        center=(matrix[0, 2], matrix[1, 2]),
        skew=matrix[0, 1],
    )
    return intrinsics


def get_matrix_from_intrinsics(intrinsics: Intrinsics) -> NDArrayF64:
    """Get the camera intrinsic matrix."""
    calibration = np.identity(3)
    calibration[0, 2] = intrinsics.center[0]
    calibration[1, 2] = intrinsics.center[1]
    calibration[0, 0] = intrinsics.focal[0]
    calibration[1, 1] = intrinsics.focal[1]
    calibration[0, 1] = intrinsics.skew
    return calibration


def get_extrinsics_from_matrix(matrix: NDArrayF64) -> Extrinsics:
    """Get extrinsics data structure from 4x4 matrix."""
    extrinsics = Extrinsics(
        location=(matrix[0, -1], matrix[1, -1], matrix[2, -1]),
        rotation=tuple(
            Rotation.from_matrix(matrix[:3, :3]).as_euler("xyz").tolist()
        ),
    )
    return extrinsics


def get_matrix_from_extrinsics(extrinsics: Extrinsics) -> NDArrayF64:
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


def check_occluded(label: Label) -> bool:
    """Check occluded attribute."""
    if label.attributes is not None:
        occluded = bool(label.attributes.get("occluded", False))
    else:
        occluded = False
    return occluded


def check_truncated(label: Label) -> bool:
    """Check truncated attribute."""
    if label.attributes is not None:
        truncated = bool(label.attributes.get("truncated", False))
    else:
        truncated = False
    return truncated


def cart2hom(pts_3d: NDArrayF64) -> NDArrayF64:
    """Nx3 points in Cartesian to Homogeneous by appending ones."""
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
    return pts_3d_hom


def project_points_to_image(
    points: NDArrayF64, intrinsics: NDArrayF64
) -> NDArrayF64:
    """Project Nx3 points to Nx2 pixel coordinates with 3x3 intrinsics."""
    hom_cam_coords = points / points[:, 2:3]
    pts_2d = np.dot(hom_cam_coords, np.transpose(intrinsics))
    return pts_2d[:, :2]  # type: ignore


def rotation_y_to_alpha(
    rotation_y: float, center: Tuple[float, float, float]
) -> float:
    """Convert rotation around y-axis to viewpoint angle (alpha)."""
    alpha = rotation_y - math.atan2(center[0], center[2])
    if alpha > math.pi:
        alpha -= 2 * math.pi
    if alpha <= -math.pi:
        alpha += 2 * math.pi
    return alpha


def compare_results(result: List[Frame], result_compare: List[Frame]) -> None:
    """Compare two list of frames."""
    for frame, frame_ref in zip(result, result_compare):
        assert frame.name == frame_ref.name
        assert frame.videoName == frame_ref.videoName
        assert frame.frameIndex == frame_ref.frameIndex
        if frame.intrinsics is not None:
            assert frame_ref.intrinsics is not None
            assert frame.intrinsics.focal == pytest.approx(
                frame_ref.intrinsics.focal
            )
            assert frame.intrinsics.center == pytest.approx(
                frame_ref.intrinsics.center
            )
            assert frame.intrinsics.skew == pytest.approx(
                frame_ref.intrinsics.skew
            )
        else:
            assert frame.intrinsics == frame_ref.intrinsics
        if frame.extrinsics is not None:
            assert frame_ref.extrinsics is not None
            assert frame.extrinsics.location == pytest.approx(
                frame_ref.extrinsics.location
            )
            assert frame.extrinsics.rotation == pytest.approx(
                frame_ref.extrinsics.rotation
            )
        else:
            assert frame.extrinsics == frame_ref.extrinsics

        if frame.labels is not None:
            assert frame_ref.labels is not None
            for label, label_ref in zip(frame.labels, frame_ref.labels):
                assert label.id == label_ref.id
                assert label.category == label_ref.category
                if label.box2d is not None:
                    assert label_ref.box2d is not None
                    assert label.box2d.x1 == pytest.approx(label_ref.box2d.x1)
                    assert label.box2d.y1 == pytest.approx(label_ref.box2d.y1)
                    assert label.box2d.x2 == pytest.approx(label_ref.box2d.x2)
                    assert label.box2d.y2 == pytest.approx(label_ref.box2d.y2)
                else:
                    assert label.box2d == label_ref.box2d

                if label.box3d is not None:
                    assert label_ref.box3d is not None
                    assert label.box3d.location == pytest.approx(
                        label_ref.box3d.location
                    )
                    assert label.box3d.dimension == pytest.approx(
                        label_ref.box3d.dimension
                    )
                    assert label.box3d.orientation == pytest.approx(
                        label_ref.box3d.orientation
                    )
                else:
                    assert label.box3d == label_ref.box3d
        else:
            assert frame.labels == frame_ref.labels
