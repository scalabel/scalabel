"""3D geometric functions used by the visualizer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import numpy as np

from scalabel.label.typing import Box3D


def rotate_vector(
    vector: np.ndarray,
    rot_x: float = 0,
    rot_y: float = 0,
    rot_z: float = 0,
    center: Union[np.ndarray, None] = None,
) -> np.ndarray:
    """Rotate a vector given axis-angles."""
    if center is not None:
        vector -= center
    x0, y0, z0 = vector.tolist()
    cos_ry, sin_ry = [np.cos(rot_y), np.sin(rot_y)]
    x1, y1, z1 = [sin_ry * z0 + cos_ry * x0, y0, cos_ry * z0 - sin_ry * x0]
    cos_rz, sin_rz = [np.cos(rot_z), np.sin(rot_z)]
    x2, y2, z2 = [cos_rz * x1 - sin_rz * y1, sin_rz * x1 + cos_rz * y1, z1]
    cos_rx, sin_rx = [np.cos(rot_x), np.sin(rot_x)]
    x3, y3, z3 = [x2, cos_rx * y2 - sin_rx * z2, sin_rx * y2 + cos_rx * z2]
    vector = np.array([x3, y3, z3])
    if center is not None:
        vector += center
    return vector


def vector_3d_to_2d(
    vector: np.ndarray, calibration: np.ndarray
) -> List[float]:
    """Project 3d vector to the 2d camera view."""
    vec_3d = np.ones(3)
    vec_3d[:3] = vector
    vec_2d = np.dot(calibration, vec_3d)
    # print(calibration, vec_3d, vec_2d)
    return [vec_2d[0] / vec_2d[2], vec_2d[1] / vec_2d[2]]


def check_side_of_line(
    point: np.ndarray, line: Tuple[np.ndarray, np.ndarray]
) -> int:
    """Chece which side does a point locate."""
    p1, p2 = line
    det = (point[0] - p1[0]) * (p2[1] - p1[1]) - (point[1] - p1[1]) * (
        p2[0] - p1[0]
    )
    return int(np.sign(det))


def check_clockwise(points: List[np.ndarray]) -> int:
    """Check whether the 4 points in a clockwise order."""
    p1, p2, p3, p4 = points
    s1 = check_side_of_line(p3, (p1, p2))
    s2 = check_side_of_line(p4, (p2, p3))
    s3 = check_side_of_line(p1, (p3, p4))
    s4 = check_side_of_line(p2, (p4, p1))
    if s1 == s2 == s3 == s4:
        return int(s1)
    return 0


@dataclass
class Vertex:
    """Calss for 3D vertex."""

    v3d: List[float]
    v2d: List[float]

    def __init__(self, vector: np.ndarray, calibration: np.ndarray) -> None:
        """Init the vector."""
        self.v3d = vector.tolist()
        self.v2d = vector_3d_to_2d(vector, calibration)


class Label3d:
    """Generate the 2D edges of a 3D bounding box."""

    def __init__(self, vertices: List[np.ndarray]) -> None:
        """Init the vector."""
        self.vertices = vertices

    @classmethod
    def from_box3d(cls, box3d: Box3D) -> Label3d:
        """Get 8 vertex points of a 3D bounding box."""
        x, y, z = box3d.location
        center = np.array([x, y, z])
        height, width, depth = np.array(box3d.dimension)

        def rotate(vector: np.ndarray) -> np.ndarray:
            if len(box3d.orientation) == 3:
                rot_x, rot_y, rot_z = box3d.orientation
                rotated = rotate_vector(vector, rot_x, rot_y, rot_z, center)
            else:
                rot_y = box3d.orientation[1]
                rotated = rotate_vector(
                    vector, 0, rot_y + np.pi / 2, 0, center
                )
            return rotated

        v000 = rotate(center + np.array([-width / 2, -height / 2, -depth / 2]))
        v001 = rotate(center + np.array([-width / 2, -height / 2, depth / 2]))
        v010 = rotate(center + np.array([-width / 2, height / 2, -depth / 2]))
        v011 = rotate(center + np.array([-width / 2, height / 2, depth / 2]))
        v100 = rotate(center + np.array([width / 2, -height / 2, -depth / 2]))
        v101 = rotate(center + np.array([width / 2, -height / 2, depth / 2]))
        v110 = rotate(center + np.array([width / 2, height / 2, -depth / 2]))
        v111 = rotate(center + np.array([width / 2, height / 2, depth / 2]))
        return cls([v000, v001, v010, v011, v100, v101, v110, v111])

    def get_edges_with_visibility(
        self, calibration: np.ndarray
    ) -> Dict[str, List[List[List[float]]]]:
        """Get edges with visibility."""
        vertices = [Vertex(v, calibration) for v in self.vertices]
        v000, v001, v010, v011, v100, v101, v110, v111 = vertices

        edges = {
            "FU": [v000, v100],
            "FR": [v100, v110],
            "FD": [v010, v110],
            "FL": [v000, v010],
            "MUL": [v000, v001],
            "MUR": [v100, v101],
            "MDR": [v110, v111],
            "MDL": [v010, v011],
            "BU": [v001, v101],
            "BR": [v101, v111],
            "BD": [v011, v111],
            "BL": [v001, v011],
        }

        faces: Dict[str, Dict[str, Union[List[str], List[Vertex]]]] = {
            "F": {
                "v": [v000, v100, v110, v010],
                "e": ["FU", "FR", "FD", "FL"],
            },
            "B": {
                "v": [v101, v001, v011, v111],
                "e": ["BU", "BR", "BD", "BL"],
            },
            "L": {
                "v": [v001, v000, v010, v011],
                "e": ["FL", "MUL", "BL", "MDL"],
            },
            "R": {
                "v": [v100, v101, v111, v110],
                "e": ["FR", "MUR", "BR", "MDR"],
            },
            "U": {
                "v": [v001, v101, v100, v000],
                "e": ["FU", "MUR", "BU", "MUL"],
            },
            "D": {
                "v": [v010, v110, v111, v011],
                "e": ["FD", "MDR", "BD", "MDL"],
            },
        }

        face_pairs = ["FB", "LR", "UD"]

        dashed_edges = {
            "FU": True,
            "FR": True,
            "FD": True,
            "FL": True,
            "MUL": True,
            "MUR": True,
            "MDR": True,
            "MDL": True,
            "BU": True,
            "BR": True,
            "BD": True,
            "BL": True,
        }
        for pair in face_pairs:
            face1: str = pair[0]
            face2: str = pair[1]
            cw1 = check_clockwise(
                [
                    np.array(v.v2d)
                    for v in faces[face1]["v"]
                    if isinstance(v, Vertex)
                ]
            )
            cw2 = check_clockwise(
                [
                    np.array(v.v2d)
                    for v in faces[face2]["v"]
                    if isinstance(v, Vertex)
                ]
            )
            if cw1 != cw2:
                vertices1 = np.array(
                    [v.v3d for v in faces[face1]["v"] if isinstance(v, Vertex)]
                )
                vertices2 = np.array(
                    [v.v3d for v in faces[face2]["v"] if isinstance(v, Vertex)]
                )
                dist1 = np.linalg.norm(np.median(vertices1, axis=0))
                dist2 = np.linalg.norm(np.median(vertices2, axis=0))
                solid_face = face1 if dist1 < dist2 else face2
                for edge in faces[solid_face]["e"]:
                    assert isinstance(edge, str)
                    dashed_edges[edge] = False

        edges_with_visibility: Dict[str, List[List[List[float]]]] = {
            "dashed": [],
            "solid": [],
        }
        for edge in edges:
            if dashed_edges[edge]:
                edges_with_visibility["dashed"].append(
                    [v.v2d for v in edges[edge]]
                )
            else:
                edges_with_visibility["solid"].append(
                    [v.v2d for v in edges[edge]]
                )
        return edges_with_visibility
