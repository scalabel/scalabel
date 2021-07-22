"""Functions for KITTI."""
import os
from typing import List

import numpy as np
import utm  # type: ignore

from ..common.typing import NDArrayF64


def angle2rot(rotation: NDArrayF64, inverse: bool = False) -> NDArrayF64:
    """Transform eular to rotation matrix.

    Args:
        rotation : rotation along X, Y, Z
        inverse: rotate order

    Returns:
        rotation matrix
    """
    return rotate(np.eye(3), rotation, inverse=inverse)


def rot_axis(angle: NDArrayF64, axis: int) -> NDArrayF64:
    """Rotation matrices around the X (gamma), Y (beta), and Z (alpha) axis.

    Input:
        angle: one of [gamma, beta, alpha]
        axis: one of [X, Y, Z]
    Output:
        rotation matrix

    r_x = np.array([ [1,             0,              0],
                    [0, np.cos(gamma), -np.sin(gamma)],
                    [0, np.sin(gamma),  np.cos(gamma)]])

    r_y = np.array([ [ np.cos(beta), 0, np.sin(beta)],
                    [            0, 1,            0],
                    [-np.sin(beta), 0, np.cos(beta)]])

    r_z = np.array([ [np.cos(alpha), -np.sin(alpha), 0],
                    [np.sin(alpha),  np.cos(alpha), 0],
                    [            0,              0, 1]])
    """
    cg = np.cos(angle)
    sg = np.sin(angle)
    if axis == 0:  # X
        v = [0, 4, 5, 7, 8]
    elif axis == 1:  # Y
        v = [4, 0, 6, 2, 8]
    else:  # Z
        v = [8, 0, 1, 3, 4]
    rot = np.zeros(9)
    rot[v[0]] = 1.0
    rot[v[1]] = cg
    rot[v[2]] = -sg
    rot[v[3]] = sg
    rot[v[4]] = cg
    return rot.reshape(3, 3)


def rotate(
    vector: NDArrayF64, angle: NDArrayF64, inverse: bool = False
) -> NDArrayF64:
    """Rotation of x, y, z axis.

    Forward rotate order: Z, Y, X
    Inverse rotate order: X^T, Y^T, Z^T

    Input:
        vector: vector in 3D coordinates
        angle: rotation along X, Y, Z
        inverse: rotate order
    Output:
        out: rotated vector
    """
    gamma, beta, alpha = angle[0], angle[1], angle[2]

    # Rotation matrices around the X (gamma), Y (beta), and Z (alpha) axis
    r_x = rot_axis(gamma, 0)
    r_y = rot_axis(beta, 1)
    r_z = rot_axis(alpha, 2)

    # Composed rotation matrix
    if inverse:
        rot_mat = np.dot(np.dot(np.dot(r_x.T, r_y.T), r_z.T), vector)
    else:
        rot_mat = np.dot(np.dot(np.dot(r_z, r_y), r_x), vector)

    return rot_mat  # type: ignore


# Functions from kio_slim
class KittiPoseParser:
    """Calibration matrices in KITTI."""

    def __init__(self, fields: List[str]) -> None:
        """Init parameters and set pose with fields."""
        self.latlon: List[float]
        self.position: NDArrayF64
        self.roll: float
        self.pitch: float
        self.yaw: float
        self.rotation: NDArrayF64

        if fields is not None:
            self.set_oxt(fields)

    def set_oxt(self, fields_str: List[str]) -> None:
        """Assign the pose information from corresponding fields.

        Input:
            fields: list of oxts information
        """
        fields = [float(f) for f in fields_str]
        self.latlon = fields[:2]
        location = utm.from_latlon(*self.latlon)
        self.position = np.array([location[0], location[1], fields[2]])

        self.roll = fields[3]
        self.pitch = fields[4]
        self.yaw = fields[5]
        rotation = angle2rot(np.array([self.roll, self.pitch, self.yaw]))
        imu_to_camera = angle2rot(
            np.array([np.pi / 2, -np.pi / 2, 0]), inverse=True
        )
        self.rotation = rotation.dot(imu_to_camera)


def read_oxts(oxts_dir: str, seq_idx: int) -> List[List[str]]:
    """Read oxts file and return each fields for KittiPoseParser.

    Input:
        oxts_dir: path of oxts file
        seq_idx: index of the sequence
    Output:
        fields: list of oxts information
    """
    oxts_path = os.path.join(oxts_dir, f"{seq_idx:04d}.txt")
    with open(oxts_path, "r") as f:
        fields = [line.strip().split() for line in f]
    return fields


def read_calib(calib_dir: str, seq_idx: int, cam: int = 2) -> NDArrayF64:
    """Read calibration file and return camera matrix.

    e.g.,
        projection = read_calib(cali_dir, vid_id)
    """
    with open(os.path.join(calib_dir, f"{seq_idx:04d}.txt")) as f:
        fields = [line.split() for line in f]
    return np.asarray(fields[cam][1:], dtype=np.float32).reshape(3, 4)


def read_calib_det(calib_dir: str, img_idx: int, cam: int = 2) -> NDArrayF64:
    """Read calibration file and return camera matrix.

    e.g.,
        projection = read_calib(cali_dir, img_id)
    """
    with open(os.path.join(calib_dir, f"{img_idx:06d}.txt")) as f:
        fields = [line.split() for line in f]
    return np.asarray(fields[cam][1:], dtype=np.float32).reshape(3, 4)


def list_from_file(
    filename: str, prefix: str = "", offset: int = 0, max_num: int = 0
) -> List[str]:
    """Load a text file and parse the content as a list of strings.

    Args:
        filename (str): Filename.
        prefix (str): The prefix to be inserted to the begining of each item.
        offset (int): The offset of lines.
        max_num (int): The maximum number of lines to be read,
            zeros and negatives mean no limitation.

    Returns:
        list[str]: A list of strings.
    """
    cnt = 0
    item_list = []
    with open(filename, "r") as f:
        for _ in range(offset):
            f.readline()
        for line in f:
            if cnt >= max_num > 0:
                break
            item_list.append(prefix + line.rstrip("\n"))
            cnt += 1
    return item_list
