"""Functions for KITTI."""
import os
from typing import List

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..common.io import open_read_text
from ..common.typing import NDArrayF64

try:
    import utm
except ImportError:
    utm = None


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
        assert utm is not None, (
            "KITTI conversion requires utm to be "
            "installed, see scripts/optional.txt"
        )
        location = utm.from_latlon(*self.latlon)
        self.position = np.array([location[0], location[1], fields[2]])

        self.roll = fields[3]
        self.pitch = fields[4]
        self.yaw = fields[5]
        rotation = R.from_euler(
            "xyz", np.array([self.roll, self.pitch, self.yaw])
        ).as_matrix()
        imu_to_camera = (
            R.from_euler(
                "xyz", np.array([np.pi / 2, -np.pi / 2, 0])
            ).as_matrix()
        ).T
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
    with open_read_text(oxts_path) as f:
        fields = [line.strip().split() for line in f]
    return fields


def read_calib(calib_dir: str, seq_idx: int, cam: int = 2) -> NDArrayF64:
    """Read calibration file and return camera matrix.

    e.g.,
        projection = read_calib(cali_dir, vid_id)
    """
    with open_read_text(os.path.join(calib_dir, f"{seq_idx:04d}.txt")) as f:
        fields = [line.split() for line in f]
    return np.asarray(fields[cam][1:], dtype=np.float32)


def read_calib_det(calib_dir: str, img_idx: int, cam: int = 2) -> NDArrayF64:
    """Read calibration file and return camera matrix.

    e.g.,
        projection = read_calib(cali_dir, img_id)
    """
    with open_read_text(os.path.join(calib_dir, f"{img_idx:06d}.txt")) as f:
        fields = [line.split() for line in f]
    return np.asarray(fields[cam][1:], dtype=np.float64)


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
    with open_read_text(filename) as f:
        for _ in range(offset):
            f.readline()
        for line in f:
            if cnt >= max_num > 0:
                break
            item_list.append(prefix + line.rstrip("\n"))
            cnt += 1
    return item_list
