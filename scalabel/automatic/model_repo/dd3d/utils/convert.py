from pyquaternion import Quaternion
from ..structures.pose import Pose
import numpy as np


def convert_3d_box_to_kitti(box):
    """Convert a single 3D bounding box (GenericBoxes3D) to KITTI convention. i.e. for evaluation. We
    assume the box is in the reference frame of camera_2 (annotations are given in this frame).

    Usage:
        >>> box_camera_2 = pose_02.inverse() * pose_0V * box_velodyne
        >>> kitti_bbox_params = convert_3d_box_to_kitti(box_camera_2)

    Parameters
    ----------
    box: GenericBoxes3D
        Box in camera frame (X-right, Y-down, Z-forward)

    Returns
    -------
    W, L, H, x, y, z, rot_y, alpha: float
        KITTI format bounding box parameters.
    """
    assert len(box) == 1

    quat = Quaternion(*box.quat.cpu().tolist()[0])
    tvec = box.tvec.cpu().numpy()[0]
    sizes = box.size.cpu().numpy()[0]

    # Re-encode into KITTI box convention
    # Translate y up by half of dimension
    tvec += np.array([0.0, sizes[2] / 2.0, 0])

    inversion = Quaternion(axis=[1, 0, 0], radians=np.pi / 2).inverse
    quat = inversion * quat

    # Construct final pose in KITTI frame (use negative of angle if about positive z)
    if quat.axis[2] > 0:
        kitti_pose = Pose(wxyz=Quaternion(axis=[0, 1, 0], radians=-quat.angle), tvec=tvec)
        rot_y = -quat.angle
    else:
        kitti_pose = Pose(wxyz=Quaternion(axis=[0, 1, 0], radians=quat.angle), tvec=tvec)
        rot_y = quat.angle

    # Construct unit vector pointing in z direction (i.e. [0, 0, 1] direction)
    # The transform this unit vector by pose of car, and drop y component, thus keeping heading direction in BEV (x-z grid)
    v_ = np.float64([[0, 0, 1], [0, 0, 0]])
    v_ = (kitti_pose * v_)[:, ::2]

    # Getting positive theta angle (we define theta as the positive angle between
    # a ray from the origin through the base of the transformed unit vector and the z-axis
    theta = np.arctan2(abs(v_[1, 0]), abs(v_[1, 1]))

    # Depending on whether the base of the transformed unit vector is in the first or
    # second quadrant we add or subtract `theta` from `rot_y` to get alpha, respectively
    alpha = rot_y + theta if v_[1, 0] < 0 else rot_y - theta
    # Bound from [-pi, pi]
    if alpha > np.pi:
        alpha -= 2.0 * np.pi
    elif alpha < -np.pi:
        alpha += 2.0 * np.pi
    alpha = np.around(alpha, decimals=2)  # KITTI precision

    # W, L, H, x, y, z, rot-y, alpha
    return sizes[0], sizes[1], sizes[2], tvec[0], tvec[1], tvec[2], rot_y, alpha
