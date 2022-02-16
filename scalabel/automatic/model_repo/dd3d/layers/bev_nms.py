# Copyright 2021 Toyota Research Institute.  All rights reserved.
import logging

import numpy as np
import torch
from pytorch3d.transforms import transform3d as t3d

from detectron2.layers.nms import batched_nms_rotated
from detectron2.structures import RotatedBoxes

from ..structures.pose import Pose

LOG = logging.getLogger(__name__)

# yapf: disable
# -------------------------------
# Convention of reference frames.
# -------------------------------
# Rotation from "camera" frame to "vehicle" frame.
# |------------------|--------------------------------|
# | Camera | Vehicle | Interpretation in Vehicle frame|
# |------------------|--------------------------------|
# |   z    |   x     |            forward             |
# |   x    |  -y     |             right              |
# |   y    |  -z     |              down              |
# |------------------|--------------------------------|
CAMERA_TO_VEHICLE_ROTATION = Pose.from_matrix(np.float32([
    [ 0,  0,  1,  0],
    [-1,  0,  0,  0],
    [ 0, -1,  0,  0],
    [ 0,  0,  0,  1]
]))

# Rotation from "vehicle" frame to "bev" frame.
# |------------------|---------------------------------|
# | Vehicle |  BEV   | Interpretation in Vehicle frame |
# |------------------|---------------------------------|
# |   x     |  -y    |             forward             |
# |   y     |  -x    |               left              |
# |   z     |  -z    |                up               |
# |------------------|---------------------------------|
VEHICLE_TO_BEV_ROTATION = Pose.from_matrix(np.float32([
    [ 0, -1,  0,  0],
    [-1,  0,  0,  0],
    [ 0,  0, -1,  0],
    [ 0,  0,  0,  1]
]))
# yapf: enable


def boxes3d_to_rotated_boxes(
    boxes3d, pose_cam_global=CAMERA_TO_VEHICLE_ROTATION, pose_global_bev=VEHICLE_TO_BEV_ROTATION, use_top_surface=True
):
    """

    Parameters
    ----------
    boxes3d: Boxes3D
        3D boxes in camera frame.
    pose_cam_global: Pose
        Transformation from sensor (camera) frame to global frame. Depending on the context, global frame can be
        "vehicle" frame which moves along with the vehicle, or "world" frame which is fixed in the world.
        By default, it is an axis-swapping rotation that convert pinhole camera frame to Vehicle frame, i.e.
            x: forward, y: left, z: up (see above for detail.)
        with no translation (i.e. moves along with camera).
    pose_global_bev: Pose
        Transformation from global frame to bird-eye-view frame. By default, "forward" matches with "up" of BEV image,
        By default, it is an axis-swapping rotation that converts Vehicle frame to BEV frame (see above for detail.)
        with no translation.
    """
    if use_top_surface:
        vertice_inds = [0, 1, 5, 4]  # (front-left, front-right, back-right, back-left) of top surface.
    else:
        # use bottom surface.
        vertice_inds = [3, 2, 6, 7]  # (front-left, front-right, back-right, back-left) of bottom surface.

    surface = boxes3d.corners[:, vertice_inds, :]
    pose_cam_bev = pose_global_bev * pose_cam_global
    cam_to_bev = t3d.Transform3d(matrix=surface.new_tensor(pose_cam_bev.matrix.T))  # Need to transpose!
    # Assumpiton: this is close to rectangles. TODO: assert it?
    rot_boxes_bev = cam_to_bev.transform_points(surface)[:, :, :2]

    # length/width of objects are equivalent to "height"/width of RotatedBoxes
    length = torch.norm(rot_boxes_bev[:, 0, :] - rot_boxes_bev[:, 3, :], dim=1).abs()
    width = torch.norm(rot_boxes_bev[:, 0, :] - rot_boxes_bev[:, 1, :], dim=1).abs()

    center = torch.mean(rot_boxes_bev[:, [0, 2], :], dim=1)
    center_x, center_y = center[:, 0], center[:, 1]

    forward = rot_boxes_bev[:, 0, :] - rot_boxes_bev[:, 3, :]
    # CCW-angle, i.e. rotation wrt -z (or "up") in BEV frame.
    angle = torch.atan2(forward[:, 0], forward[:, 1])
    angle = 180.0 / np.pi * angle

    rot_boxes = RotatedBoxes(torch.stack([center_x, center_y, width, length, angle], dim=1))
    return rot_boxes


def bev_nms(
    boxes3d, scores, iou_threshold, pose_cam_global=CAMERA_TO_VEHICLE_ROTATION, class_idxs=None, class_agnostic=False
):
    """

    Parameters
    ----------
    boxes3d: Boxes3D
        3D boxes in camera frame.

    scores: Tensor
        1D score vector. Must be of same size 'boxes3d'

    iou_threshold: float
        Two rotated boxes in BEV frame cannot overlap (according to IoU) more than this threshold.

    class_idxs: Tensor or None
        If not None, 1D integer vector. Must be of same size 'boxes3d'

    class_agnostic: bool
        If True, then category ID is not considered in NMS.
        If False, then NMS is performed per-cateogry ('class_idxs' must not be None.)

    Returns
    -------
    keep: Tensor
        1D integer vector that contains filtered indices to 'boxes3d' to keep after NMS.
    """
    rot_boxes = boxes3d_to_rotated_boxes(boxes3d, pose_cam_global=pose_cam_global)
    if class_agnostic:
        class_idxs = torch.zeros_like(scores, dtype=torch.int64)
    else:
        assert class_idxs is not None
    keep = batched_nms_rotated(rot_boxes.tensor, scores, class_idxs, iou_threshold)
    return keep
