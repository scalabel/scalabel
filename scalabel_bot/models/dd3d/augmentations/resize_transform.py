# Copyright 2021 Toyota Research Institute.  All rights reserved.
# pylint: disable=unused-argument
import cv2
import numpy as np

from detectron2.data.transforms import ResizeShortestEdge as _ResizeShortestEdge
from detectron2.data.transforms import ResizeTransform

CV2_INTERPOLATION_MODES = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC}
DEFAULT_DEPTH_INTERPOLOATION_MODE = "nearest"


def apply_imresize_intrinsics(resize_tfm, intrinsics):
    assert intrinsics.shape == (3, 3)
    assert intrinsics[0, 1] == 0  # undistorted
    assert np.allclose(intrinsics, np.triu(intrinsics))  # check if upper triangular

    factor_x = resize_tfm.new_w / resize_tfm.w
    factor_y = resize_tfm.new_h / resize_tfm.h
    new_intrinsics = intrinsics * np.float32([factor_x, factor_y, 1]).reshape(3, 1)  # pylint: disable=too-many-function-args
    return new_intrinsics


def apply_imresize_depth(resize_tfm, depth):
    assert depth.shape == (resize_tfm.h, resize_tfm.w)
    interp = CV2_INTERPOLATION_MODES[DEFAULT_DEPTH_INTERPOLOATION_MODE]
    resized_depth = cv2.resize(depth, (resize_tfm.new_w, resize_tfm.new_h), interpolation=interp)
    return resized_depth


def resize_depth_preserve(resize_tfm, depth):
    """
    Adapted from:
        https://github.com/TRI-ML/packnet-sfm_internal/blob/919ab604ae2319e4554d3b588877acfddf877f9c/packnet_sfm/datasets/augmentations.py#L93

    -------------------------------------------------------------------------------------------------------------------

    Resizes depth map preserving all valid depth pixels
    Multiple downsampled points can be assigned to the same pixel.
    Parameters
    ----------
    depth : np.array [h,w]
        Depth map
    shape : tuple (H,W)
        Output shape
    Returns
    -------
    depth : np.array [H,W,1]
        Resized depth map
    """
    assert depth.shape == (resize_tfm.h, resize_tfm.w)

    new_shape = (resize_tfm.new_h, resize_tfm.new_w)

    h, w = depth.shape
    x = depth.reshape(-1)
    # Create coordinate grid
    uv = np.mgrid[:h, :w].transpose(1, 2, 0).reshape(-1, 2)
    # Filters valid points
    idx = x > 0
    crd, val = uv[idx], x[idx]
    # Downsamples coordinates
    crd[:, 0] = (crd[:, 0] * (new_shape[0] / h)).astype(np.int32)
    crd[:, 1] = (crd[:, 1] * (new_shape[1] / w)).astype(np.int32)
    # Filters points inside image
    idx = (crd[:, 0] < new_shape[0]) & (crd[:, 1] < new_shape[1])
    crd, val = crd[idx], val[idx]
    # Creates downsampled depth image and assigns points
    resized_depth = np.zeros(new_shape)
    resized_depth[crd[:, 0], crd[:, 1]] = val
    return resized_depth


def apply_imresize_box3d(resize_tfm, box3d):
    return box3d


# (dennis.park) Augment ResizeTransform to handle intrinsics, depth
ResizeTransform.register_type("intrinsics", apply_imresize_intrinsics)
# ResizeTransform.register_type("depth", apply_imresize_depth)
ResizeTransform.register_type("depth", resize_depth_preserve)
ResizeTransform.register_type("box3d", apply_imresize_box3d)


class ResizeShortestEdge(_ResizeShortestEdge):
    def get_transform(self, image):
        tfm = super().get_transform(image)
        return ResizeTransform(tfm.h, tfm.w, tfm.new_h, tfm.new_w)
