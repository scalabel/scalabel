# Copyright 2021 Toyota Research Institute.  All rights reserved.
import numpy as np
from fvcore.transforms.transform import CropTransform

from detectron2.data.transforms import RandomCrop as _RandomCrop


def apply_imcrop_intrinsics(crop_tfm, intrinsics):
    assert intrinsics.shape == (3, 3)
    assert intrinsics[0, 1] == 0  # undistorted
    assert np.allclose(intrinsics, np.triu(intrinsics))  # check if upper triangular

    x0, y0 = crop_tfm.x0, crop_tfm.y0
    new_intrinsics = intrinsics.copy()
    new_intrinsics[0, 2] -= x0
    new_intrinsics[1, 2] -= y0

    return new_intrinsics


def apply_imcrop_depth(crop_tfm, depth):
    assert len(depth.shape) == 2
    x0, y0, w, h = crop_tfm.x0, crop_tfm.y0, crop_tfm.w, crop_tfm.h
    return depth[y0:y0 + h, x0:x0 + w]


def apply_imcrop_box3d(crop_tfm, box3d):  # pylint: disable=unused-argument
    return box3d


# (dennis.park) Augment ResizeTransform to handle intrinsics, depth
CropTransform.register_type("intrinsics", apply_imcrop_intrinsics)
CropTransform.register_type("depth", apply_imcrop_depth)
CropTransform.register_type("box3d", apply_imcrop_box3d)


class RandomCrop(_RandomCrop):
    def get_transform(self, image):
        tfm = super().get_transform(image)
        return CropTransform(tfm.x0, tfm.y0, tfm.w, tfm.h)
