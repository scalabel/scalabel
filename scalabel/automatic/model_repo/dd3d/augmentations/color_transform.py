# Copyright 2021 Toyota Research Institute.  All rights reserved.
# pylint: disable=unused-argument
from fvcore.transforms.transform import BlendTransform

from detectron2.data.transforms import RandomBrightness as _RandomBrightness
from detectron2.data.transforms import RandomContrast as _RandomContrast
from detectron2.data.transforms import RandomSaturation as _RandomSaturation


def apply_no_op_intrinsics(blend_tfm, intrinsics):
    return intrinsics


def apply_no_op_depth(blend_tfm, depth):
    return depth


def apply_no_op_box3d(blend_tfm, box3d):
    return box3d


# (dennis.park) Augment ResizeTransform to handle intrinsics, depth
BlendTransform.register_type("intrinsics", apply_no_op_intrinsics)
BlendTransform.register_type("depth", apply_no_op_depth)
BlendTransform.register_type("box3d", apply_no_op_box3d)


class RandomContrast(_RandomContrast):
    def get_transform(self, image):
        tfm = super().get_transform(image)
        return BlendTransform(tfm.src_image, tfm.src_weight, tfm.dst_weight)


class RandomBrightness(_RandomBrightness):
    def get_transform(self, image):
        tfm = super().get_transform(image)
        return BlendTransform(tfm.src_image, tfm.src_weight, tfm.dst_weight)


class RandomSaturation(_RandomSaturation):
    def get_transform(self, image):
        tfm = super().get_transform(image)
        return BlendTransform(tfm.src_image, tfm.src_weight, tfm.dst_weight)
