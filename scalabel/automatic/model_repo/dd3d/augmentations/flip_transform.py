# Copyright 2021 Toyota Research Institute.  All rights reserved.
import numpy as np
from fvcore.transforms.transform import HFlipTransform, NoOpTransform, VFlipTransform

from detectron2.data.transforms import RandomFlip as _RandomFlip


def apply_hflip_intrinsics(hflip_tfm, intrinsics):
    intrinsics[0, 2] = hflip_tfm.width - intrinsics[0, 2]
    return intrinsics


def apply_vflip_intrinsics(vflip_tfm, intrinsics):
    intrinsics[1, 2] = vflip_tfm.height - intrinsics[1, 2]
    return intrinsics


def apply_hflip_depth(hflip_tfm, depth):  # pylint: disable=unused-argument
    assert depth.ndim == 2
    return np.flip(depth, axis=1).copy()


def apply_vflip_depth(vflip_tfm, depth):  # pylint: disable=unused-argument
    assert depth.ndim == 2
    return np.flip(depth, axis=0).copy()


def apply_hflip_box3d(hflip_tfm, box3d):  # pylint: disable=unused-argument
    """Horizontally flip 3D box.

    CAVEAT: This function makes assumption about the object symmetry wrt *y=0* plane.

    new quaternion: [quat.z, -quat.y, -quat.x, quat.w]
    https://stackoverflow.com/questions/32438252/efficient-way-to-apply-mirror-effect-on-quaternion-rotation

    Parameters
    ----------
    hflip_tfm: HFlipTransform

    box3d: np.array
        10D representation of 3D box. quaternion (4) + location (3) + dimension (3)

    Returns
    -------
    np.array
        10D representation of flipped 3D box.
    """
    quat, tvec, dims = box3d[:4], box3d[4:7], box3d[7:]

    quat_new = np.float32([quat[3], -quat[2], -quat[1], quat[0]])
    tvec_new = tvec.copy()
    tvec_new[0] = -tvec_new[0]
    dims_new = dims.copy()
    return np.concatenate([quat_new, tvec_new, dims_new])


def apply_vflip_box3d(vflip_tfm, box3d):  # pylint: disable=unused-argument
    # TODO
    raise NotImplementedError()


HFlipTransform.register_type("intrinsics", apply_hflip_intrinsics)
HFlipTransform.register_type("depth", apply_hflip_depth)
HFlipTransform.register_type("box3d", apply_hflip_box3d)

VFlipTransform.register_type("intrinsics", apply_vflip_intrinsics)
VFlipTransform.register_type("depth", apply_vflip_depth)
VFlipTransform.register_type("box3d", apply_vflip_box3d)


class RandomFlip(_RandomFlip):
    def get_transform(self, image):
        tfm = super().get_transform(image)
        if isinstance(tfm, NoOpTransform):
            return tfm
        elif isinstance(tfm, HFlipTransform):
            return HFlipTransform(tfm.width)
        else:
            assert isinstance(tfm, VFlipTransform)
            return VFlipTransform(tfm.height)
