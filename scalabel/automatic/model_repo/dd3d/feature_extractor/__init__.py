# Copyright 2021 Toyota Research Institute.  All rights reserved.
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import BACKBONE_REGISTRY, Backbone

from .dla import (
    build_dla_backbone,
    build_dla_fpn_backbone,
    build_fcos_dla_fpn_backbone_p6,
    build_fcos_dla_fpn_backbone_p67,
)
from .vovnet import build_fcos_vovnet_fpn_backbone_p6, build_vovnet_backbone, build_vovnet_fpn_backbone


def build_feature_extractor(cfg, input_shape=None):
    """
    Build a backbone from `cfg.FE.BUILDER`

    Returns:
        an instance of :class:`Backbone`
    """
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

    builder_name = cfg.FE.BUILDER
    feature_extractor = BACKBONE_REGISTRY.get(builder_name)(cfg, input_shape)
    assert isinstance(feature_extractor, Backbone)
    return feature_extractor
