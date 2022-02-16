# Copyright (c) Youngwan Lee (ETRI) All Rights Reserved.
# Copyright 2021 Toyota Research Institute.  All rights reserved.
from collections import OrderedDict

import fvcore.nn.weight_init as weight_init
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.layers import FrozenBatchNorm2d, ShapeSpec, get_norm
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool

__all__ = ["VoVNet", "build_vovnet_backbone", "build_vovnet_fpn_backbone"]

_NORM = False

VoVNet19_slim_dw_eSE = {
    "stem": [64, 64, 64],
    "stage_conv_ch": [64, 80, 96, 112],
    "stage_out_ch": [112, 256, 384, 512],
    "layer_per_block": 3,
    "block_per_stage": [1, 1, 1, 1],
    "eSE": True,
    "dw": True,
}

VoVNet19_dw_eSE = {
    "stem": [64, 64, 64],
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 3,
    "block_per_stage": [1, 1, 1, 1],
    "eSE": True,
    "dw": True,
}

VoVNet19_slim_eSE = {
    "stem": [64, 64, 128],
    "stage_conv_ch": [64, 80, 96, 112],
    "stage_out_ch": [112, 256, 384, 512],
    "layer_per_block": 3,
    "block_per_stage": [1, 1, 1, 1],
    "eSE": True,
    "dw": False,
}

VoVNet19_eSE = {
    "stem": [64, 64, 128],
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 3,
    "block_per_stage": [1, 1, 1, 1],
    "eSE": True,
    "dw": False,
}

VoVNet39_eSE = {
    "stem": [64, 64, 128],
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 5,
    "block_per_stage": [1, 1, 2, 2],
    "eSE": True,
    "dw": False,
}

VoVNet57_eSE = {
    "stem": [64, 64, 128],
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 5,
    "block_per_stage": [1, 1, 4, 3],
    "eSE": True,
    "dw": False,
}

VoVNet99_eSE = {
    "stem": [64, 64, 128],
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 5,
    "block_per_stage": [1, 3, 9, 3],
    "eSE": True,
    "dw": False,
}

_STAGE_SPECS = {
    "V-19-slim-dw-eSE": VoVNet19_slim_dw_eSE,
    "V-19-dw-eSE": VoVNet19_dw_eSE,
    "V-19-slim-eSE": VoVNet19_slim_eSE,
    "V-19-eSE": VoVNet19_eSE,
    "V-39-eSE": VoVNet39_eSE,
    "V-57-eSE": VoVNet57_eSE,
    "V-99-eSE": VoVNet99_eSE,
}


def dw_conv3x3(in_channels, out_channels, module_name, postfix, stride=1, kernel_size=3, padding=1):
    """3x3 convolution with padding"""
    return [
        (
            "{}_{}/dw_conv3x3".format(module_name, postfix),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=out_channels,
                bias=False,
            ),
        ),
        (
            "{}_{}/pw_conv1x1".format(module_name, postfix),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
        ),
        ("{}_{}/pw_norm".format(module_name, postfix), get_norm(_NORM, out_channels)),
        ("{}_{}/pw_relu".format(module_name, postfix), nn.ReLU(inplace=True)),
    ]


def conv3x3(in_channels, out_channels, module_name, postfix, stride=1, groups=1, kernel_size=3, padding=1):
    """3x3 convolution with padding"""
    return [
        (
            f"{module_name}_{postfix}/conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
        ),
        (f"{module_name}_{postfix}/norm", get_norm(_NORM, out_channels)),
        (f"{module_name}_{postfix}/relu", nn.ReLU(inplace=True)),
    ]


def conv1x1(in_channels, out_channels, module_name, postfix, stride=1, groups=1, kernel_size=1, padding=0):
    """1x1 convolution with padding"""
    return [
        (
            f"{module_name}_{postfix}/conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
        ),
        (f"{module_name}_{postfix}/norm", get_norm(_NORM, out_channels)),
        (f"{module_name}_{postfix}/relu", nn.ReLU(inplace=True)),
    ]


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class eSEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(eSEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channel, channel, kernel_size=1, padding=0)
        self.hsigmoid = Hsigmoid()

    def forward(self, x):
        input = x
        x = self.avg_pool(x)
        x = self.fc(x)
        x = self.hsigmoid(x)
        return input * x


class _OSA_module(nn.Module):
    def __init__(
        self, in_ch, stage_ch, concat_ch, layer_per_block, module_name, SE=False, identity=False, depthwise=False
    ):

        super(_OSA_module, self).__init__()

        self.identity = identity
        self.depthwise = depthwise
        self.isReduced = False
        self.layers = nn.ModuleList()
        in_channel = in_ch
        if self.depthwise and in_channel != stage_ch:
            self.isReduced = True
            self.conv_reduction = nn.Sequential(
                OrderedDict(conv1x1(in_channel, stage_ch, "{}_reduction".format(module_name), "0"))
            )
        for i in range(layer_per_block):
            if self.depthwise:
                self.layers.append(nn.Sequential(OrderedDict(dw_conv3x3(stage_ch, stage_ch, module_name, i))))
            else:
                self.layers.append(nn.Sequential(OrderedDict(conv3x3(in_channel, stage_ch, module_name, i))))
            in_channel = stage_ch

        # feature aggregation
        in_channel = in_ch + layer_per_block * stage_ch
        self.concat = nn.Sequential(OrderedDict(conv1x1(in_channel, concat_ch, module_name, "concat")))

        self.ese = eSEModule(concat_ch)

    def forward(self, x):

        identity_feat = x

        output = []
        output.append(x)
        if self.depthwise and self.isReduced:
            x = self.conv_reduction(x)
        for layer in self.layers:
            x = layer(x)
            output.append(x)

        x = torch.cat(output, dim=1)
        xt = self.concat(x)

        xt = self.ese(xt)

        if self.identity:
            xt = xt + identity_feat

        return xt


class _OSA_stage(nn.Sequential):
    def __init__(
        self, in_ch, stage_ch, concat_ch, block_per_stage, layer_per_block, stage_num, SE=False, depthwise=False
    ):

        super(_OSA_stage, self).__init__()

        if not stage_num == 2:
            self.add_module("Pooling", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))

        if block_per_stage != 1:
            SE = False
        module_name = f"OSA{stage_num}_1"
        self.add_module(
            module_name, _OSA_module(in_ch, stage_ch, concat_ch, layer_per_block, module_name, SE, depthwise=depthwise)
        )
        for i in range(block_per_stage - 1):
            if i != block_per_stage - 2:  # last block
                SE = False
            module_name = f"OSA{stage_num}_{i + 2}"
            self.add_module(
                module_name,
                _OSA_module(
                    concat_ch,
                    stage_ch,
                    concat_ch,
                    layer_per_block,
                    module_name,
                    SE,
                    identity=True,
                    depthwise=depthwise,
                ),
            )


class VoVNet(Backbone):
    def __init__(self, cfg, input_ch, out_features=None):
        """
        Args:
            input_ch(int) : the number of input channel
            out_features (list[str]): name of the layers whose outputs should
                be returned in forward. Can be anything in "stem", "stage2" ...
        """
        super(VoVNet, self).__init__()

        global _NORM
        _NORM = cfg.NORM

        stage_specs = _STAGE_SPECS[cfg.NAME]

        stem_ch = stage_specs["stem"]
        config_stage_ch = stage_specs["stage_conv_ch"]
        config_concat_ch = stage_specs["stage_out_ch"]
        block_per_stage = stage_specs["block_per_stage"]
        layer_per_block = stage_specs["layer_per_block"]
        SE = stage_specs["eSE"]
        depthwise = stage_specs["dw"]

        self._out_features = out_features

        # Stem module
        conv_type = dw_conv3x3 if depthwise else conv3x3
        stem = conv3x3(input_ch, stem_ch[0], "stem", "1", 2)
        stem += conv_type(stem_ch[0], stem_ch[1], "stem", "2", 1)
        stem += conv_type(stem_ch[1], stem_ch[2], "stem", "3", 2)
        self.add_module("stem", nn.Sequential((OrderedDict(stem))))
        current_stirde = 4
        self._out_feature_strides = {"stem": current_stirde, "stage2": current_stirde}
        self._out_feature_channels = {"stem": stem_ch[2]}

        stem_out_ch = [stem_ch[2]]
        in_ch_list = stem_out_ch + config_concat_ch[:-1]
        # OSA stages
        self.stage_names = []
        for i in range(4):  # num_stages
            name = "stage%d" % (i + 2)  # stage 2 ... stage 5
            self.stage_names.append(name)
            self.add_module(
                name,
                _OSA_stage(
                    in_ch_list[i],
                    config_stage_ch[i],
                    config_concat_ch[i],
                    block_per_stage[i],
                    layer_per_block,
                    i + 2,
                    SE,
                    depthwise,
                ),
            )

            self._out_feature_channels[name] = config_concat_ch[i]
            if not i == 0:
                self._out_feature_strides[name] = current_stirde = int(current_stirde * 2)

        # initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def _freeze_backbone(self, freeze_at):
        if freeze_at < 0:
            return

        for stage_index in range(freeze_at):
            if stage_index == 0:
                m = self.stem  # stage 0 is the stem
            else:
                m = getattr(self, "stage" + str(stage_index + 1))
            for p in m.parameters():
                p.requires_grad = False
                FrozenBatchNorm2d.convert_frozen_batchnorm(self)

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for name in self.stage_names:
            x = getattr(self, name)(x)
            if name in self._out_features:
                outputs[name] = x

        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(channels=self._out_feature_channels[name], stride=self._out_feature_strides[name])
            for name in self._out_features
        }


@BACKBONE_REGISTRY.register()
def build_vovnet_backbone(cfg, input_shape):
    """
    Create a VoVNet instance from config.

    Returns:
        VoVNet: a :class:`VoVNet` instance.
    """
    out_features = cfg.OUT_FEATURES
    return VoVNet(cfg, input_shape.channels, out_features=out_features)


@BACKBONE_REGISTRY.register()
def build_vovnet_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_vovnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.FE.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.FE.FPN.FUSE_TYPE,
    )
    return backbone


class LastLevelP6(nn.Module):
    """
    This module is used in FCOS to generate extra layers
    """

    def __init__(self, in_channels, out_channels, in_features="res5"):
        super().__init__()
        self.num_levels = 1
        self.in_feature = in_features
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        for module in [self.p6]:
            weight_init.c2_xavier_fill(module)

    def forward(self, x):
        p6 = self.p6(x)
        return [p6]


@BACKBONE_REGISTRY.register()
def build_fcos_vovnet_fpn_backbone_p6(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_vovnet_backbone(cfg.FE.BACKBONE, input_shape)
    in_features = cfg.FE.FPN.IN_FEATURES
    out_channels = cfg.FE.FPN.OUT_CHANNELS
    in_channels_top = out_channels
    top_block = LastLevelP6(in_channels_top, out_channels, "p5")
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.FE.FPN.NORM,
        # top_block=LastLevelMaxPool(),
        top_block=top_block,
        fuse_type=cfg.FE.FPN.FUSE_TYPE,
    )

    backbone._size_divisibility *= 2

    return backbone
