# Copyright 2021 Toyota Research Institute.  All rights reserved.
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import Conv2d, get_norm
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

from ..layers.normalization import ModuleListDial, Offset, Scale
from .dense_depth_loss import build_dense_depth_loss
from ..feature_extractor import build_feature_extractor
from ..structures.image_list import ImageList
from ..utils.tensor2d import aligned_bilinear


class DD3DDenseDepthHead(nn.Module):
    def __init__(self, cfg, input_shape):
        super().__init__()
        self.in_strides = [shape.stride for shape in input_shape]
        self.num_levels = len(input_shape)
        assert self.in_strides == [shape.stride for shape in input_shape]

        self.mean_depth_per_level = torch.FloatTensor(cfg.DD3D.FCOS3D.MEAN_DEPTH_PER_LEVEL)
        self.std_depth_per_level = torch.FloatTensor(cfg.DD3D.FCOS3D.STD_DEPTH_PER_LEVEL)

        self.scale_depth_by_focal_lengths_factor = cfg.DD3D.FCOS3D.SCALE_DEPTH_BY_FOCAL_LENGTHS_FACTOR

        self.use_scale = cfg.DD3D.FCOS3D.USE_SCALE
        self.depth_scale_init_factor = cfg.DD3D.FCOS3D.DEPTH_SCALE_INIT_FACTOR

        box3d_tower = []
        in_channels = input_shape[0].channels

        num_convs = cfg.DD3D.FCOS3D.NUM_CONVS
        use_deformable = cfg.DD3D.FCOS3D.USE_DEFORMABLE
        norm = cfg.DD3D.FCOS3D.NORM

        if use_deformable:
            raise ValueError("Not supported yet.")

        for i in range(num_convs):
            if norm in ("BN", "FrozenBN"):
                # Each FPN level has its own batchnorm layer.
                # "BN" is converted to "SyncBN" in distributed training (see train.py)
                norm_layer = ModuleListDial([get_norm(norm, in_channels) for _ in range(self.num_levels)])
            else:
                norm_layer = get_norm(norm, in_channels)
            box3d_tower.append(
                Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=norm_layer is None,
                    norm=norm_layer,
                    activation=F.relu,
                )
            )
        self.add_module("box3d_tower", nn.Sequential(*box3d_tower))

        # Each FPN level has its own predictor layer.
        self.dense_depth = nn.ModuleList(
            [
                Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1, bias=(not cfg.DD3D.FCOS3D.USE_SCALE))
                for _ in range(self.num_levels)
            ]
        )

        if self.use_scale:
            self.scales_depth = nn.ModuleList(
                [Scale(init_value=sigma * self.depth_scale_init_factor) for sigma in self.std_depth_per_level]
            )
            self.offsets_depth = nn.ModuleList([Offset(init_value=b) for b in self.mean_depth_per_level])

        self._init_weights()

    def _init_weights(self):

        for l in self.box3d_tower.modules():
            if isinstance(l, nn.Conv2d):
                torch.nn.init.kaiming_normal_(l.weight, mode="fan_out", nonlinearity="relu")
                if l.bias is not None:
                    torch.nn.init.constant_(l.bias, 0)

        for l in self.dense_depth.modules():
            if isinstance(l, nn.Conv2d):
                torch.nn.init.kaiming_uniform_(l.weight, a=1)
                if l.bias is not None:  # depth head may not have bias.
                    torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        assert len(x) == self.num_levels
        dense_depth = []
        for l, features in enumerate(x):
            box3d_tower_out = self.box3d_tower(features)
            dense_depth_lvl = self.dense_depth[l](box3d_tower_out)
            if self.use_scale:
                dense_depth_lvl = self.offsets_depth[l](self.scales_depth[l](dense_depth_lvl))
            dense_depth.append(dense_depth_lvl)
        return dense_depth


@META_ARCH_REGISTRY.register()
class DD3DDenseDepth(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.in_features = cfg.DD3D.IN_FEATURES
        self.feature_locations_offset = cfg.DD3D.FEATURE_LOCATIONS_OFFSET

        self.backbone = build_feature_extractor(cfg)
        backbone_output_shape = self.backbone.output_shape()
        backbone_output_shape = [backbone_output_shape[f] for f in self.in_features]

        self.fcos3d_head = DD3DDenseDepthHead(cfg, backbone_output_shape)
        self.depth_loss = build_dense_depth_loss(cfg)

        self.scale_depth_by_focal_lengths = cfg.DD3D.FCOS3D.SCALE_DEPTH_BY_FOCAL_LENGTHS
        self.scale_depth_by_focal_lengths_factor = cfg.DD3D.FCOS3D.SCALE_DEPTH_BY_FOCAL_LENGTHS_FACTOR

        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

    @property
    def device(self):
        return self.pixel_mean.device

    def preprocess_image(self, x):
        return (x - self.pixel_mean) / self.pixel_std

    def forward(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.preprocess_image(x) for x in images]

        if "intrinsics" in batched_inputs[0]:
            intrinsics = [x["intrinsics"].to(self.device) for x in batched_inputs]
        else:
            intrinsics = None
        images = ImageList.from_tensors(images, self.backbone.size_divisibility, intrinsics=intrinsics)

        gt_dense_depth = None
        if "depth" in batched_inputs[0]:
            gt_dense_depth = [x["depth"].to(self.device) for x in batched_inputs]
            gt_dense_depth = ImageList.from_tensors(
                gt_dense_depth, self.backbone.size_divisibility, intrinsics=intrinsics
            )

        features = self.backbone(images.tensor)
        features = [features[f] for f in self.in_features]
        dense_depth = self.fcos3d_head(features)

        inv_intrinsics = images.intrinsics.inverse() if images.intrinsics is not None else None

        # Upsample.
        dense_depth = [
            aligned_bilinear(x, factor=stride, offset=self.feature_locations_offset).squeeze(1)
            for x, stride in zip(dense_depth, self.in_strides)
        ]

        if self.scale_depth_by_focal_lengths:
            assert inv_intrinsics is not None
            pixel_size = torch.norm(torch.stack([inv_intrinsics[:, 0, 0], inv_intrinsics[:, 1, 1]], dim=-1), dim=-1)
            scaled_pixel_size = (pixel_size * self.scale_depth_by_focal_lengths_factor).reshape(-1, 1, 1)
            dense_depth = [x / scaled_pixel_size for x in dense_depth]

        if self.training:
            losses = {}
            for lvl, x in enumerate(dense_depth):
                loss_lvl = self.depth_loss(x, gt_dense_depth.tensor)["loss_dense_depth"]
                loss_lvl = loss_lvl / (np.sqrt(2) ** lvl)  # Is sqrt(2) good?
                losses.update({f"loss_dense_depth_lvl_{lvl}": loss_lvl})
            return losses
        else:
            raise NotImplementedError()
