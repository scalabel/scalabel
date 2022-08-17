# Copyright 2021 Toyota Research Institute.  All rights reserved.
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import Conv2d, cat, get_norm

from ..layers.normalization import ModuleListDial, Offset, Scale
from .disentangled_box3d_loss import DisentangledBox3DLoss
from ..structures.boxes3d import Boxes3D
from ..utils.geometry import allocentric_to_egocentric, unproject_points2d

EPS = 1e-7


def predictions_to_boxes3d(
    quat,
    proj_ctr,
    depth,
    size,
    locations,
    inv_intrinsics,
    canon_box_sizes,
    min_depth,
    max_depth,
    scale_depth_by_focal_lengths_factor,
    scale_depth_by_focal_lengths=True,
    quat_is_allocentric=True,
    depth_is_distance=False,
):
    # Normalize to make quat unit norm.
    quat = quat / quat.norm(dim=1, keepdim=True).clamp(min=EPS)
    # Make sure again it's numerically unit-norm.
    quat = quat / quat.norm(dim=1, keepdim=True)

    if scale_depth_by_focal_lengths:
        pixel_size = torch.norm(torch.stack([inv_intrinsics[:, 0, 0], inv_intrinsics[:, 1, 1]], dim=-1), dim=-1)
        depth = depth / (pixel_size * scale_depth_by_focal_lengths_factor)

    if depth_is_distance:
        depth = depth / unproject_points2d(locations, inv_intrinsics).norm(dim=1).clamp(min=EPS)

    depth = depth.reshape(-1, 1).clamp(min_depth, max_depth)

    proj_ctr = proj_ctr + locations

    if quat_is_allocentric:
        quat = allocentric_to_egocentric(quat, proj_ctr, inv_intrinsics)

    size = (size.tanh() + 1.0) * canon_box_sizes  # max size = 2 * canon_size

    return Boxes3D(quat, proj_ctr, depth, size, inv_intrinsics)


class FCOS3DHead(nn.Module):
    def __init__(self, cfg, input_shape):
        super().__init__()
        self.num_classes = cfg.DD3D.NUM_CLASSES
        self.in_strides = [shape.stride for shape in input_shape]
        self.num_levels = len(input_shape)

        self.use_scale = cfg.DD3D.FCOS3D.USE_SCALE
        self.depth_scale_init_factor = cfg.DD3D.FCOS3D.DEPTH_SCALE_INIT_FACTOR
        self.proj_ctr_scale_init_factor = cfg.DD3D.FCOS3D.PROJ_CTR_SCALE_INIT_FACTOR
        self.use_per_level_predictors = cfg.DD3D.FCOS3D.PER_LEVEL_PREDICTORS

        self.register_buffer("mean_depth_per_level", torch.Tensor(cfg.DD3D.FCOS3D.MEAN_DEPTH_PER_LEVEL))
        self.register_buffer("std_depth_per_level", torch.Tensor(cfg.DD3D.FCOS3D.STD_DEPTH_PER_LEVEL))

        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        num_convs = cfg.DD3D.FCOS3D.NUM_CONVS
        use_deformable = cfg.DD3D.FCOS3D.USE_DEFORMABLE
        norm = cfg.DD3D.FCOS3D.NORM

        if use_deformable:
            raise ValueError("Not supported yet.")

        box3d_tower = []
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

        num_classes = self.num_classes if not cfg.DD3D.FCOS3D.CLASS_AGNOSTIC_BOX3D else 1
        num_levels = self.num_levels if cfg.DD3D.FCOS3D.PER_LEVEL_PREDICTORS else 1

        # 3D box branches.
        self.box3d_quat = nn.ModuleList(
            [
                Conv2d(in_channels, 4 * num_classes, kernel_size=3, stride=1, padding=1, bias=True)
                for _ in range(num_levels)
            ]
        )
        self.box3d_ctr = nn.ModuleList(
            [
                Conv2d(in_channels, 2 * num_classes, kernel_size=3, stride=1, padding=1, bias=True)
                for _ in range(num_levels)
            ]
        )
        self.box3d_depth = nn.ModuleList(
            [
                Conv2d(in_channels, 1 * num_classes, kernel_size=3, stride=1, padding=1, bias=(not self.use_scale))
                for _ in range(num_levels)
            ]
        )
        self.box3d_size = nn.ModuleList(
            [
                Conv2d(in_channels, 3 * num_classes, kernel_size=3, stride=1, padding=1, bias=True)
                for _ in range(num_levels)
            ]
        )
        self.box3d_conf = nn.ModuleList(
            [
                Conv2d(in_channels, 1 * num_classes, kernel_size=3, stride=1, padding=1, bias=True)
                for _ in range(num_levels)
            ]
        )

        if self.use_scale:
            self.scales_proj_ctr = nn.ModuleList(
                [Scale(init_value=stride * self.proj_ctr_scale_init_factor) for stride in self.in_strides]
            )
            # (pre-)compute (mean, std) of depth for each level, and determine the init value here.
            self.scales_size = nn.ModuleList([Scale(init_value=1.0) for _ in range(self.num_levels)])
            self.scales_conf = nn.ModuleList([Scale(init_value=1.0) for _ in range(self.num_levels)])

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

        predictors = [self.box3d_quat, self.box3d_ctr, self.box3d_depth, self.box3d_size, self.box3d_conf]

        for modules in predictors:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.kaiming_uniform_(l.weight, a=1)
                    if l.bias is not None:  # depth head may not have bias.
                        torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        box3d_quat, box3d_ctr, box3d_depth, box3d_size, box3d_conf = [], [], [], [], []
        dense_depth = None
        for l, features in enumerate(x):
            box3d_tower_out = self.box3d_tower(features)

            _l = l if self.use_per_level_predictors else 0

            # 3D box
            quat = self.box3d_quat[_l](box3d_tower_out)
            proj_ctr = self.box3d_ctr[_l](box3d_tower_out)
            depth = self.box3d_depth[_l](box3d_tower_out)
            size3d = self.box3d_size[_l](box3d_tower_out)
            conf3d = self.box3d_conf[_l](box3d_tower_out)

            if self.use_scale:
                # TODO: to optimize the runtime, apply this scaling in inference (and loss compute) only on FG pixels?
                proj_ctr = self.scales_proj_ctr[l](proj_ctr)
                size3d = self.scales_size[l](size3d)
                conf3d = self.scales_conf[l](conf3d)
                depth = self.offsets_depth[l](self.scales_depth[l](depth))

            box3d_quat.append(quat)
            box3d_ctr.append(proj_ctr)
            box3d_depth.append(depth)
            box3d_size.append(size3d)
            box3d_conf.append(conf3d)

        return box3d_quat, box3d_ctr, box3d_depth, box3d_size, box3d_conf, dense_depth


class FCOS3DLoss:
    def __init__(self, cfg):
        self.canon_box_sizes = cfg.DD3D.FCOS3D.CANONICAL_BOX3D_SIZES
        self.min_depth = cfg.DD3D.FCOS3D.MIN_DEPTH
        self.max_depth = cfg.DD3D.FCOS3D.MAX_DEPTH
        self.predict_allocentric_rot = cfg.DD3D.FCOS3D.PREDICT_ALLOCENTRIC_ROT
        self.scale_depth_by_focal_lengths = cfg.DD3D.FCOS3D.SCALE_DEPTH_BY_FOCAL_LENGTHS
        self.scale_depth_by_focal_lengths_factor = cfg.DD3D.FCOS3D.SCALE_DEPTH_BY_FOCAL_LENGTHS_FACTOR
        self.predict_distance = cfg.DD3D.FCOS3D.PREDICT_DISTANCE

        self.box3d_reg_loss_fn = DisentangledBox3DLoss(cfg)  # pylint: disable=no-value-for-parameter
        self.box3d_loss_weight = cfg.DD3D.FCOS3D.LOSS.WEIGHT_BOX3D
        self.conf3d_loss_weight = cfg.DD3D.FCOS3D.LOSS.WEIGHT_CONF3D
        self.conf_3d_temperature = cfg.DD3D.FCOS3D.LOSS.CONF_3D_TEMPERATURE

        self.num_classes = cfg.DD3D.NUM_CLASSES
        self.class_agnostic = cfg.DD3D.FCOS3D.CLASS_AGNOSTIC_BOX3D

    def __call__(
        self,
        box3d_quat,
        box3d_ctr,
        box3d_depth,
        box3d_size,
        box3d_conf,
        dense_depth,
        inv_intrinsics,
        fcos2d_info,
        targets,
    ):
        labels = targets["labels"]
        box3d_targets = targets["box3d_targets"]
        pos_inds = targets["pos_inds"]

        if pos_inds.numel() == 0:
            losses = {
                "loss_box3d_quat": box3d_quat.sum() * 0.0,
                "loss_box3d_proj_ctr": box3d_ctr.sum() * 0.0,
                "loss_box3d_depth": box3d_depth.sum() * 0.0,
                "loss_box3d_size": box3d_size.sum() * 0.0,
                "loss_conf3d": box3d_conf.sum() * 0.0,
            }
            return losses

        if len(labels) != len(box3d_targets):
            raise ValueError(
                f"The size of 'labels' and 'box3d_targets' does not match: a={len(labels)}, b={len(box3d_targets)}"
            )

        num_classes = self.num_classes if not self.class_agnostic else 1

        box3d_quat_pred = cat([x.permute(0, 2, 3, 1).reshape(-1, 4, num_classes) for x in box3d_quat])
        box3d_ctr_pred = cat([x.permute(0, 2, 3, 1).reshape(-1, 2, num_classes) for x in box3d_ctr])
        box3d_depth_pred = cat([x.permute(0, 2, 3, 1).reshape(-1, num_classes) for x in box3d_depth])
        box3d_size_pred = cat([x.permute(0, 2, 3, 1).reshape(-1, 3, num_classes) for x in box3d_size])
        box3d_conf_pred = cat([x.permute(0, 2, 3, 1).reshape(-1, num_classes) for x in box3d_conf])

        # ----------------------
        # 3D box disentangled loss
        # ----------------------
        box3d_targets = box3d_targets[pos_inds]

        box3d_quat_pred = box3d_quat_pred[pos_inds]
        box3d_ctr_pred = box3d_ctr_pred[pos_inds]
        box3d_depth_pred = box3d_depth_pred[pos_inds]
        box3d_size_pred = box3d_size_pred[pos_inds]
        box3d_conf_pred = box3d_conf_pred[pos_inds]

        if self.class_agnostic:
            box3d_quat_pred = box3d_quat_pred.squeeze(-1)
            box3d_ctr_pred = box3d_ctr_pred.squeeze(-1)
            box3d_depth_pred = box3d_depth_pred.squeeze(-1)
            box3d_size_pred = box3d_size_pred.squeeze(-1)
            box3d_conf_pred = box3d_conf_pred.squeeze(-1)
        else:
            I = labels[pos_inds][..., None, None]
            box3d_quat_pred = torch.gather(box3d_quat_pred, dim=2, index=I.repeat(1, 4, 1)).squeeze(-1)
            box3d_ctr_pred = torch.gather(box3d_ctr_pred, dim=2, index=I.repeat(1, 2, 1)).squeeze(-1)
            box3d_depth_pred = torch.gather(box3d_depth_pred, dim=1, index=I.squeeze(-1)).squeeze(-1)
            box3d_size_pred = torch.gather(box3d_size_pred, dim=2, index=I.repeat(1, 3, 1)).squeeze(-1)
            box3d_conf_pred = torch.gather(box3d_conf_pred, dim=1, index=I.squeeze(-1)).squeeze(-1)

        canon_box_sizes = box3d_quat_pred.new_tensor(self.canon_box_sizes)[labels[pos_inds]]

        locations = targets["locations"][pos_inds]
        im_inds = targets["im_inds"][pos_inds]
        inv_intrinsics = inv_intrinsics[im_inds]

        box3d_pred = predictions_to_boxes3d(
            box3d_quat_pred,
            box3d_ctr_pred,
            box3d_depth_pred,
            box3d_size_pred,
            locations,
            inv_intrinsics,
            canon_box_sizes,
            self.min_depth,
            self.max_depth,
            scale_depth_by_focal_lengths_factor=self.scale_depth_by_focal_lengths_factor,
            scale_depth_by_focal_lengths=self.scale_depth_by_focal_lengths,
            quat_is_allocentric=self.predict_allocentric_rot,
            depth_is_distance=self.predict_distance,
        )

        centerness_targets = fcos2d_info["centerness_targets"]
        loss_denom = fcos2d_info["loss_denom"]
        losses_box3d, box3d_l1_error = self.box3d_reg_loss_fn(box3d_pred, box3d_targets, locations, centerness_targets)

        losses_box3d = {k: self.box3d_loss_weight * v / loss_denom for k, v in losses_box3d.items()}

        conf_3d_targets = torch.exp(-1.0 / self.conf_3d_temperature * box3d_l1_error)
        loss_conf3d = F.binary_cross_entropy_with_logits(box3d_conf_pred, conf_3d_targets, reduction="none")
        loss_conf3d = self.conf3d_loss_weight * (loss_conf3d * centerness_targets).sum() / loss_denom

        losses = {"loss_conf3d": loss_conf3d, **losses_box3d}

        return losses


class FCOS3DInference:
    def __init__(self, cfg):
        self.canon_box_sizes = cfg.DD3D.FCOS3D.CANONICAL_BOX3D_SIZES
        self.min_depth = cfg.DD3D.FCOS3D.MIN_DEPTH
        self.max_depth = cfg.DD3D.FCOS3D.MAX_DEPTH
        self.predict_allocentric_rot = cfg.DD3D.FCOS3D.PREDICT_ALLOCENTRIC_ROT
        self.scale_depth_by_focal_lengths = cfg.DD3D.FCOS3D.SCALE_DEPTH_BY_FOCAL_LENGTHS
        self.scale_depth_by_focal_lengths_factor = cfg.DD3D.FCOS3D.SCALE_DEPTH_BY_FOCAL_LENGTHS_FACTOR
        self.predict_distance = cfg.DD3D.FCOS3D.PREDICT_DISTANCE

        self.num_classes = cfg.DD3D.NUM_CLASSES
        self.class_agnostic = cfg.DD3D.FCOS3D.CLASS_AGNOSTIC_BOX3D

    def __call__(
        self, box3d_quat, box3d_ctr, box3d_depth, box3d_size, box3d_conf, inv_intrinsics, pred_instances, fcos2d_info
    ):
        # pred_instances: # List[List[Instances]], shape = (L, B)
        for lvl, (box3d_quat_lvl, box3d_ctr_lvl, box3d_depth_lvl, box3d_size_lvl, box3d_conf_lvl) in enumerate(
            zip(box3d_quat, box3d_ctr, box3d_depth, box3d_size, box3d_conf)
        ):

            # In-place modification: update per-level pred_instances.
            self.forward_for_single_feature_map(
                box3d_quat_lvl,
                box3d_ctr_lvl,
                box3d_depth_lvl,
                box3d_size_lvl,
                box3d_conf_lvl,
                inv_intrinsics,
                pred_instances[lvl],
                fcos2d_info[lvl],
            )  # List of Instances; one for each image.

    def forward_for_single_feature_map(
        self, box3d_quat, box3d_ctr, box3d_depth, box3d_size, box3d_conf, inv_intrinsics, pred_instances, fcos2d_info
    ):
        N = box3d_quat.shape[0]

        num_classes = self.num_classes if not self.class_agnostic else 1

        box3d_quat = box3d_quat.permute(0, 2, 3, 1).reshape(N, -1, 4, num_classes)
        box3d_ctr = box3d_ctr.permute(0, 2, 3, 1).reshape(N, -1, 2, num_classes)
        box3d_depth = box3d_depth.permute(0, 2, 3, 1).reshape(N, -1, num_classes)
        box3d_size = box3d_size.permute(0, 2, 3, 1).reshape(N, -1, 3, num_classes)
        box3d_conf = box3d_conf.permute(0, 2, 3, 1).reshape(N, -1, num_classes).sigmoid()

        for i in range(N):
            fg_inds_per_im = fcos2d_info["fg_inds_per_im"][i]
            class_inds_per_im = fcos2d_info["class_inds_per_im"][i]
            topk_indices = fcos2d_info["topk_indices"][i]

            box3d_quat_per_im = box3d_quat[i][fg_inds_per_im]
            box3d_ctr_per_im = box3d_ctr[i][fg_inds_per_im]
            box3d_depth_per_im = box3d_depth[i][fg_inds_per_im]
            box3d_size_per_im = box3d_size[i][fg_inds_per_im]
            box3d_conf_per_im = box3d_conf[i][fg_inds_per_im]

            if self.class_agnostic:
                box3d_quat_per_im = box3d_quat_per_im.squeeze(-1)
                box3d_ctr_per_im = box3d_ctr_per_im.squeeze(-1)
                box3d_depth_per_im = box3d_depth_per_im.squeeze(-1)
                box3d_size_per_im = box3d_size_per_im.squeeze(-1)
                box3d_conf_per_im = box3d_conf_per_im.squeeze(-1)
            else:
                I = class_inds_per_im[..., None, None]
                box3d_quat_per_im = torch.gather(box3d_quat_per_im, dim=2, index=I.repeat(1, 4, 1)).squeeze(-1)
                box3d_ctr_per_im = torch.gather(box3d_ctr_per_im, dim=2, index=I.repeat(1, 2, 1)).squeeze(-1)
                box3d_depth_per_im = torch.gather(box3d_depth_per_im, dim=1, index=I.squeeze(-1)).squeeze(-1)
                box3d_size_per_im = torch.gather(box3d_size_per_im, dim=2, index=I.repeat(1, 3, 1)).squeeze(-1)
                box3d_conf_per_im = torch.gather(box3d_conf_per_im, dim=1, index=I.squeeze(-1)).squeeze(-1)

            if topk_indices is not None:
                box3d_quat_per_im = box3d_quat_per_im[topk_indices]
                box3d_ctr_per_im = box3d_ctr_per_im[topk_indices]
                box3d_depth_per_im = box3d_depth_per_im[topk_indices]
                box3d_size_per_im = box3d_size_per_im[topk_indices]
                box3d_conf_per_im = box3d_conf_per_im[topk_indices]

            # scores_per_im = pred_instances[i].scores.square()
            # NOTE: Before refactoring, the squared score was used. Is raw 2D score better?
            scores_per_im = pred_instances[i].scores
            scores_3d_per_im = scores_per_im * box3d_conf_per_im

            canon_box_sizes = box3d_quat.new_tensor(self.canon_box_sizes)[pred_instances[i].pred_classes]
            inv_K = inv_intrinsics[i][None, ...].expand(len(box3d_quat_per_im), 3, 3)
            locations = pred_instances[i].locations
            pred_boxes3d = predictions_to_boxes3d(
                box3d_quat_per_im,
                box3d_ctr_per_im,
                box3d_depth_per_im,
                box3d_size_per_im,
                locations,
                inv_K,
                canon_box_sizes,
                self.min_depth,
                self.max_depth,
                scale_depth_by_focal_lengths_factor=self.scale_depth_by_focal_lengths_factor,
                scale_depth_by_focal_lengths=self.scale_depth_by_focal_lengths,
                quat_is_allocentric=self.predict_allocentric_rot,
                depth_is_distance=self.predict_distance,
            )

            # In-place modification: add fields to instances.
            pred_instances[i].pred_boxes3d = pred_boxes3d
            pred_instances[i].scores_3d = scores_3d_per_im
