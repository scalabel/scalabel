# Copyright 2021 Toyota Research Institute.  All rights reserved.
import torch
from torch import nn

from detectron2.config.config import configurable

from ..layers import smooth_l1_loss


class DenseDepthL1Loss(nn.Module):
    @configurable
    def __init__(self, beta, min_depth=0.0, max_depth=100.0, loss_weight=1.0):
        super().__init__()
        self.beta = beta
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.loss_weight = loss_weight

    @classmethod
    def from_config(cls, cfg):
        return {
            "beta": cfg.DD3D.FCOS3D.LOSS.SMOOTH_L1_BETA,
            "min_depth": cfg.DD3D.FCOS3D.MIN_DEPTH,
            "max_depth": cfg.DD3D.FCOS3D.MAX_DEPTH,
            "loss_weight": cfg.DD3D.FCOS3D.DEPTH_HEAD.LOSS_WEIGHT,
        }

    def forward(self, depth_pred, depth_gt, masks=None):
        M = (depth_gt < self.min_depth).to(torch.float32) + (depth_gt > self.max_depth).to(torch.float32)
        if masks is not None:
            M += (1.0 - masks).to(torch.float32)

        M = M == 0.0
        loss = smooth_l1_loss(depth_pred[M], depth_gt[M], beta=self.beta, reduction="mean")

        return {"loss_dense_depth": self.loss_weight * loss}


def build_dense_depth_loss(cfg):
    if cfg.DD3D.FCOS3D.DEPTH_HEAD.LOSS_TYPE == "L1":
        return DenseDepthL1Loss(cfg)
    else:
        ValueError(f"Not supported depth loss: {cfg.DD3D.FCOS3D.DEPTH_HEAD.LOSS_TYPE}")
