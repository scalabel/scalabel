import torch
from torch import nn
import numpy as np
import skimage.transform as transform
import logging

from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.config import get_cfg, CfgNode
from detectron2.modeling import build_model

from .resnet_skip import SkipResnet50
from .first_v import FirstVertex
from .conv_lstm import AttConvLSTM
from .evaluator import Evaluator
from ..utils import utils


@META_ARCH_REGISTRY.register()
class PolyRNNPP(nn.Module):
    def __init__(self, cfg):
        super(PolyRNNPP, self).__init__()
        self.cfg = cfg.clone()
        self.device = self.cfg.MODEL.DEVICE

        self.encoder = SkipResnet50()
        self.first_v = FirstVertex(self.encoder.feat_size, self.encoder.final_dim)
        self.conv_lstm = AttConvLSTM(
            cfg,
            feats_channels=self.encoder.final_dim,
            feats_dim=self.encoder.feat_size,
            time_steps=self.cfg.MODEL.POLYRNNPP.MAX_POLY_LEN,
            use_bn=self.cfg.MODEL.POLYRNNPP.USE_BN_LSTM
        )
        self.evaluator = Evaluator(
            cfg,
            feats_dim=self.encoder.feat_size,
            feats_channels=self.encoder.final_dim,
            hidden_channels=self.conv_lstm.hidden_dim
        )

    def preprocess(self, batched_inputs):
        img = batched_inputs["image"].cpu().numpy().transpose(1, 2, 0)
        x0, y0, x1, y1 = batched_inputs["bbox"]
        poly = batched_inputs["poly"]

        w = x1 - x0
        h = y1 - y0
        x_center = (x0 + x1) / 2
        y_center = (y0 + y1) / 2

        widescreen = True if w > h else False
        if not widescreen:
            x_center, y_center, w, h = y_center, x_center, h, w

        x_min = int(np.floor(x_center - w * (1 + 0.15) / 2.))
        x_max = int(np.ceil(x_center + w * (1 + 0.15) / 2.))

        x_min = max(0, x_min)
        x_max = min(img.shape[1] - 1, x_max)

        patch_w = x_max - x_min

        y_min = int(np.floor(y_center - patch_w / 2.))
        y_max = y_min + patch_w

        top_margin_x = 0
        top_margin_y = max(0, y_min) - y_min

        y_min = max(0, y_min)
        y_max = min(img.shape[0] - 1, y_max)

        scale_factor = float(224) / patch_w

        if not widescreen:
            x_min, x_max, y_min, y_max = y_min, y_max, x_min, x_max
            top_margin_x, top_margin_y = top_margin_y, top_margin_x

        patch_img = img[y_min:y_max, x_min:x_max, :]

        new_img = np.zeros([patch_w, patch_w, 3], dtype=np.float32)
        new_img[top_margin_y: top_margin_y + patch_img.shape[0], top_margin_x: top_margin_x + patch_img.shape[1], :] = patch_img

        new_img = transform.rescale(new_img, scale_factor, order=1, preserve_range=True, multichannel=True)
        new_img = new_img.astype(np.float32)

        EPS = 1e-7
        starting_point = [x_min - top_margin_x, y_min - top_margin_y]
        if poly is not None:
            poly[:, 0] = (poly[:, 0] - starting_point[0]) / float(patch_w)
            poly[:, 1] = (poly[:, 1] - starting_point[1]) / float(patch_w)

            poly[:, 0] = np.clip(poly[:, 0], 0 + EPS, 1 - EPS)
            poly[:, 1] = np.clip(poly[:, 1], 0 + EPS, 1 - EPS)

            poly = utils.poly01_to_poly0g(poly, self.encoder.feat_size)
            arr_poly = np.ones((self.cfg.MODEL.POLYRNNPP.MAX_POLY_LEN, 2), np.float32) * -1
            len_to_keep = min(len(poly), self.cfg.MODEL.POLYRNNPP.MAX_POLY_LEN)
            arr_poly[:len_to_keep] = poly[:len_to_keep]
        else:
            arr_poly = None

        return_dict = {
            'img': torch.as_tensor(new_img.astype("float32").transpose(2, 0, 1)),
            'patch_w': patch_w,
            'starting_point': starting_point,
            'poly': torch.as_tensor(arr_poly.astype("float32")) if arr_poly is not None else None
        }

        return return_dict
    
    def postprocess(self, processed_dict, out_dicts):
        grid_size = self.encoder.feat_size

        poly = out_dicts["pred_polys"].cpu().numpy()[0]
        poly = utils.get_masked_poly(poly, grid_size)
        poly = utils.class_to_xy(poly, grid_size)
        poly = utils.poly0g_to_poly01(poly, grid_size)
        poly = poly * processed_dict['patch_w']
        poly = poly + processed_dict['starting_point']

        return poly.astype(np.int32).tolist()

    def forward(self, batched_inputs):
        processed_dict = self.preprocess(batched_inputs[0])

        x = processed_dict["img"].to(self.device).unsqueeze(0)
        poly = processed_dict["poly"]
        if poly is not None:
            poly = poly.to(self.device).unsqueeze(0)

        batch_size = x.shape[0]
        concat_feats, feats = self.encoder(x)

        _, _, first_logprob, first_v = self.first_v(feats, beam_size=5)

        poly_class = None
        if poly is not None:
            poly_class = utils.xy_to_class(poly, grid_size=self.encoder.feat_size)
            first_v = poly_class[:, 0]
            first_logprob = None

        out_dict = self.conv_lstm(feats, first_v, poly_class, fp_beam_size=5, first_log_prob=first_logprob)
        ious = self.evaluator(out_dict['feats'], out_dict['rnn_state'], out_dict['pred_polys'])
        comparison_metric = ious
        out_dict['ious'] = ious

        isect = utils.count_self_intersection(out_dict["pred_polys"].cpu().numpy(), self.encoder.feat_size)
        isect[isect != 0] -= float("inf")
        isect = torch.from_numpy(isect).to(torch.float32).to(self.device)
        comparison_metric = comparison_metric + isect

        comparison_metric = comparison_metric.view(batch_size, 5, 1)
        out_dict['pred_polys'] = out_dict['pred_polys'].view(batch_size, 5, 1, -1)

        # Max across beams
        comparison_metric, beam_idx = torch.max(comparison_metric, dim=-1)

        # Max across first points
        comparison_metric, fp_beam_idx = torch.max(comparison_metric, dim=-1)

        pred_polys = torch.zeros(batch_size,
                                 self.cfg.MODEL.POLYRNNPP.MAX_POLY_LEN,
                                 device=self.device,
                                 dtype=out_dict['pred_polys'].dtype)

        for b in torch.arange(batch_size, dtype=torch.int32):
            # Get best beam from all first points and all beams
            pred_polys[b, :] = out_dict['pred_polys'][b, fp_beam_idx[b], beam_idx[b, fp_beam_idx[b]], :]

        out_dict['pred_polys'] = pred_polys

        poly = self.postprocess(processed_dict, out_dict)

        return [poly]
