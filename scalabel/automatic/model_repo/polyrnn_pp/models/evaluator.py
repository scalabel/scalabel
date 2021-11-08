import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..utils import utils


class Evaluator(nn.Module):
    def __init__(self, cfg, feats_dim, feats_channels, hidden_channels):
        super(Evaluator, self).__init__()
        self.device = cfg.MODEL.DEVICE

        self.grid_size = feats_dim

        input_channels = feats_channels + sum(hidden_channels) + 2
        # +2 for the full mask and vertex mask

        conv1 = nn.Conv2d(
            in_channels = input_channels,
            out_channels = 16,
            kernel_size = 3,
            padding = 1
        )

        bn1 = nn.BatchNorm2d(16)
        relu1 = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(conv1, bn1, relu1)

        conv2 = nn.Conv2d(
            in_channels = 16,
            out_channels = 1,
            kernel_size = 3,
            padding = 1
        )

        bn2 = nn.BatchNorm2d(1)
        relu2 = nn.ReLU(inplace=True)

        self.conv2 = nn.Sequential(conv2, bn2, relu2)

        self.fc = nn.Linear(
            in_features = feats_dim**2,
            out_features = 1
        )

    def forward(self, feats, last_rnn_state, pred_poly):
        pred_poly = pred_poly.detach().cpu().numpy() #[bs, time]
        # we will use numpy functions to get pred_mask and pred_vertex_mask

        pred_mask = np.zeros((pred_poly.shape[0], 1, self.grid_size, self.grid_size), dtype=np.uint8)
        pred_vertex_mask = np.zeros((pred_poly.shape[0], 1, self.grid_size, self.grid_size), dtype=np.uint8)

        # Draw Vertex mask and full polygon mask
        for b in range(pred_poly.shape[0]):
            masked_poly = utils.get_masked_poly(pred_poly[b], self.grid_size)
            xy_poly = utils.class_to_xy(masked_poly, self.grid_size)

            utils.get_vertices_mask(xy_poly, pred_vertex_mask[b,0])
            utils.draw_poly(pred_mask[b,0], xy_poly)

        pred_mask = torch.from_numpy(pred_mask).to(self.device).to(torch.float32)
        pred_vertex_mask = torch.from_numpy(pred_vertex_mask).to(self.device).to(torch.float32)

        inp = torch.cat([feats, last_rnn_state[0][0], last_rnn_state[1][0], pred_mask, pred_vertex_mask],
            dim=1)

        conv1 = self.conv1(inp)
        conv2 = self.conv2(conv1)
        conv2 = conv2.view(conv2.size(0), -1)
        pred_iou = self.fc(conv2)

        return pred_iou.view(-1)