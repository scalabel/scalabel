import torch
import torch.nn as nn
import torch.nn.functional as F


class FirstVertex(nn.Module):
    def __init__(self, feats_dim, feats_channels):
        super(FirstVertex, self).__init__()
        self.grid_size = feats_dim

        self.edge_conv = nn.Conv2d(in_channels=feats_channels, out_channels=16, kernel_size=3, padding=1)
        self.edge_fc = nn.Linear(in_features=feats_dim ** 2 * 16, out_features=feats_dim ** 2)
        self.vertex_conv = nn.Conv2d(in_channels=feats_channels, out_channels=16, kernel_size=3, padding=1)
        self.vertex_fc = nn.Linear(in_features=feats_dim ** 2 * 16, out_features=feats_dim ** 2)

    def forward(self, feats, beam_size=1):
        batch_size = feats.shape[0]
        conv_edge = self.edge_conv(feats)
        conv_edge = F.relu(conv_edge, inplace=True)
        edge_logits = self.edge_fc(conv_edge.view(batch_size, -1))

        # Different from before, this used to take conv_edge as input before
        conv_vertex = self.vertex_conv(feats)
        conv_vertex = F.relu(conv_vertex)
        vertex_logits = self.vertex_fc(conv_vertex.view(batch_size, -1))
        logprobs = F.log_softmax(vertex_logits, -1)

        # Sample a first vertex
        logprob, pred_first = torch.topk(logprobs, beam_size, dim=-1)

        # Remove the last dimension if it is 1
        pred_first = torch.squeeze(pred_first, dim=-1)
        logprob = torch.squeeze(logprob, dim=-1)

        return edge_logits, vertex_logits, logprob, pred_first
