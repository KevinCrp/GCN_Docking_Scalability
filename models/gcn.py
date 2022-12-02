from typing import List

import torch
import torch_geometric.nn as pyg_nn


class MolGCN(torch.nn.Module):
    def __init__(self,
                 gcn_in_channels: int,
                 gcn_hidden_channels: int,
                 gcn_num_layers: int,
                 gcn_out_channels: int,
                 mlp_channel_list: List[int]):
        super().__init__()
        self.gcn = pyg_nn.GCN(in_channels=gcn_in_channels,
                              hidden_channels=gcn_hidden_channels,
                              num_layers=gcn_num_layers,
                              out_channels=gcn_out_channels)
        self.mlp = pyg_nn.MLP(channel_list=mlp_channel_list)

    def forward(self, x, edge_index, batch=None):
        x_gcn = self.gcn(x, edge_index)
        x_mlp = self.mlp(x_gcn)
        y = pyg_nn.global_mean_pool(x_mlp, batch)
        return y
