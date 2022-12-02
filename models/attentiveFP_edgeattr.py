import torch
import torch_geometric.nn as pyg_nn


class MolAttentiveFP_EA(torch.nn.Module):
    def __init__(self,
                 afp_in_channels: int,
                 afp_hidden_channels: int,
                 afp_out_channels: int,
                 afp_num_layers: int,
                 afp_num_timesteps: int):
        super().__init__()
        self.afp = pyg_nn.AttentiveFP(in_channels=afp_in_channels,
                                      hidden_channels=afp_hidden_channels,
                                      out_channels=afp_out_channels,
                                      edge_dim=25,
                                      num_layers=afp_num_layers,
                                      num_timesteps=afp_num_timesteps)

    def forward(self, x, edge_index, batch=None):
        edge_attr = torch.ones(edge_index.size()[1]*25, device=x.device).reshape(
            edge_index.size()[1], 25)
        y = self.afp(x, edge_index, edge_attr, batch)
        return y
