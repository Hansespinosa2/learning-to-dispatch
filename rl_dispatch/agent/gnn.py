import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, SAGEConv, global_mean_pool, global_max_pool, GATv2Conv
import random
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import numpy as np

class GNN(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, action_space_dim):
        super(GNN, self).__init__()
        self.conv1 = GATv2Conv(in_channels=node_feat_dim, edge_dim=edge_feat_dim, out_channels=64, heads=1)
        self.conv2 = GATv2Conv(in_channels=64, edge_dim=edge_feat_dim, out_channels=64, heads=1)
        self.fc0 = nn.Linear(2,2)
        self.fc1 = nn.Linear(64 + 2, 32)  # Adding 2 to the input dimension due to concatenation
        self.fc2 = nn.Linear(32, action_space_dim)

    def forward(self, data, node_idx):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = torch.relu(x)

        y = data.y.reshape(-1, 2)
        repeat_factor = x.shape[0] // y.shape[0]
        y = y.repeat_interleave(repeat_factor, dim=0)
        y = self.fc0(y)
        y = torch.relu(y)

        x = torch.cat([x, y], dim=1)

        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x[node_idx]

    