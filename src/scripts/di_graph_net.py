# Libraries
import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.utils import convert
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.loader import DataLoader
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool, global_max_pool


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels=64):
        """
        This class takes in 2 inputs. Each input is a graph representing a single protein structure in a protein pair.
        Each graph is processed by seperate nerual networks, the outputs of which are then concatenated into a single
        hiddden state. Finally this hidden state is passed through a simple dense layer for feature extraction, and then 
        returned for use with a BCEWithLogitLoss (recommended for binary classificatio) method inside the main training loop.

        :param hidden_channels: the number of hidden neurons to use, defaults to 64
        :type hidden_channels: int, optional

        """
        super(GCN, self).__init__()

        # Parameters
        self.num_node_features = 16
        self.num_classes = 1
        self.hidden_channels = hidden_channels

        #### Graph 1 ####
        # Layers (consider using class SAGEConv instead)
        self.g1_conv_1 = GCNConv(self.num_node_features, self.hidden_channels)
        self.g1_linear_1 = Linear(self.hidden_channels, self.hidden_channels)

        # Paramteric RelU (prelu)
        self.g1_prelu_1 = torch.nn.PReLU()
        self.g1_prelu_2 = torch.nn.PReLU()

        # Batch normilization
        self.g1_batch_norm_1 = torch.nn.BatchNorm1d(
            num_features=self.hidden_channels, track_running_stats=False,
            momentum=None)

        self.g1_batch_norm_2 = torch.nn.BatchNorm1d(
            num_features=self.hidden_channels,
            track_running_stats=False, momentum=None)

        #### Graph 2 ####
        # Layers (consider using class SAGEConv instead)
        self.g2_conv_1 = GCNConv(self.num_node_features, self.hidden_channels)
        self.g2_linear_1 = Linear(self.hidden_channels, self.hidden_channels)

        # Paramteric RelU (prelu)
        self.g2_prelu_1 = torch.nn.PReLU()
        self.g2_prelu_2 = torch.nn.PReLU()

        # Batch normilization
        self.g2_batch_norm_1 = torch.nn.BatchNorm1d(
            num_features=self.hidden_channels, track_running_stats=False,
            momentum=None)

        self.g2_batch_norm_2 = torch.nn.BatchNorm1d(
            num_features=self.hidden_channels,
            track_running_stats=False, momentum=None)

        #### Concatenated graphs ####

        # Layers
        self.cat_linear = Linear(
            self.hidden_channels * 2, self.hidden_channels)

        # Paramteric RelU
        self.cat_prelu = torch.nn.PReLU()

        # Batch normilization
        self.cat_batch_norm = torch.nn.BatchNorm1d(
            num_features=self.hidden_channels, track_running_stats=False,
            momentum=None)

    def forward(self, x_1, x_2, edge_index_1, edge_index_2, batch_1, batch_2):

        #### Graph 1 ####
        # Conv block graph 1
        x_1 = self.g1_conv_1(x_1, edge_index_1)
        x_1 = self.g1_batch_norm_1(x_1)
        x_1 = self.g1_prelu_1(x_1)
        x_1 = global_max_pool(x_1, batch_1)

        # Linear block graph 1
        x_1 = self.g1_linear_1(x_1)
        x_1 = self.g1_batch_norm_2(x_1)
        x_1 = self.g1_prelu_2(x_1)
        ################

        #### Graph 2 ####
        # Conv block graph 2
        x_2 = self.g2_conv_1(x_2, edge_index_2)
        x_2 = self.g2_batch_norm_1(x_2)
        x_2 = self.g2_prelu_1(x_2)
        x_2 = global_max_pool(x_2, batch_2)

        # Linear block graph 2
        x_2 = self.g2_linear_1(x_2)
        x_2 = self.g2_batch_norm_2(x_2)
        x_2 = self.g2_prelu_2(x_2)
        ################

        #### di_graph ####
        x = torch.concat((x_1, x_2), dim=0)
        x = self.cat_batch_norm(x)
        x = self.cat_linear(x)
        return x
