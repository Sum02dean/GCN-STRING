import os
import sys
sys.path.append("../")  # nopep8
import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import networkx as nx
import glob
import pandas as pd
import scipy
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.utils import convert
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.loader import DataLoader
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool, global_max_pool
from utilities.gcn_utills import *


# Import the data
graph_dir_path = '/mnt/mnemo5/sum02dean/sl_projects/GCN/GCN-STRING/src/scripts/graph_data'
labels_dir_path = '/mnt/mnemo5/sum02dean/sl_projects/GCN/GCN-STRING/src/scripts/graph_labels'

graph_files = glob.glob(os.path.join(graph_dir_path, '*'))
graph_labels = glob.glob(os.path.join(labels_dir_path, '*'))
graph_labels = pd.read_csv(graph_labels[0])

# Create positive and negative sets
positives = []
pos_labels = []
negatives = []
neg_labels = []

for i, file in enumerate(graph_files):
    obs, label = get_label(file, graph_labels)

    if label == 1:
        positives.append(obs)
        pos_labels.append([1, 0])
    else:
        negatives.append(obs)
        neg_labels.append([0, 1])

# Balance the number of negatives with number of positives
negatives = np.random.choice(negatives, size=len(positives), replace=False)

# Read in the positives
pos_graphs = read_graphs(positives)
neg_graphs = read_graphs(negatives)

# Format graphs
positive_graphs = format_graphs(pos_graphs, label=1)
negative_graphs = format_graphs(neg_graphs, label=0)

# Make sure number of negative graphs equal number of positives graphs
assert (len(negative_graphs) == len(positive_graphs))

# Combine negative and positive data
balanced_graphs = positive_graphs + negative_graphs

# Split into train and test
train_idx = np.random.choice(a=[False, True], size=len(balanced_graphs))
test_idx = ~train_idx

# Convert range to array
full_idx = np.array(range(len(balanced_graphs)))

# Grab indices using Boolean array
tr_idx = full_idx[train_idx]
te_idx = full_idx[test_idx]

# Slice train and test data
train_data = [balanced_graphs[x] for x in tr_idx]
test_data = [balanced_graphs[x] for x in te_idx]

# Select appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Build model
model = GCN(hidden_channels=200)
print(model)

# Configs
BATCH_SIZE = 50
EPOCHS = 50
LEARNING_RATE = 0.002

# Optimizers & Criterion
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.BCEWithLogitsLoss()

# Data-loaders
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

# Train loop
print("Begining training!")
model.train()
model = model.double()

for epoch in range(EPOCHS):
    epoch_loss = 0
    for data in tqdm(train_loader):

        # Grab inputs
        X, Y, EI = data.x, data.y, data.edge_index
        B = data.batch

        # Zero the gradient
        optimizer.zero_grad()

        # Compute model outputs
        logits = model(X, EI, B).flatten()
        probas = torch.sigmoid(logits)

        # Grab the loss and step gradients
        loss = criterion(probas, Y)
        epoch_loss += loss.item()

        # Backpropogate the loss
        loss.backward()
        optimizer.step()
        # Print the epochs
    print(epoch_loss)

model.eval()
with torch.no_grad():
    for data in tqdm(train_loader):

        # Grab inputs
        X, Y, EI = data.x, data.y, data.edge_index
        B = data.batch

        # # Compute preds
        logits = model(X, EI, B).flatten()
        acc = binary_acc(logits, Y)
        print(acc)
