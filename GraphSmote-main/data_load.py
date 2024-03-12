import argparse
import scipy.sparse as sp
import numpy as np
import torch
import ipdb
from scipy.io import loadmat
import utils
from collections import defaultdict
import random

import warnings

import os
import dgl
import torch

from sklearn.model_selection import train_test_split
from utils import normalize_features

from dgl.data import FraudYelpDataset, FraudAmazonDataset
from dgl.data.utils import load_graphs, save_graphs
import pandas as pd
from pygod.utils import load_data as pygod_load_data

warnings.simplefilter("ignore")

IMBALANCE_THRESH = 101

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--target', type=int, default=4)    
    parser.add_argument('--k', type = int, default = 5)
    if hasattr(Trainer, 'add_args'):
        Trainer.add_args(parser)
    
    return parser

def refine_label_order(labels):
    max_label = labels.max()
    j = 0

    for i in range(labels.max(),0,-1):
        if sum(labels==i) >= IMBALANCE_THRESH and i>j:
            while sum(labels==j) >= IMBALANCE_THRESH and i>j:
                j = j+1
            if i > j:
                head_ind = labels == j
                tail_ind = labels == i
                labels[head_ind] = i
                labels[tail_ind] = j
                j = j+1
            else:
                break
        elif i <= j:
            break

    return labels

# DataLoading 

def load_data(args):
    dataset_str = args

    if dataset_str == 'yelp':
        dataset = FraudYelpDataset()
        full_graph = dataset[0]

        full_graph = dgl.to_homogeneous(full_graph, ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
        full_graph = dgl.add_self_loop(full_graph)

        # Select a random subset
        random.seed(42)
        subset_size = int(full_graph.num_nodes() * 0.03)
        subset_nodes = random.sample(range(full_graph.num_nodes()), subset_size)

        graph = full_graph.subgraph(subset_nodes)
    
        # Normalize features
        features = torch.tensor(normalize_features(graph.ndata['feature'], norm_row=False), dtype=torch.float)
        labels = graph.ndata['label']

        # Convert the DGL graph to an adjacency matrix
        src, dst = graph.edges()
        data = torch.ones(src.size(0))
        adj = sp.coo_matrix((data.numpy(), (src.numpy(), dst.numpy())), shape=(graph.num_nodes(), graph.num_nodes()))

         # Ensure the adjacency matrix is of type float
        adj = adj.astype(np.float32)

        # Symmetrize the adjacency matrix if your model expects an undirected graph
        # Here, we manually ensure that for every (i, j), there's also (j, i)
        adj = adj + adj.T - sp.coo_matrix(adj.multiply(adj.T > adj))

        # Convert the adjacency matrix from SciPy COO matrix format to PyTorch sparse tensor format
        # Ensure adj is in COO format
        if not sp.isspmatrix_coo(adj):
            adj = sp.coo_matrix(adj)
        adj = torch.sparse_coo_tensor(torch.LongTensor([adj.row, adj.col]), torch.FloatTensor(adj.data), adj.shape)

        # Return the adjacency matrix, features, and labels in the desired format
        return adj, features, labels

    elif dataset_str == 'amazon':
        dataset = FraudAmazonDataset()
        full_graph = dataset[0]

        full_graph = dgl.to_homogeneous(full_graph, ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
        full_graph = dgl.add_self_loop(full_graph)

        # Select a random subset
        random.seed(42)
        subset_size = int(full_graph.num_nodes() * 0.1)
        subset_nodes = random.sample(range(full_graph.num_nodes()), subset_size)

        graph = full_graph.subgraph(subset_nodes)

        # Normalize features
        features = torch.tensor(normalize_features(graph.ndata['feature'], norm_row=False), dtype=torch.float)
        labels = graph.ndata['label']

        # Convert the DGL graph to an adjacency matrix
        src, dst = graph.edges()
        data = torch.ones(src.size(0))
        adj = sp.coo_matrix((data.numpy(), (src.numpy(), dst.numpy())), shape=(graph.num_nodes(), graph.num_nodes()))

         # Ensure the adjacency matrix is of type float
        adj = adj.astype(np.float32)

        # Symmetrize the adjacency matrix if your model expects an undirected graph
        # Here, we manually ensure that for every (i, j), there's also (j, i)
        adj = adj + adj.T - sp.coo_matrix(adj.multiply(adj.T > adj))

        # Convert the adjacency matrix from SciPy COO matrix format to PyTorch sparse tensor format
        # Ensure adj is in COO format
        if not sp.isspmatrix_coo(adj):
            adj = sp.coo_matrix(adj)
        adj = torch.sparse_coo_tensor(torch.LongTensor([adj.row, adj.col]), torch.FloatTensor(adj.data), adj.shape)

        # Return the adjacency matrix, features, and labels in the desired format
        return adj, features, labels

    elif dataset_str == 'reddit':
        data = pygod_load_data(dataset_str)

        full_graph = dgl.graph((data.edge_index[0], data.edge_index[1]))
        full_graph.ndata['feature'] = data.x
        full_graph.ndata['label'] = data.y.type(torch.LongTensor)

        # Select a random subset of nodes
        random.seed(42)
        subset_size = int(full_graph.num_nodes() * 0.1)  # For example, 10% of the full dataset
        subset_nodes = random.sample(range(full_graph.num_nodes()), subset_size)

        # Extract the subgraph containing only the selected nodes
        subgraph = full_graph.subgraph(subset_nodes)
        subgraph.ndata['feature'] = full_graph.ndata['feature'][subset_nodes]
        subgraph.ndata['label'] = full_graph.ndata['label'][subset_nodes]
    
        # Normalize features
        features = torch.tensor(normalize_features(subgraph.ndata['feature'], norm_row=False), dtype=torch.float)
        labels = subgraph.ndata['label']

        # Convert the DGL graph to an adjacency matrix
        src, dst = subgraph.edges()
        data = torch.ones(src.size(0))
        adj = sp.coo_matrix((data.numpy(), (src.numpy(), dst.numpy())), shape=(subgraph.num_nodes(), subgraph.num_nodes()))

         # Ensure the adjacency matrix is of type float
        adj = adj.astype(np.float32)

        # Symmetrize the adjacency matrix if your model expects an undirected graph
        # Here, we manually ensure that for every (i, j), there's also (j, i)
        adj = adj + adj.T - sp.coo_matrix(adj.multiply(adj.T > adj))

        # Convert the adjacency matrix from SciPy COO matrix format to PyTorch sparse tensor format
        # Ensure adj is in COO format
        if not sp.isspmatrix_coo(adj):
            adj = sp.coo_matrix(adj)
        adj = torch.sparse_coo_tensor(torch.LongTensor([adj.row, adj.col]), torch.FloatTensor(adj.data), adj.shape)

        # Return the adjacency matrix, features, and labels in the desired format
        return adj, features, labels

    else:
        raise NotImplementedError


def graph_split(dataset, labels, train_ratio=0.01, folds=5):
    """split dataset into train and test

    Args:
        dataset (str): name of dataset
        labels (list): list of labels of nodes
    """
    assert dataset in ['amazon', 'yelp', 'reddit']
    if dataset == 'amazon':
        index = np.array(range(3305, len(labels)))
        stratify_labels = labels[3305:]

    elif dataset == 'yelp' or dataset == 'reddit':
        index = np.array(range(len(labels)))
        stratify_labels = labels

    else:
        index = np.array(range(46564))
        stratify_labels = labels[:46564]

    # generate mask
    train_mask, valid_mask, test_mask = [], [], []

    for fold in range(folds):
        idx_train, idx_test = train_test_split(index,
                                               stratify=stratify_labels,
                                               train_size=train_ratio,
                                               random_state=fold,
                                               shuffle=True)

        idx_valid, idx_test = train_test_split(idx_test,
                                               stratify=np.array(labels)[idx_test],
                                               test_size=2.0 / 3,
                                               random_state=fold, shuffle=True)

        train_mask_fold = torch.BoolTensor([False for _ in range(len(labels))])
        valid_mask_fold = torch.BoolTensor([False for _ in range(len(labels))])
        test_mask_fold = torch.BoolTensor([False for _ in range(len(labels))])
        train_mask_fold[idx_train] = True
        valid_mask_fold[idx_valid] = True
        test_mask_fold[idx_test] = True

        train_mask.append(train_mask_fold)
        valid_mask.append(valid_mask_fold)
        test_mask.append(test_mask_fold)

    train_mask = torch.vstack(train_mask)
    valid_mask = torch.vstack(valid_mask)
    test_mask = torch.vstack(test_mask)

    return train_mask, valid_mask, test_mask


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def norm_sparse(adj):#normalize a torch dense tensor for GCN, and change it into sparse.
    adj = adj + torch.eye(adj.shape[0]).to(adj)
    rowsum = torch.sum(adj,1)
    r_inv = 1/rowsum
    r_inv[torch.isinf(r_inv)] = 0.
    new_adj = torch.mul(r_inv.reshape(-1,1), adj)

    indices = torch.nonzero(new_adj).t()
    values = new_adj[indices[0], indices[1]] # modify this based on dimensionality

    return torch.sparse.FloatTensor(indices, values, new_adj.size())

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



def find_shown_index(adj, center_ind, steps = 2):
    seen_nodes = {}
    shown_index = []

    if isinstance(center_ind, int):
        center_ind = [center_ind]

    for center in center_ind:
        shown_index.append(center)
        if center not in seen_nodes:
            seen_nodes[center] = 1

    start_point = center_ind
    for step in range(steps):
        new_start_point = []
        candid_point = set(adj[start_point,:].reshape(-1, adj.shape[1]).nonzero()[:,1])
        for i, c_p in enumerate(candid_point):
            if c_p.item() in seen_nodes:
                pass
            else:
                seen_nodes[c_p.item()] = 1
                shown_index.append(c_p.item())
                new_start_point.append(c_p)
        start_point = new_start_point

    return shown_index

