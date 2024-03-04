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

# def load_data(path="data/cora/", dataset="cora"):#modified from code: pygcn
#     """Load citation network dataset (cora only for now)"""
#     #input: idx_features_labels, adj
#     #idx,labels are not required to be processed in advance
#     #adj: save in the form of edges. idx1 idx2 
#     #output: adj, features, labels are all torch.tensor, in the dense form
#     #-------------------------------------------------------

#     print('Loading {} dataset...'.format(dataset))

#     idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
#                                         dtype=np.dtype(str))
#     features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
#     labels = idx_features_labels[:, -1]
#     set_labels = set(labels)
#     classes_dict = {c: np.arange(len(set_labels))[i] for i, c in enumerate(set_labels)}
#     classes_dict = {'Neural_Networks': 0, 'Reinforcement_Learning': 1, 'Probabilistic_Methods': 2, 'Case_Based': 3, 'Theory': 4, 'Rule_Learning': 5, 'Genetic_Algorithms': 6}

#     #ipdb.set_trace()
#     labels = np.array(list(map(classes_dict.get, labels)))

#     # build graph
#     idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
#     idx_map = {j: i for i, j in enumerate(idx)}
#     edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
#                                     dtype=np.int32)
#     edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
#                      dtype=np.int32).reshape(edges_unordered.shape)
#     adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
#                         shape=(labels.shape[0], labels.shape[0]),
#                         dtype=np.float32)

#     # build symmetric adjacency matrix
#     adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

#     features = normalize(features)

#     features = torch.FloatTensor(np.array(features.todense()))
#     labels = torch.LongTensor(labels)

#     utils.print_edges_num(adj.todense(), labels)

#     adj = sparse_mx_to_torch_sparse_tensor(adj)
#     #adj = torch.FloatTensor(np.array(adj.todense()))

#     return adj, features, labels


# def Extract_graph(edgelist, fake_node, node_num):
    
#     node_list = range(node_num+1)[1:]
#     node_set = set(node_list)
#     adj_1 = sp.coo_matrix((np.ones(len(edgelist)), (edgelist[:, 0], edgelist[:, 1])), shape=(edgelist.max()+1, edgelist.max()+1), dtype=np.float32)
#     adj_1 = adj_1 + adj_1.T.multiply(adj_1.T > adj_1) - adj_1.multiply(adj_1.T > adj_1)
#     adj_csr = adj_1.tocsr()
#     for i in np.arange(node_num):
#         for j in adj_csr[i].nonzero()[1]:
#             node_set.add(j)

#     node_set_2 = node_set
#     '''
#     node_set_2 = set(node_list)
#     for i in node_set:
#         for j in adj_csr[i].nonzero()[1]:
#             node_set_2.add(j)
#     '''
#     node_list = np.array(list(node_set_2))
#     node_list = np.sort(node_list)
#     adj_new = adj_csr[node_list,:]

#     node_mapping = dict(zip(node_list, range(0, len(node_list), 1)))

#     edge_list = []
#     for i in range(len(node_list)):
#         for j in adj_new[i].nonzero()[1]:
#             if j in node_list:
#                 edge_list.append([i, node_mapping[j]])

#     edge_list = np.array(edge_list)
#     #adj_coo_new = sp.coo_matrix((np.ones(len(edge_list)), (edge_list[:,0], edge_list[:,1])), shape=(len(node_list), len(node_list)), dtype=np.float32)

#     label_new = np.array(list(map(lambda x: 1 if x in fake_node else 0, node_list)))
#     np.savetxt('data/twitter/sub_twitter_edges', edge_list,fmt='%d')
#     np.savetxt('data/twitter/sub_twitter_labels', label_new,fmt='%d')

#     return

# def load_data_twitter():
#     adj_path = 'data/twitter/twitter.csv'
#     fake_id_path = 'data/twitter/twitter_fake_ids.csv'

#     adj = np.loadtxt(adj_path, delimiter=',', skiprows=1)#(total: 16011444 edges, 5384162 nodes)
#     adj = adj.astype(int)
#     adj = np.array(adj,dtype=int)
#     fake_node = np.genfromtxt(fake_id_path, delimiter=',',skip_header=1, usecols=(0), dtype=int)#(12437)
    
#     #'''#using broad walk
#     if False:
#         Extract_graph(adj, fake_node, node_num=1000)

#     #'''

#     '''generated edgelist for deepwalk for embedding
#     np.savetxt('data/twitter/twitter_edges', adj,fmt='%d')
#     '''

#     #process adj:
#     adj[adj>50000] = 0 #save top 50000 node, start from 1
#     adj = sp.coo_matrix((np.ones(len(adj)), (adj[:, 0], adj[:, 1])), shape=(adj.max()+1, adj.max()+1), dtype=np.float32)
#     adj = np.array(adj.todense())
#     adj = adj[1:, 1:]
#     adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
#     adj = adj.tocoo()

#     fake_node = np.sort(fake_node)
#     fake_node = fake_node[fake_node<=50000]
#     fake_id = fake_node-1

#     #process label:
#     labels = np.zeros((50000,),dtype=int)
#     labels[fake_id] = 1


#     #filtering out outliers:
#     node_degree = adj.sum(axis=1)
#     chosen_idx = np.arange(50000)[node_degree>=4]
#     ipdb.set_trace()


#     #embed need to be read sequentially, due to the size
#     embed = np.genfromtxt('data/twitter/twitter.embeddings_64', max_rows=50000)
#     feature = np.zeros((embed.shape[0],embed.shape[1]-1))
#     feature[embed[:,0].astype(int),:] = embed[:,1:]
#     features = normalize(feature)

#     adj = adj[chosen_idx,:][:,chosen_idx]     #shape:
#     labels = labels[chosen_idx]     #shape:
#     features = features[chosen_idx]

    

#     features = torch.FloatTensor(np.array(features.todense()))
#     labels = torch.LongTensor(labels)

#     utils.print_edges_num(adj.todense(), labels)

#     adj = sparse_mx_to_torch_sparse_tensor(adj)

#     return adj, features, labels

# def load_sub_data_twitter():
#     adj_path = 'data/twitter/sub_twitter_edges'
#     fake_id_path = 'data/twitter/sub_twitter_labels'

#     adj = np.loadtxt(adj_path, delimiter=' ', dtype=int)#
#     adj = np.array(adj,dtype=int)
#     labels = np.genfromtxt(fake_id_path, dtype=int)#(63167)
    
#     #process adj:
#     adj = sp.coo_matrix((np.ones(len(adj)), (adj[:, 0], adj[:, 1])), shape=(adj.max()+1, adj.max()+1), dtype=np.float32)
#     adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

#     #filtering out outliers:
#     node_degree = np.array(adj.sum(axis=1)).reshape(-1)
#     chosen_idx = np.arange(adj.shape[0])[node_degree>=4]#44982 nodes were left

#     #embed need to be read sequentially, due to the size
#     embed = np.genfromtxt('data/twitter/sub_node_embedding_64', skip_header=1)
#     feature = np.zeros((embed.shape[0],embed.shape[1]-1))
#     feature[embed[:,0].astype(int),:] = embed[:,1:]
#     features = normalize(feature)

#     features = torch.FloatTensor(np.array(features))
#     labels = torch.LongTensor(labels)

#     utils.print_edges_num(adj.todense(), labels)

#     adj = sparse_mx_to_torch_sparse_tensor(adj)

#     return adj, features, labels

# def load_data_Blog():#
#     #--------------------
#     #
#     #--------------------
#     mat = loadmat('data/BlogCatalog/blogcatalog.mat')
#     adj = mat['network']
#     label = mat['group']

#     embed = np.loadtxt('data/BlogCatalog/blogcatalog.embeddings_64')
#     feature = np.zeros((embed.shape[0],embed.shape[1]-1))
#     feature[embed[:,0].astype(int),:] = embed[:,1:]

#     features = normalize(feature)
#     labels = np.array(label.todense().argmax(axis=1)).squeeze()

#     labels[labels>16] = labels[labels>16]-1

#     print("change labels order, imbalanced classes to the end.")
#     #ipdb.set_trace()
#     labels = refine_label_order(labels)

#     features = torch.FloatTensor(features)
#     labels = torch.LongTensor(labels)

#     #adj = torch.FloatTensor(np.array(adj.todense()))
#     adj = sparse_mx_to_torch_sparse_tensor(adj)

#     return adj, features, labels

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

        # important modification by Yandong
        # train_mask, val_mask, test_mask = graph_split(dataset_str, graph.ndata['label'], train_ratio=0.3,
        #                                               folds=args.ntrials)

        # x_data = torch.tensor(normalize_features(graph.ndata['feature'], norm_row=False), dtype=torch.float)

        # return x_data, graph.ndata['feature'].size()[-1], graph.ndata['label'], 2, \
        #     train_mask, val_mask, test_mask, graph
        ## graph.ndata['train_mask'].bool(), graph.ndata['val_mask'].bool(), graph.ndata['test_mask'].bool()
    
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

        # train_mask, val_mask, test_mask = graph_split(dataset_str, graph.ndata['label'], train_ratio=args.train_ratio,
        #                                               folds=args.ntrials)

        # graph.ndata['feature'] = torch.tensor(normalize_features(graph.ndata['feature'], norm_row=True),
        #                                       dtype=torch.float)

        # return graph.ndata['feature'], graph.ndata['feature'].size()[-1], graph.ndata['label'], 2, \
        #     train_mask, val_mask, test_mask, graph

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
        random.seed(42)

        full_graph = dgl.graph((data.edge_index[0], data.edge_index[1]))
        full_graph.ndata['feature'] = data.x
        full_graph.ndata['label'] = data.y.type(torch.LongTensor)

        # Select a random subset of nodes
        subset_size = int(full_graph.num_nodes() * 0.1)  # For example, 10% of the full dataset
        subset_nodes = random.sample(range(full_graph.num_nodes()), subset_size)

        # Extract the subgraph containing only the selected nodes
        subgraph = full_graph.subgraph(subset_nodes)
        subgraph.ndata['feature'] = full_graph.ndata['feature'][subset_nodes]
        subgraph.ndata['label'] = full_graph.ndata['label'][subset_nodes]

        # train_mask, val_mask, test_mask = graph_split(dataset_str, graph.ndata['label'], train_ratio=args.train_ratio, folds=args.ntrials)

        # graph.ndata['feature'] = torch.tensor(normalize_features(graph.ndata['feature'], norm_row=True), dtype=torch.float)

        # return graph.ndata['feature'], graph.ndata['feature'].size()[-1], graph.ndata['label'], 2, \
        #     train_mask, val_mask, test_mask, graph
    
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

