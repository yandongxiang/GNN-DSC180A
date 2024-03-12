import dgl
import torch
import random
import matplotlib
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt

from dgl.data import FraudYelpDataset, FraudAmazonDataset
from pygod.utils import load_data as pygod_load_data

# Yelp
dataset = FraudYelpDataset()
full_graph = dataset[0]

full_graph = dgl.to_homogeneous(full_graph, ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
full_graph = dgl.add_self_loop(full_graph)

# Select a random subset
random.seed(42)
subset_size = int(full_graph.num_nodes() * 0.03)
subset_nodes = random.sample(range(full_graph.num_nodes()), subset_size)

yelp_graph = full_graph.subgraph(subset_nodes)


# Amazon
dataset = FraudAmazonDataset()
full_graph = dataset[0]

full_graph = dgl.to_homogeneous(full_graph, ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
full_graph = dgl.add_self_loop(full_graph)

# Select a random subset
random.seed(42)
subset_size = int(full_graph.num_nodes() * 0.1)
subset_nodes = random.sample(range(full_graph.num_nodes()), subset_size)

amazon_graph = full_graph.subgraph(subset_nodes)

# Reddit
data = pygod_load_data('reddit')
random.seed(42)
full_graph = dgl.graph((data.edge_index[0], data.edge_index[1]))
full_graph.ndata['feature'] = data.x
full_graph.ndata['label'] = data.y.type(torch.LongTensor)

# Select a random subset of nodes
subset_size = int(full_graph.num_nodes() * 0.1)  # For example, 10% of the full dataset
subset_nodes = random.sample(range(full_graph.num_nodes()), subset_size)

# Extract the subgraph containing only the selected nodes
reddit_graph = full_graph.subgraph(subset_nodes)
reddit_graph.ndata['feature'] = full_graph.ndata['feature'][subset_nodes]
reddit_graph.ndata['label'] = full_graph.ndata['label'][subset_nodes]

# Number of Nodes and Edges
print('yelp:')
num_nodes = yelp_graph.num_nodes()
num_edges = yelp_graph.num_edges()

print(f"Number of Nodes: {num_nodes}")
print(f"Number of Edges: {num_edges}")

labels_tensor = yelp_graph.ndata['label']  # Assuming this is how you access labels

unique_labels, counts = torch.unique(labels_tensor, return_counts=True)
print(f'Number of normal users in yelp dataset is {counts[0]}')
print(f'Number of Fraudulent users in yelp dataset is {counts[1]}')

# Number of Nodes and Edges
print('amazon:')
num_nodes = amazon_graph.num_nodes()
num_edges = amazon_graph.num_edges()

print(f"Number of Nodes: {num_nodes}")
print(f"Number of Edges: {num_edges}")

labels_tensor = amazon_graph.ndata['label']  # Assuming this is how you access labels

unique_labels, counts = torch.unique(labels_tensor, return_counts=True)
print(f'Number of normal users in amazon dataset is {counts[0]}')
print(f'Number of Fraudulent users in amazon dataset is {counts[1]}')

# Number of Nodes and Edges
print('reddit:')
num_nodes = reddit_graph.num_nodes()
num_edges = reddit_graph.num_edges()

print(f"Number of Nodes: {num_nodes}")
print(f"Number of Edges: {num_edges}")

labels_tensor = reddit_graph.ndata['label']  # Assuming this is how you access labels

unique_labels, counts = torch.unique(labels_tensor, return_counts=True)
print(f'Number of normal users in reddit dataset is {counts[0]}')
print(f'Number of Fraudulent users in reddit dataset is {counts[1]}')

yelp_degrees = yelp_graph.in_degrees().numpy()
amazon_degrees = amazon_graph.in_degrees().numpy()
reddit_degrees = reddit_graph.in_degrees().numpy()

matplotlib.rcParams.update({'font.size': 12})

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Yelp
sns.kdeplot(yelp_degrees, ax=axes[0], label='Yelp', color='skyblue', fill=True)
axes[0].set_title('Yelp Degree Distribution')
axes[0].set_xlabel('Degree')
axes[0].set_ylabel('Density')
# Adding mean and median for Yelp
yelp_mean = np.mean(yelp_degrees)
yelp_median = np.median(yelp_degrees)
axes[0].axvline(yelp_mean, color='#009af0', linestyle='--', label=f'Mean: {yelp_mean:.2f}')
axes[0].axvline(yelp_median, color='#FCA42D', linestyle='-', label=f'Median: {yelp_median:.2f}')
axes[0].legend()

# Amazon
sns.kdeplot(amazon_degrees, ax=axes[1], bw_adjust=0.5, label='Amazon', color='lightgreen', fill=True)
axes[1].set_title('Amazon Degree Distribution')
axes[1].set_xlabel('Degree')
axes[1].set_ylabel('Density')
# Adding mean and median for Amazon
amazon_mean = np.mean(amazon_degrees)
amazon_median = np.median(amazon_degrees)
axes[1].axvline(amazon_mean, color='#009af0', linestyle='--', label=f'Mean: {amazon_mean:.2f}')
axes[1].axvline(amazon_median, color='#FCA42D', linestyle='-', label=f'Median: {amazon_median:.2f}')
axes[1].legend()

# Reddit
sns.kdeplot(reddit_degrees, ax=axes[2], bw_adjust=0.5, label='Reddit', color='salmon', fill=True)
axes[2].set_title('Reddit Degree Distribution')
axes[2].set_xlabel('Degree')
axes[2].set_ylabel('Density')
# Adding mean and median for Reddit
reddit_mean = np.mean(reddit_degrees)
reddit_median = np.median(reddit_degrees)
axes[2].axvline(reddit_mean, color='#009af0', linestyle='--', label=f'Mean: {reddit_mean:.2f}')
axes[2].axvline(reddit_median, color='#FCA42D', linestyle='-', label=f'Median: {reddit_median:.2f}')
axes[2].legend()

plt.tight_layout()
plt.show()

nx_graph = dgl.to_networkx(yelp_graph)
self_loops = list(nx.selfloop_edges(nx_graph))
nx_graph.remove_edges_from(self_loops)


node_labels_tensor = yelp_graph.ndata['label']


unique_labels = node_labels_tensor.unique()
label_color_map = {label.item(): plt.cm.tab10(i / len(unique_labels)) for i, label in enumerate(unique_labels)}

node_colors = [label_color_map[label.item()] for label in node_labels_tensor]

node_sizes = 1

plt.figure(figsize=(8, 8))
pos = nx.spring_layout(nx_graph, seed=42, k=0.04)
nx.draw(nx_graph, pos, node_color=node_colors, with_labels=False, node_size=50)
plt.title('yelp dataset', fontsize=20)
plt.show()

nx_graph = dgl.to_networkx(amazon_graph)
self_loops = list(nx.selfloop_edges(nx_graph))
nx_graph.remove_edges_from(self_loops)


node_labels_tensor = amazon_graph.ndata['label']


unique_labels = node_labels_tensor.unique()
label_color_map = {label.item(): plt.cm.tab10(i / len(unique_labels)) for i, label in enumerate(unique_labels)}

node_colors = [label_color_map[label.item()] for label in node_labels_tensor]

node_sizes = 1

plt.figure(figsize=(8, 8))
pos = nx.spring_layout(nx_graph, seed=42, k=0.15)
nx.draw(nx_graph, pos, node_color=node_colors, with_labels=False, node_size=50)
plt.title('amazon dataset', fontsize=20)
plt.show()


nx_graph = dgl.to_networkx(reddit_graph)
self_loops = list(nx.selfloop_edges(nx_graph))
nx_graph.remove_edges_from(self_loops)


node_labels_tensor = reddit_graph.ndata['label']


unique_labels = node_labels_tensor.unique()
label_color_map = {label.item(): plt.cm.tab10(i / len(unique_labels)) for i, label in enumerate(unique_labels)}

node_colors = [label_color_map[label.item()] for label in node_labels_tensor]

node_sizes = 1

plt.figure(figsize=(8, 8))
pos = nx.spring_layout(nx_graph, seed=42, k=0.1)
nx.draw(nx_graph, pos, node_color=node_colors, with_labels=False, node_size=50)
plt.title('reddit dataset', fontsize=20)
plt.show()


# Yelp
dataset = FraudYelpDataset()
full_graph = dataset[0]

full_graph = dgl.to_homogeneous(full_graph, ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
full_graph = dgl.add_self_loop(full_graph)

yelp_graph = full_graph


# Amazon
dataset = FraudAmazonDataset()
full_graph = dataset[0]

full_graph = dgl.to_homogeneous(full_graph, ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
full_graph = dgl.add_self_loop(full_graph)

amazon_graph = full_graph

# Reddit
data = pygod_load_data('reddit')
random.seed(42)
full_graph = dgl.graph((data.edge_index[0], data.edge_index[1]))
full_graph.ndata['feature'] = data.x
full_graph.ndata['label'] = data.y.type(torch.LongTensor)

reddit_graph = full_graph

yelp_degrees = yelp_graph.in_degrees().numpy()
amazon_degrees = amazon_graph.in_degrees().numpy()
reddit_degrees = reddit_graph.in_degrees().numpy()

matplotlib.rcParams.update({'font.size': 12})

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Yelp
sns.kdeplot(yelp_degrees, ax=axes[0], label='Yelp', color='skyblue', fill=True)
axes[0].set_title('Yelp Degree Distribution')
axes[0].set_xlabel('Degree')
axes[0].set_ylabel('Density')
# Adding mean and median for Yelp
yelp_mean = np.mean(yelp_degrees)
yelp_median = np.median(yelp_degrees)
axes[0].axvline(yelp_mean, color='#009af0', linestyle='--', label=f'Mean: {yelp_mean:.2f}')
axes[0].axvline(yelp_median, color='#FCA42D', linestyle='-', label=f'Median: {yelp_median:.2f}')
axes[0].legend()

# Amazon
sns.kdeplot(amazon_degrees, ax=axes[1], bw_adjust=0.5, label='Amazon', color='lightgreen', fill=True)
axes[1].set_title('Amazon Degree Distribution')
axes[1].set_xlabel('Degree')
axes[1].set_ylabel('Density')
# Adding mean and median for Amazon
amazon_mean = np.mean(amazon_degrees)
amazon_median = np.median(amazon_degrees)
axes[1].axvline(amazon_mean, color='#009af0', linestyle='--', label=f'Mean: {amazon_mean:.2f}')
axes[1].axvline(amazon_median, color='#FCA42D', linestyle='-', label=f'Median: {amazon_median:.2f}')
axes[1].legend()

# Reddit
sns.kdeplot(reddit_degrees, ax=axes[2], bw_adjust=0.5, label='Reddit', color='salmon', fill=True)
axes[2].set_title('Reddit Degree Distribution')
axes[2].set_xlabel('Degree')
axes[2].set_ylabel('Density')
# Adding mean and median for Reddit
reddit_mean = np.mean(reddit_degrees)
reddit_median = np.median(reddit_degrees)
axes[2].axvline(reddit_mean, color='#009af0', linestyle='--', label=f'Mean: {reddit_mean:.2f}')
axes[2].axvline(reddit_median, color='#FCA42D', linestyle='-', label=f'Median: {reddit_median:.2f}')
axes[2].legend()

plt.tight_layout()
plt.show()
