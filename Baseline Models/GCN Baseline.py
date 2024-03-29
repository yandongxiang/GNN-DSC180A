import os

os.environ["DGLBACKEND"] = "pytorch"
from functools import partial

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn import GraphConv
from dgl.nn import GATConv
import numpy as np
from pygod.utils import load_data as pygod_load_data
from sklearn.metrics import roc_auc_score

class RGCNLayer(nn.Module):
    def __init__(
        self,
        in_feat,
        out_feat,
        num_rels,
        num_bases=-1,
        bias=None,
        activation=None,
        is_input_layer=False,
    ):
        super(RGCNLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.is_input_layer = is_input_layer

        # sanity check
        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels
        # weight bases in equation (3)
        self.weight = nn.Parameter(
            torch.Tensor(self.num_bases, self.in_feat, self.out_feat)
        )
        if self.num_bases < self.num_rels:
            # linear combination coefficients in equation (3)
            self.w_comp = nn.Parameter(
                torch.Tensor(self.num_rels, self.num_bases)
            )
        # add bias
        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_feat))
        # init trainable parameters
        nn.init.xavier_uniform_(
            self.weight, gain=nn.init.calculate_gain("relu")
        )
        if self.num_bases < self.num_rels:
            nn.init.xavier_uniform_(
                self.w_comp, gain=nn.init.calculate_gain("relu")
            )
        if self.bias:
            nn.init.xavier_uniform_(
                self.bias, gain=nn.init.calculate_gain("relu")
            )

    def forward(self, g):
        if self.num_bases < self.num_rels:
            # generate all weights from bases (equation (3))
            weight = self.weight.view(
                self.in_feat, self.num_bases, self.out_feat
            )
            weight = torch.matmul(self.w_comp, weight).view(
                self.num_rels, self.in_feat, self.out_feat
            )
        else:
            weight = self.weight
        if self.is_input_layer:

            def message_func(edges):
                # for input layer, matrix multiply can be converted to be
                # an embedding lookup using source node id
                embed = weight.view(-1, self.out_feat)
                index = edges.data[dgl.ETYPE] * self.in_feat + edges.src["id"]
                return {"msg": embed[index] * edges.data["norm"]}

        else:

            def message_func(edges):
                w = weight[edges.data[dgl.ETYPE]]
                msg = torch.bmm(edges.src["h"].unsqueeze(1), w).squeeze()
                msg = msg * edges.data["norm"]
                return {"msg": msg}

        def apply_func(nodes):
            h = nodes.data["h"]
            if self.bias:
                h = h + self.bias
            if self.activation:
                h = self.activation(h)
            return {"h": h}

        g.update_all(message_func, fn.sum(msg="msg", out="h"), apply_func)
        
        
class Model(nn.Module):
    def __init__(
        self,
        num_nodes,
        h_dim,
        out_dim,
        num_rels,
        num_bases=-1,
        num_hidden_layers=1,
    ):
        super(Model, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers

        # create rgcn layers
        self.build_model()

        # create initial features
        self.features = self.create_features()

    def build_model(self):
        self.layers = nn.ModuleList()
        # input to hidden
        i2h = self.build_input_layer()
        self.layers.append(i2h)
        # hidden to hidden
        for _ in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer()
            self.layers.append(h2h)
        # hidden to output
        h2o = self.build_output_layer()
        self.layers.append(h2o)

    # initialize feature for each node
    def create_features(self):
        features = torch.arange(self.num_nodes)
        return features

    def build_input_layer(self):
        return RGCNLayer(
            self.num_nodes,
            self.h_dim,
            self.num_rels,
            self.num_bases,
            activation=F.relu,
            is_input_layer=True,
        )

    def build_hidden_layer(self):
        return RGCNLayer(
            self.h_dim,
            self.h_dim,
            self.num_rels,
            self.num_bases,
            activation=F.relu,
        )

    def build_output_layer(self):
        return RGCNLayer(
            self.h_dim,
            self.out_dim,
            self.num_rels,
            self.num_bases,
            activation=partial(F.softmax, dim=1),
        )

    def forward(self, g):
        if self.features is not None:
            g.ndata["id"] = self.features
        for layer in self.layers:
            layer(g)
        return g.ndata.pop("h")
    
    
# Fraud Amazon Dataset
from dgl.data import FraudAmazonDataset
amazon = FraudAmazonDataset()
g1 = amazon[0]
num_classes = amazon.num_classes
feat = g1.ndata['feature']
label = g1.ndata['label']

train_mask = g1.nodes['user'].data['train_mask']
test_mask = g1.nodes['user'].data['test_mask']
train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()
test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze()
labels = g1.nodes['user'].data.pop("label")
num_rels = len(g1.canonical_etypes)
num_classes = amazon.num_classes

# normalization factor
for cetype in g1.canonical_etypes:
    g1.edges[cetype].data["norm"] = dgl.norm_by_dst(g1, cetype).unsqueeze(1)
    
    
# configurations
n_hidden = 16  # number of hidden units
n_bases = -1  # use number of relations as number of bases
n_hidden_layers = 0  # use 1 input layer, 1 output layer, no hidden layer
n_epochs = 100  # epochs to train
lr = 0.01  # learning rate
l2norm = 0  # L2 norm coefficient

# create graph
g = dgl.to_homogeneous(g1, edata=["norm"])
node_ids = torch.arange(g.num_nodes())
#target_idx = node_ids[g1.ndata[dgl.NTYPE] == category_id]

# create model
model = Model(
    g.num_nodes(),
    n_hidden,
    num_classes,
    num_rels,
    num_bases=n_bases,
    num_hidden_layers=n_hidden_layers,
)


# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2norm)

print("start training...")
model.train()
for epoch in range(n_epochs):
    optimizer.zero_grad()
    logits = model.forward(g)
    #logits = logits[target_idx]
    loss = F.cross_entropy(logits[train_idx], labels[train_idx])
    loss.backward()

    optimizer.step()

    train_acc = torch.sum(logits[train_idx].argmax(dim=1) == labels[train_idx])
    train_acc = train_acc.item() / len(train_idx)
    val_loss = F.cross_entropy(logits[test_idx], labels[test_idx])
    val_acc = torch.sum(logits[test_idx].argmax(dim=1) == labels[test_idx])
    val_acc = val_acc.item() / len(test_idx)
    print(
        "Epoch {:05d} | ".format(epoch)
        + "Train Accuracy: {:.4f} | Train Loss: {:.4f} | ".format(
            train_acc, loss.item()
        )
        + "Validation Accuracy: {:.4f} | Validation loss: {:.4f}".format(
            val_acc, val_loss.item()
        )
    )
    


# After training, when evaluating on test data:
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    logits = model.forward(g)
    probabilities = F.softmax(logits[test_idx], dim=1)  # Convert logits to probabilities
    # Assuming your positive class is 1 (adjust accordingly if it's different)
    positive_probabilities = probabilities[:, 1]  # Get probabilities for the positive class
    # Calculate ROC-AUC
    roc_auc = roc_auc_score(labels[test_idx].cpu(), positive_probabilities.cpu())
    print(f"ROC-AUC Score: {roc_auc}")
    
    
    

# Fraud Yelp Dataset
from dgl.data import FraudYelpDataset
yelp = FraudYelpDataset()
g2 = yelp[0]
num_classes = yelp.num_classes
feat = g2.ndata['feature']
label = g2.ndata['label']
train_mask = g2.ndata['train_mask']
test_mask = g2.ndata['test_mask']


train_mask = g2.nodes['review'].data['train_mask']
test_mask = g2.nodes['review'].data['test_mask']
train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()
test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze()
labels = g2.nodes['review'].data.pop("label")
num_rels = len(g2.canonical_etypes)
num_classes = amazon.num_classes

# normalization factor
for cetype in g2.canonical_etypes:
    g2.edges[cetype].data["norm"] = dgl.norm_by_dst(g2, cetype).unsqueeze(1)
    
    
# configurations
n_hidden = 16  # number of hidden units
n_bases = -1  # use number of relations as number of bases
n_hidden_layers = 0  # use 1 input layer, 1 output layer, no hidden layer
n_epochs = 100  # epochs to train
lr = 0.01  # learning rate
l2norm = 0  # L2 norm coefficient

# create graph
g = dgl.to_homogeneous(g2, edata=["norm"])
node_ids = torch.arange(g.num_nodes())
#target_idx = node_ids[g2.ndata[dgl.NTYPE] == category_id]

# create model
model = Model(
    g.num_nodes(),
    n_hidden,
    num_classes,
    num_rels,
    num_bases=n_bases,
    num_hidden_layers=n_hidden_layers,
)


# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2norm)

print("start training...")
model.train()
for epoch in range(n_epochs):
    optimizer.zero_grad()
    logits = model.forward(g)
    #logits = logits[target_idx]
    loss = F.cross_entropy(logits[train_idx], labels[train_idx])
    loss.backward()

    optimizer.step()

    train_acc = torch.sum(logits[train_idx].argmax(dim=1) == labels[train_idx])
    train_acc = train_acc.item() / len(train_idx)
    val_loss = F.cross_entropy(logits[test_idx], labels[test_idx])
    val_acc = torch.sum(logits[test_idx].argmax(dim=1) == labels[test_idx])
    val_acc = val_acc.item() / len(test_idx)
    print(
        "Epoch {:05d} | ".format(epoch)
        + "Train Accuracy: {:.4f} | Train Loss: {:.4f} | ".format(
            train_acc, loss.item()
        )
        + "Validation Accuracy: {:.4f} | Validation loss: {:.4f}".format(
            val_acc, val_loss.item()
        )
    )
    
    
# After training, when evaluating on test data:
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    logits = model.forward(g)
    probabilities = F.softmax(logits[test_idx], dim=1)  # Convert logits to probabilities
    # Assuming your positive class is 1 (adjust accordingly if it's different)
    positive_probabilities = probabilities[:, 1]  # Get probabilities for the positive class
    # Calculate ROC-AUC
    roc_auc = roc_auc_score(labels[test_idx].cpu(), positive_probabilities.cpu())
    print(f"ROC-AUC Score: {roc_auc}")
    

# Reddit Dataset
reddit = pygod_load_data('reddit')
g3 = dgl.graph((reddit.edge_index[0], reddit.edge_index[1]))
g3.ndata['feature'] = reddit.x
g3.ndata['label'] = reddit.y.type(torch.LongTensor)

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h
    
    
# Example values, adjust based on your dataset
in_feats = g3.ndata['feature'].shape[1]
h_feats = 16  # Example hidden feature size
num_classes = reddit.y.max().item() + 1  # Assuming reddit.y contains class labels starting from 0

model = GCN(in_feats, h_feats, int(num_classes))


import torch

# Randomly select 80% of the nodes for training
num_nodes = g3.number_of_nodes()
num_train = int(num_nodes * 0.8)  # For example, 80% of the nodes for training
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[torch.randperm(num_nodes)[:num_train]] = True

val_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask[torch.randperm(num_nodes)[num_train:]] = True

# Add the train mask to your graph
g3.ndata['train_mask'] = train_mask
g3.ndata['val_mask'] = val_mask


from sklearn.metrics import accuracy_score
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):  # Example number of epochs
    model.train()
    logits = model(g3, g3.ndata['feature'])
    train_loss = F.cross_entropy(logits[g3.ndata['train_mask']], g3.ndata['label'][g3.ndata['train_mask']])
    
    # Compute training accuracy
    _, train_preds = torch.max(logits[g3.ndata['train_mask']], 1)
    train_acc = accuracy_score(g3.ndata['label'][g3.ndata['train_mask']].cpu().numpy(), train_preds.cpu().numpy())

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    # Validation phase
    model.eval()
    with torch.no_grad():
        logits = model(g3, g3.ndata['feature'])
        val_loss = F.cross_entropy(logits[g3.ndata['val_mask']], g3.ndata['label'][g3.ndata['val_mask']])
        
        # Compute validation accuracy
        _, val_preds = torch.max(logits[g3.ndata['val_mask']], 1)
        val_acc = accuracy_score(g3.ndata['label'][g3.ndata['val_mask']].cpu().numpy(), val_preds.cpu().numpy())

    print(
        "Epoch {:05d} | ".format(epoch)
        + "Train Accuracy: {:.4f} | Train Loss: {:.4f} | ".format(
            train_acc, train_loss.item()
        )
        + "Validation Accuracy: {:.4f} | Validation loss: {:.4f}".format(
            val_acc, val_loss.item()
        )
    )
    
# After training, when evaluating on test data:
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    logits = model.forward(g)
    probabilities = F.softmax(logits[test_idx], dim=1)  # Convert logits to probabilities
    # Assuming your positive class is 1 (adjust accordingly if it's different)
    positive_probabilities = probabilities[:, 1]  # Get probabilities for the positive class
    # Calculate ROC-AUC
    roc_auc = roc_auc_score(labels[test_idx].cpu(), positive_probabilities.cpu())
    print(f"ROC-AUC Score: {roc_auc}")