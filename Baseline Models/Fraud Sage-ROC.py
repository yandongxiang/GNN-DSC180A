import os

os.environ["DGLBACKEND"] = "pytorch"
from functools import partial

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn import GINConv
from dgl.nn import SAGEConv
import numpy as np
from pygod.utils import load_data as pygod_load_data


# # Amazon


from dgl.data import FraudAmazonDataset
amazon = FraudAmazonDataset()
g1 = amazon[0]

# If you're using a GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
g1 = g1.to(device)



class GraphSAGEModel(torch.nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConv(in_feats=in_feats, out_feats=h_feats, aggregator_type='mean')
        self.conv2 = SAGEConv(in_feats=h_feats, out_feats=num_classes, aggregator_type='mean')

    def forward(self, g, features):
        h = self.conv1(g, features)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h


in_feats = g1.ndata['feature'].shape[1]
h_feats = 16  # Example size of hidden layers
num_classes = len(torch.unique(g1.ndata['label']))  # Assuming node labels are used for classification

model = GraphSAGEModel(in_feats, h_feats, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)



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



import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

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



# # Yelp
from dgl.data import FraudYelpDataset
yelp = FraudYelpDataset()
g2 = yelp[0]

# If you're using a GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
g2 = g2.to(device)



class GraphSAGEModel(torch.nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConv(in_feats=in_feats, out_feats=h_feats, aggregator_type='mean')
        self.conv2 = SAGEConv(in_feats=h_feats, out_feats=num_classes, aggregator_type='mean')

    def forward(self, g, features):
        h = self.conv1(g, features)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h



in_feats = g2.ndata['feature'].shape[1]
h_feats = 16  # Example size of hidden layers
num_classes = len(torch.unique(g2.ndata['label']))  # Assuming node labels are used for classification

model = GraphSAGEModel(in_feats, h_feats, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


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



import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

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


# # Reddit
reddit = pygod_load_data('reddit')
g3 = dgl.graph((reddit.edge_index[0], reddit.edge_index[1]))
g3.ndata['feature'] = reddit.x
g3.ndata['label'] = reddit.y.type(torch.LongTensor)



import torch
import torch.nn.functional as F
from dgl.nn import SAGEConv

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats=in_feats, out_feats=h_feats, aggregator_type='mean')
        self.conv2 = SAGEConv(in_feats=h_feats, out_feats=num_classes, aggregator_type='mean')

    def forward(self, g, features):
        h = self.conv1(g, features)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h



from sklearn.model_selection import train_test_split

# Assuming reddit.y contains binary labels for a binary classification task
train_mask, test_mask = train_test_split(torch.arange(g3.number_of_nodes()), test_size=0.1, random_state=42)

# Optional: Convert masks to boolean tensors for PyTorch
train_mask = torch.isin(torch.arange(g3.number_of_nodes()), train_mask)
test_mask = torch.isin(torch.arange(g3.number_of_nodes()), test_mask)



# Initialize the model
num_classes = len(torch.unique(g3.ndata['label']))
model = GraphSAGE(g3.ndata['feature'].shape[1], 128, num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_func = torch.nn.CrossEntropyLoss()

for epoch in range(100):
    model.train()
    logits = model(g3, g3.ndata['feature'])
    loss = loss_func(logits[train_mask], g3.ndata['label'][train_mask])
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f'Epoch {epoch}, Loss: {loss.item()}')



from sklearn.metrics import roc_auc_score

model.eval()
with torch.no_grad():
    test_logits = model(g3, g3.ndata['feature'])[test_mask]
    test_probs = torch.softmax(test_logits, dim=1)[:, 1]  # Get probability for the positive class
    test_labels = g3.ndata['label'][test_mask]
    
# Calculate ROC-AUC
roc_auc = roc_auc_score(test_labels.cpu(), test_probs.cpu())
print(f'ROC-AUC score: {roc_auc}')
