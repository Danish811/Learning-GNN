# GNN Frameworks Guide

A comprehensive guide to the major GNN frameworks with installation, comparison, and usage examples.

## Framework Comparison

| Framework | Backend | Best For | Learning Curve |
|-----------|---------|----------|----------------|
| **PyTorch Geometric (PyG)** | PyTorch | Research, production | Medium |
| **Deep Graph Library (DGL)** | PyTorch/TensorFlow/MXNet | Flexibility | Medium |
| **Spektral** | TensorFlow/Keras | Keras users | Easy |
| **StellarGraph** | TensorFlow | Easy API | Easy |
| **GraphNets** | TensorFlow | DeepMind style | Hard |

## PyTorch Geometric (Recommended)

The most popular GNN framework with excellent documentation and active community.

### Installation

```bash
# Check your CUDA version first
nvidia-smi

# Install PyTorch (match your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric
pip install torch-geometric

# Optional: Install additional packages
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### Key Concepts

#### Data Object

```python
from torch_geometric.data import Data

# Create a simple graph
edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[1], [2], [3]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)
print(data)
# Data(x=[3, 1], edge_index=[2, 4])

# Access attributes
print(data.num_nodes)  # 3
print(data.num_edges)  # 4
print(data.is_directed())  # False
```

#### Built-in Datasets

```python
from torch_geometric.datasets import Planetoid, TUDataset, QM9

# Citation networks
cora = Planetoid(root='./data', name='Cora')
citeseer = Planetoid(root='./data', name='CiteSeer')

# Graph classification
proteins = TUDataset(root='./data', name='PROTEINS')
mutag = TUDataset(root='./data', name='MUTAG')

# Molecular
qm9 = QM9(root='./data')
```

#### Common Layers

```python
from torch_geometric.nn import (
    GCNConv,      # Graph Convolutional
    GATConv,      # Graph Attention
    SAGEConv,     # GraphSAGE
    GINConv,      # Graph Isomorphism
    TransformerConv,  # Graph Transformer
    
    # Pooling
    global_mean_pool,
    global_max_pool,
    global_add_pool,
)
```

#### Complete Example

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

# Load data
dataset = Planetoid(root='./data', name='Cora')
data = dataset[0]

# Define model
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Train
model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

# Evaluate
model.eval()
pred = model(data).argmax(dim=1)
acc = (pred[data.test_mask] == data.y[data.test_mask]).sum() / data.test_mask.sum()
print(f'Accuracy: {acc:.4f}')
```

---

## Deep Graph Library (DGL)

Flexible framework supporting multiple backends.

### Installation

```bash
# For PyTorch backend
pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html
pip install dglgo -f https://data.dgl.ai/wheels-test/repo.html

# For CPU only
pip install dgl
```

### Key Concepts

```python
import dgl
import torch

# Create graph
src = torch.tensor([0, 1, 1, 2])
dst = torch.tensor([1, 0, 2, 1])
g = dgl.graph((src, dst))

# Add node features
g.ndata['feat'] = torch.randn(3, 5)

# Add edge features
g.edata['weight'] = torch.ones(4)
```

### Example Model

```python
import dgl.nn as dglnn

class GCN(torch.nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super().__init__()
        self.conv1 = dglnn.GraphConv(in_feats, hidden_size)
        self.conv2 = dglnn.GraphConv(hidden_size, num_classes)

    def forward(self, g, features):
        x = F.relu(self.conv1(g, features))
        x = self.conv2(g, x)
        return x
```

---

## Spektral (TensorFlow/Keras)

Best for Keras users who prefer high-level APIs.

### Installation

```bash
pip install spektral
```

### Example

```python
from spektral.layers import GCNConv
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout

class GCN(Model):
    def __init__(self, n_hidden, n_classes):
        super().__init__()
        self.conv1 = GCNConv(n_hidden, activation='relu')
        self.conv2 = GCNConv(n_classes, activation='softmax')
        self.dropout = Dropout(0.5)

    def call(self, inputs):
        x, a = inputs
        x = self.conv1([x, a])
        x = self.dropout(x)
        return self.conv2([x, a])
```

---

## Framework Selection Guide

```
Need PyTorch? ──Yes──▶ PyTorch Geometric (most popular)
    │ No                        │
    │                           └── OR DGL (more flexible)
    ▼
Need TensorFlow? ──Yes──▶ Spektral (Keras-like)
    │ No                         │
    │                            └── OR DGL (TF backend)
    ▼
Need multi-backend? ──Yes──▶ DGL
    │ No
    ▼
Research with latest methods? ──▶ PyTorch Geometric
```

---

## Useful Resources

### PyTorch Geometric
- [Documentation](https://pytorch-geometric.readthedocs.io/)
- [GitHub](https://github.com/pyg-team/pytorch_geometric)
- [Tutorial Notebooks](https://pytorch-geometric.readthedocs.io/en/latest/notes/colabs.html)

### DGL
- [Documentation](https://docs.dgl.ai/)
- [GitHub](https://github.com/dmlc/dgl)
- [Tutorials](https://docs.dgl.ai/tutorials/blitz/index.html)

### Spektral
- [Documentation](https://graphneural.network/)
- [GitHub](https://github.com/danielegrattarola/spektral)
