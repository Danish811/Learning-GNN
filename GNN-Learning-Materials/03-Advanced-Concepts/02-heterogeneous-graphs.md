# Heterogeneous Graph Neural Networks

Real-world graphs often have **multiple types** of nodes and edges. Standard GNNs treat everything uniformly—heterogeneous GNNs don't.

## What are Heterogeneous Graphs?

### Homogeneous vs Heterogeneous

| Graph Type | Node Types | Edge Types | Example |
|------------|------------|------------|---------|
| **Homogeneous** | 1 | 1 | Citation network |
| **Heterogeneous** | Multiple | Multiple | Knowledge graphs, social networks |

### Examples

**1. Academic Network (DBLP)**
- Nodes: Author, Paper, Venue, Term
- Edges: writes, published_in, contains

**2. Movie Database (IMDB)**
- Nodes: Movie, Actor, Director
- Edges: acted_in, directed

**3. E-commerce**
- Nodes: User, Product, Category, Brand
- Edges: purchased, belongs_to, manufactured_by

## Representing Heterogeneous Graphs in PyTorch Geometric

```python
from torch_geometric.data import HeteroData

# Create heterogeneous graph
data = HeteroData()

# Node features for different types
data['user'].x = torch.randn(100, 32)     # 100 users, 32 features
data['movie'].x = torch.randn(50, 64)     # 50 movies, 64 features
data['actor'].x = torch.randn(30, 16)     # 30 actors, 16 features

# Edge indices for different relations
data['user', 'rates', 'movie'].edge_index = torch.randint(0, 50, (2, 500))
data['actor', 'acts_in', 'movie'].edge_index = torch.randint(0, 30, (2, 100))

# Edge attributes (optional)
data['user', 'rates', 'movie'].edge_attr = torch.randn(500, 1)  # Ratings

print(data)
# HeteroData(
#   user={ x=[100, 32] },
#   movie={ x=[50, 64] },
#   actor={ x=[30, 16] },
#   (user, rates, movie)={ edge_index=[2, 500], edge_attr=[500, 1] },
#   (actor, acts_in, movie)={ edge_index=[2, 100] }
# )
```

## Metapaths

A **metapath** is a sequence of node types connected by edge types.

### Examples in IMDB
- **MAM**: Movie → Actor → Movie (movies sharing actors)
- **MDM**: Movie → Director → Movie (movies by same director)
- **MADAM**: Movie → Actor → Director → Actor → Movie

### Computing Metapath-based Neighbors

```python
import torch
from torch_geometric.utils import coalesce

def compute_metapath_adj(data, metapath):
    """Compute adjacency for a metapath."""
    # Start with identity
    adj = None
    
    for i, (src, rel, dst) in enumerate(metapath):
        edge_index = data[src, rel, dst].edge_index
        
        if adj is None:
            adj = edge_index
        else:
            # Matrix multiplication of adjacencies
            # A @ B means: for each edge (i,j) in A and (j,k) in B, add edge (i,k)
            adj = torch.stack([
                adj[0].repeat_interleave(edge_index.size(1)),
                edge_index[1].repeat(adj.size(1))
            ])
            # Filter to valid connections
            mask = adj[0] < data[dst].num_nodes
            adj = adj[:, mask]
            adj = coalesce(adj)
    
    return adj
```

## Heterogeneous GNN Architectures

### 1. HAN: Heterogeneous Graph Attention Network

**Paper:** "Heterogeneous Graph Attention Network" (Wang et al., 2019)

Uses hierarchical attention:
1. **Node-level attention**: Attention over neighbors in each metapath
2. **Semantic-level attention**: Combines different metapaths

```python
from torch_geometric.nn import HANConv

class HAN(torch.nn.Module):
    def __init__(self, metadata, in_channels, hidden_channels, out_channels, num_heads=8):
        super().__init__()
        self.conv1 = HANConv(in_channels, hidden_channels, metadata, heads=num_heads)
        self.conv2 = HANConv(hidden_channels, out_channels, metadata, heads=1)
    
    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.elu(x) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict
```

### 2. R-GCN: Relational Graph Convolutional Network

**Paper:** "Modeling Relational Data with Graph Convolutional Networks" (Schlichtkrull et al., 2018)

Separate weight matrices for each relation type.

```python
from torch_geometric.nn import RGCNConv

class RGCN(torch.nn.Module):
    def __init__(self, num_nodes, num_relations, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = RGCNConv(num_nodes, hidden_channels, num_relations)
        self.conv2 = RGCNConv(hidden_channels, out_channels, num_relations)
    
    def forward(self, x, edge_index, edge_type):
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = self.conv2(x, edge_index, edge_type)
        return x
```

### 3. HGT: Heterogeneous Graph Transformer

**Paper:** "Heterogeneous Graph Transformer" (Hu et al., 2020)

Applies transformer attention with type-specific projections.

```python
from torch_geometric.nn import HGTConv

class HGT(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels, num_heads, num_layers):
        super().__init__()
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin_dict[node_type] = torch.nn.Linear(hidden_channels, hidden_channels)
        
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata, num_heads)
            self.convs.append(conv)
        
        self.lin = torch.nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x_dict, edge_index_dict):
        x_dict = {key: self.lin_dict[key](x) for key, x in x_dict.items()}
        
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
        
        return {key: self.lin(x) for key, x in x_dict.items()}
```

### 4. Simple Heterogeneous GNN (to_hetero)

PyG can automatically convert homogeneous GNNs to heterogeneous!

```python
from torch_geometric.nn import GCNConv, to_hetero

# Define a homogeneous GNN
class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(-1, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

# Convert to heterogeneous
model = GNN(64, dataset.num_classes)
model = to_hetero(model, data.metadata(), aggr='sum')
```

## Complete Example: Movie Recommendation

```python
import torch
import torch.nn.functional as F
from torch_geometric.datasets import IMDB
from torch_geometric.nn import HANConv

# Load IMDB dataset
dataset = IMDB(root='./data/IMDB')
data = dataset[0]

print(f"Node types: {data.node_types}")  # ['movie', 'director', 'actor']
print(f"Edge types: {data.edge_types}")

class MovieHAN(torch.nn.Module):
    def __init__(self, metadata, hidden_channels=64, out_channels=3):
        super().__init__()
        # Project all node features to same dimension
        self.movie_lin = torch.nn.Linear(data['movie'].x.size(1), hidden_channels)
        self.director_lin = torch.nn.Linear(data['director'].x.size(1), hidden_channels)
        self.actor_lin = torch.nn.Linear(data['actor'].x.size(1), hidden_channels)
        
        self.conv1 = HANConv(hidden_channels, hidden_channels, metadata, heads=8)
        self.conv2 = HANConv(hidden_channels, out_channels, metadata, heads=1)
    
    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            'movie': self.movie_lin(x_dict['movie']),
            'director': self.director_lin(x_dict['director']),
            'actor': self.actor_lin(x_dict['actor']),
        }
        
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.elu(v) for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        
        return x_dict['movie']  # Classify movies

# Training
model = MovieHAN(data.metadata())
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    loss = F.cross_entropy(out[data['movie'].train_mask], 
                           data['movie'].y[data['movie'].train_mask])
    loss.backward()
    optimizer.step()

# Evaluate
model.eval()
pred = model(data.x_dict, data.edge_index_dict).argmax(dim=1)
acc = (pred[data['movie'].test_mask] == data['movie'].y[data['movie'].test_mask]).float().mean()
print(f"Test Accuracy: {acc:.4f}")
```

## Comparison of Heterogeneous GNNs

| Model | Key Feature | Best For |
|-------|-------------|----------|
| **HAN** | Hierarchical attention | Metapath-based tasks |
| **R-GCN** | Relation-specific weights | Knowledge graphs |
| **HGT** | Transformer architecture | Large-scale hetero graphs |
| **to_hetero** | Easy conversion | Quick prototyping |

---

## References

- Wang, X., et al. (2019). "Heterogeneous Graph Attention Network." WWW. [arXiv](https://arxiv.org/abs/1903.07293)
- Schlichtkrull, M., et al. (2018). "Modeling Relational Data with Graph Convolutional Networks." ESWC. [arXiv](https://arxiv.org/abs/1703.06103)
- Hu, Z., et al. (2020). "Heterogeneous Graph Transformer." WWW. [arXiv](https://arxiv.org/abs/2003.01332)

---

**Next:** [Temporal & Dynamic GNNs →](./03-temporal-gnns.md)
