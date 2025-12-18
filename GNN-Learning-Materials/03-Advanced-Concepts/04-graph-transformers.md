# Graph Transformers

Applying the powerful Transformer architecture to graphs, combining attention mechanisms with graph structure.

## Why Graph Transformers?

| GNN Limitation | Transformer Solution |
|----------------|---------------------|
| Limited receptive field | Global attention |
| Over-smoothing | Skip connections |
| Fixed aggregation | Learned attention |
| Sequential message passing | Parallel computation |

## Evolution of Graph Attention

```
GAT (2018) → Graph Transformer (2020) → GPS (2022) → Uni-Mol (2023)
   ↓              ↓                        ↓
Local           Global                 Hybrid
attention       attention              local + global
```

## Key Difference: Local vs Global Attention

### GAT: Local Attention
Attention only over immediate neighbors.

```python
# GAT: Only attends to neighbors
for neighbor in neighbors(node):
    attention_weight = compute_attention(node, neighbor)
```

### Graph Transformer: Global Attention
Attention over ALL nodes (optionally constrained by structure).

```python
# Graph Transformer: Attends to all nodes
for other_node in all_nodes:
    attention_weight = compute_attention(node, other_node)
```

## Graph Transformer Architectures

### 1. Vanilla Graph Transformer

```python
import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv

class GraphTransformer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 num_heads=8, num_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        self.layers = nn.ModuleList([
            TransformerConv(hidden_channels, hidden_channels // num_heads, 
                           heads=num_heads, dropout=0.1)
            for _ in range(num_layers)
        ])
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels) for _ in range(num_layers)
        ])
        
        self.output = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = self.input_proj(x)
        
        for layer, norm in zip(self.layers, self.norms):
            x_new = layer(x, edge_index)
            x = norm(x + x_new)  # Residual + LayerNorm
            x = F.gelu(x)
        
        return self.output(x)
```

### 2. GPS: General, Powerful, Scalable

**Paper:** "Recipe for a General, Powerful, Scalable Graph Transformer" (Rampášek et al., 2022)

Combines local message passing with global attention.

```python
from torch_geometric.nn import GPSConv, GINEConv

class GPS(nn.Module):
    """General Powerful Scalable Graph Transformer."""
    
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 num_layers=5, heads=4):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            # Local message passing (GIN)
            local_nn = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
            local_model = GINEConv(local_nn)
            
            # GPS layer combines local + global
            self.convs.append(GPSConv(
                channels=hidden_channels,
                conv=local_model,
                heads=heads,
                attn_dropout=0.1,
                ffn_dropout=0.1
            ))
        
        self.output = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x, edge_index, edge_attr, batch):
        x = self.input_proj(x)
        
        for conv in self.convs:
            x = conv(x, edge_index, batch, edge_attr=edge_attr)
        
        return self.output(x)
```

### 3. Graphormer

**Paper:** "Do Transformers Really Perform Bad for Graph Representation?" (Ying et al., 2021)

Uses special positional encodings for graphs.

```python
class GraphormerEncoder(nn.Module):
    """Graphormer-style encoder with graph-specific encodings."""
    
    def __init__(self, hidden_dim, num_heads, num_layers, max_nodes=512):
        super().__init__()
        
        # Centrality encoding (based on node degree)
        self.centrality_encoder = nn.Embedding(max_nodes, hidden_dim)
        
        # Spatial encoding (based on shortest path distance)
        self.spatial_encoder = nn.Embedding(max_nodes, num_heads)
        
        # Edge encoding
        self.edge_encoder = nn.Embedding(max_nodes, num_heads)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden_dim, num_heads, 
                                       dim_feedforward=hidden_dim * 4)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, edge_index, degree, shortest_paths):
        # Add centrality encoding
        centrality_encoding = self.centrality_encoder(degree)
        x = x + centrality_encoding
        
        # Compute attention bias from spatial encoding
        attn_bias = self.spatial_encoder(shortest_paths)  # [N, N, heads]
        
        # Apply transformer with attention bias
        for layer in self.layers:
            x = layer(x)  # Note: real implementation adds attn_bias
        
        return x
```

## Positional Encodings for Graphs

Unlike sequences, graphs have no inherent ordering. We need special positional encodings.

### 1. Laplacian Eigenvector Encoding

```python
from torch_geometric.transforms import AddLaplacianEigenvectorPE

transform = AddLaplacianEigenvectorPE(k=8)  # Top 8 eigenvectors
data = transform(data)
# data.laplacian_eigenvector_pe: [num_nodes, 8]
```

### 2. Random Walk Encoding

```python
from torch_geometric.transforms import AddRandomWalkPE

transform = AddRandomWalkPE(walk_length=20)
data = transform(data)
# data.random_walk_pe: [num_nodes, 20]
```

### 3. Distance Encoding

```python
def compute_distance_encoding(edge_index, num_nodes, max_dist=5):
    """Shortest path distance encoding."""
    import networkx as nx
    from torch_geometric.utils import to_networkx
    
    G = to_networkx(Data(edge_index=edge_index, num_nodes=num_nodes))
    dist_matrix = torch.zeros(num_nodes, num_nodes)
    
    for i, distances in nx.shortest_path_length(G):
        for j, d in distances.items():
            dist_matrix[i, j] = min(d, max_dist)
    
    return dist_matrix
```

## Complete Example: GPS for Graph Classification

```python
import torch
import torch.nn.functional as F
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import AddRandomWalkPE
from torch_geometric.nn import GPSConv, GINEConv, global_add_pool

# Load ZINC with positional encoding
transform = AddRandomWalkPE(walk_length=20)
train_dataset = ZINC(root='./data', subset=True, split='train', pre_transform=transform)
test_dataset = ZINC(root='./data', subset=True, split='test', pre_transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

class GPSModel(torch.nn.Module):
    def __init__(self, hidden_channels=64, num_layers=5, heads=4):
        super().__init__()
        
        # Input embedding (node features + PE)
        self.node_embed = torch.nn.Embedding(28, hidden_channels - 20)  # ZINC atoms
        self.edge_embed = torch.nn.Embedding(4, hidden_channels)  # ZINC bonds
        
        # GPS layers
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            local_nn = torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels)
            )
            self.convs.append(GPSConv(
                hidden_channels, GINEConv(local_nn),
                heads=heads, attn_dropout=0.1
            ))
        
        # Output
        self.output = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 1)
        )
    
    def forward(self, data):
        # Combine node embedding with PE
        x = torch.cat([
            self.node_embed(data.x.squeeze()),
            data.random_walk_pe
        ], dim=-1)
        
        edge_attr = self.edge_embed(data.edge_attr.squeeze())
        
        for conv in self.convs:
            x = conv(x, data.edge_index, data.batch, edge_attr=edge_attr)
        
        x = global_add_pool(x, data.batch)
        return self.output(x)

# Training
model = GPSModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data).squeeze()
        loss = F.l1_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: MAE = {total_loss/len(train_dataset):.4f}")
```

## When to Use Graph Transformers

| Use Graph Transformers When | Stick with GNNs When |
|----------------------------|---------------------|
| Need global context | Local patterns suffice |
| Graph is small/medium | Very large graphs (scalability) |
| High-quality predictions needed | Speed is priority |
| Sufficient compute available | Limited resources |

---

## References

- Ying, C., et al. (2021). "Do Transformers Really Perform Bad for Graph Representation?" NeurIPS. [arXiv](https://arxiv.org/abs/2106.05234)
- Rampášek, L., et al. (2022). "Recipe for a General, Powerful, Scalable Graph Transformer." NeurIPS. [arXiv](https://arxiv.org/abs/2205.12454)
- Dwivedi, V. P., & Bresson, X. (2020). "A Generalization of Transformer Networks to Graphs." AAAI Workshop. [arXiv](https://arxiv.org/abs/2012.09699)

---

**Next Module:** [Training & Optimization →](../04-Training-Optimization/01-training-techniques.md)
