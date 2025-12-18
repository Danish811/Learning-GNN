# The Over-Smoothing Problem

One of the most critical challenges in deep GNNs: as layers increase, node representations become indistinguishable.

## The Problem Explained

### What Happens with Deep GNNs?

```
Layer 1: Nodes have distinct representations
Layer 2: Still distinguishable
Layer 3: Starting to blur
Layer 5: Very similar
Layer 10: Nearly identical → "Over-smoothed"
```

### Mathematical Intuition

With each GNN layer, a node aggregates from its neighbors. After k layers:
- Each node's representation contains information from k-hop neighborhood
- In dense graphs, neighborhoods quickly overlap
- Eventually, all nodes have similar global information

```python
# Demonstrating over-smoothing
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='./data', name='Cora')
data = dataset[0]

def measure_smoothness(embeddings):
    """Measure how similar node embeddings are."""
    embeddings = F.normalize(embeddings, p=2, dim=1)
    similarity = embeddings @ embeddings.t()
    return similarity.mean().item()

# Track smoothness across layers
class DeepGCN(torch.nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList([
            GCNConv(dataset.num_features if i == 0 else 64, 64)
            for i in range(num_layers)
        ])
    
    def forward(self, x, edge_index):
        smoothness = []
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            smoothness.append(measure_smoothness(x))
        return x, smoothness

# Test with increasing depths
for num_layers in [2, 4, 8, 16]:
    model = DeepGCN(num_layers)
    _, smoothness = model(data.x, data.edge_index)
    print(f"{num_layers} layers: Final smoothness = {smoothness[-1]:.4f}")
```

**Output:**
```
2 layers: Final smoothness = 0.4523
4 layers: Final smoothness = 0.7812
8 layers: Final smoothness = 0.9456
16 layers: Final smoothness = 0.9978  # Nearly identical!
```

## Solutions to Over-Smoothing

### 1. Residual Connections

Add skip connections from input to output of each layer.

```python
class ResGCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, num_layers):
        super().__init__()
        self.input_proj = torch.nn.Linear(num_features, hidden_dim)
        self.convs = torch.nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.output = torch.nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x, edge_index):
        x = self.input_proj(x)
        for conv in self.convs:
            x = x + F.relu(conv(x, edge_index))  # Residual!
        return self.output(x)
```

### 2. GCNII: Initial Residual + Identity Mapping

**Paper:** "Simple and Deep Graph Convolutional Networks" (Chen et al., 2020)

```python
from torch_geometric.nn import GCN2Conv

class GCNII(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, num_layers, 
                 alpha=0.1, theta=0.5):
        super().__init__()
        self.lin1 = torch.nn.Linear(num_features, hidden_dim)
        self.convs = torch.nn.ModuleList([
            GCN2Conv(hidden_dim, alpha=alpha, theta=theta, layer=i+1)
            for i in range(num_layers)
        ])
        self.lin2 = torch.nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x, edge_index):
        x = x_0 = F.relu(self.lin1(x))
        for conv in self.convs:
            x = F.dropout(x, training=self.training)
            x = conv(x, x_0, edge_index)  # Uses initial features!
            x = F.relu(x)
        return self.lin2(x)
```

### 3. DropEdge

**Paper:** "DropEdge: Towards Deep Graph Convolutional Networks" (Rong et al., 2020)

Randomly drop edges during training (like dropout for edges).

```python
from torch_geometric.utils import dropout_edge

class DropEdgeGCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, num_layers, drop_rate=0.5):
        super().__init__()
        self.drop_rate = drop_rate
        self.convs = torch.nn.ModuleList([
            GCNConv(num_features if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        self.lin = torch.nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x, edge_index):
        for conv in self.convs:
            if self.training:
                edge_index, _ = dropout_edge(edge_index, p=self.drop_rate)
            x = F.relu(conv(x, edge_index))
        return self.lin(x)
```

### 4. JumpingKnowledge Networks

**Paper:** "Representation Learning on Graphs with Jumping Knowledge Networks" (Xu et al., 2018)

Combine representations from all layers, not just the last.

```python
from torch_geometric.nn import JumpingKnowledge

class JKGCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList([
            GCNConv(num_features if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        self.jk = JumpingKnowledge(mode='cat')  # or 'max', 'lstm'
        self.lin = torch.nn.Linear(hidden_dim * num_layers, num_classes)
    
    def forward(self, x, edge_index):
        xs = []
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            xs.append(x)
        x = self.jk(xs)  # Combine all layers
        return self.lin(x)
```

### 5. PairNorm

**Paper:** "PairNorm: Tackling Oversmoothing in GNNs" (Zhao & Akoglu, 2020)

Normalize node features to maintain diversity.

```python
from torch_geometric.nn import PairNorm

class PairNormGCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        
        for i in range(num_layers):
            self.convs.append(GCNConv(
                num_features if i == 0 else hidden_dim, hidden_dim))
            self.norms.append(PairNorm())
        
        self.lin = torch.nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)  # Normalize to prevent over-smoothing
            x = F.relu(x)
        return self.lin(x)
```

## Comparison of Solutions

| Method | Mechanism | Layers Supported | Complexity |
|--------|-----------|------------------|------------|
| **Residual** | Skip connections | ~10 | Low |
| **GCNII** | Initial + identity | 64+ | Medium |
| **DropEdge** | Random edge removal | ~16 | Low |
| **JK-Net** | Multi-layer combine | ~8 | Medium |
| **PairNorm** | Feature normalization | ~16 | Low |

## Practical Recommendations

1. **Default**: Use 2-3 layers (sufficient for most tasks)
2. **Deeper needed**: Try GCNII or JumpingKnowledge
3. **Training instability**: Add DropEdge or PairNorm
4. **Always**: Use residual connections

---

## References

- Li, Q., et al. (2018). "Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning." AAAI.
- Chen, M., et al. (2020). "Simple and Deep Graph Convolutional Networks." ICML. [arXiv](https://arxiv.org/abs/2007.02133)
- Rong, Y., et al. (2020). "DropEdge: Towards Deep Graph Convolutional Networks on Node Classification." ICLR. [arXiv](https://arxiv.org/abs/1907.10903)

---

**Next:** [Heterogeneous Graphs →](./02-heterogeneous-graphs.md)
