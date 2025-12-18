# Scalability Techniques for Large Graphs

Training GNNs on graphs with millions of nodes requires special techniques.

## The Scalability Challenge

| Graph Size | Challenge | Solution |
|------------|-----------|----------|
| <100K nodes | Full-batch OK | Standard training |
| 100K-1M nodes | Memory limited | Mini-batching |
| >1M nodes | Memory + compute | Sampling + distributed |

## Mini-Batch Training with Neighbor Sampling

### NeighborLoader

```python
from torch_geometric.loader import NeighborLoader

# Sample fixed number of neighbors per layer
train_loader = NeighborLoader(
    data,
    num_neighbors=[25, 10],  # Layer 1: 25, Layer 2: 10
    batch_size=1024,
    input_nodes=data.train_mask,
    shuffle=True
)

for batch in train_loader:
    # batch.batch_size: number of target nodes
    # batch.n_id: node IDs in the batch
    # batch.edge_index: edges for the batch
    
    out = model(batch.x, batch.edge_index)
    # Only compute loss on target nodes
    loss = F.cross_entropy(out[:batch.batch_size], batch.y[:batch.batch_size])
```

### Understanding the Batch Structure

```
Original Graph:
    0 --- 1 --- 2 --- 3 --- 4
    
Batch for node 2 (num_neighbors=[2, 2]):
    
    Layer 0 (target):     2
                         /|\
    Layer 1 (1-hop):    1   3
                       /|   |\
    Layer 2 (2-hop):  0  2  2  4
    
batch.n_id = [2, 1, 3, 0, 4]  (node 2 first, then neighbors)
batch.batch_size = 1
```

## GraphSAINT: Subgraph Sampling

**Paper:** "GraphSAINT: Graph Sampling Based Inductive Learning Method" (2020)

Sample subgraphs instead of neighborhoods.

```python
from torch_geometric.loader import GraphSAINTRandomWalkSampler

# Random walk based sampling
loader = GraphSAINTRandomWalkSampler(
    data,
    batch_size=6000,
    walk_length=2,
    num_steps=5,
    sample_coverage=100
)

for batch in loader:
    # batch is a subgraph (like a small complete graph)
    out = model(batch.x, batch.edge_index)
    loss = F.cross_entropy(out, batch.y)
```

### GraphSAINT Variants

```python
from torch_geometric.loader import (
    GraphSAINTRandomWalkSampler,  # Random walk
    GraphSAINTEdgeSampler,        # Edge sampling
    GraphSAINTNodeSampler         # Node sampling
)
```

## Cluster-GCN

**Paper:** "Cluster-GCN: An Efficient Algorithm for Training Deep and Large GCNs" (2019)

Partition graph into clusters, train on clusters.

```python
from torch_geometric.loader import ClusterData, ClusterLoader

# Partition graph into 1500 clusters
cluster_data = ClusterData(data, num_parts=1500)

# Load batches of clusters
loader = ClusterLoader(cluster_data, batch_size=20, shuffle=True)

for batch in loader:
    out = model(batch.x, batch.edge_index)
    loss = F.cross_entropy(out, batch.y)
```

## Comparison of Sampling Methods

| Method | Memory | Variance | Best For |
|--------|--------|----------|----------|
| **NeighborLoader** | O(batch × ∏k_i) | Low | General |
| **GraphSAINT** | O(subgraph) | Medium | Medium graphs |
| **Cluster-GCN** | O(cluster) | Higher | Very large graphs |

## Implementation Tips

### 1. Pin Memory for Faster Loading

```python
loader = NeighborLoader(
    data,
    num_neighbors=[25, 10],
    batch_size=1024,
    num_workers=4,
    pin_memory=True  # Faster GPU transfer
)
```

### 2. Use Mixed Precision

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in loader:
    optimizer.zero_grad()
    
    with autocast():  # FP16 forward pass
        out = model(batch.x, batch.edge_index)
        loss = criterion(out[:batch.batch_size], batch.y[:batch.batch_size])
    
    scaler.scale(loss).backward()  # Scaled backward
    scaler.step(optimizer)
    scaler.update()
```

### 3. Gradient Accumulation

```python
accumulation_steps = 4
optimizer.zero_grad()

for i, batch in enumerate(loader):
    out = model(batch.x, batch.edge_index)
    loss = criterion(out[:batch.batch_size], batch.y[:batch.batch_size])
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 4. Cache Sampling (for fixed samples)

```python
# Pre-compute samples for deterministic training
from torch_geometric.loader import NeighborLoader

loader = NeighborLoader(data, ...)
cached_batches = [batch for batch in loader]

for epoch in range(epochs):
    for batch in cached_batches:  # Reuse same samples
        train(batch)
```

## Large-Scale Example: ogbn-products

```python
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.loader import NeighborLoader

# Load ogbn-products (2.4M nodes, 61M edges)
dataset = PygNodePropPredDataset(name='ogbn-products')
data = dataset[0]
split_idx = dataset.get_idx_split()

# Create loaders with sampling
train_loader = NeighborLoader(
    data,
    input_nodes=split_idx['train'],
    num_neighbors=[15, 10, 5],  # 3 layers
    batch_size=1024,
    shuffle=True,
    num_workers=8,
)

# 3-layer GraphSAGE
class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            out_ch = out_channels if i == num_layers - 1 else hidden_channels
            self.convs.append(SAGEConv(in_ch, out_ch))
    
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x

model = SAGE(100, 256, 47).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

for epoch in range(10):
    model.train()
    for batch in train_loader:
        batch = batch.cuda()
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)[:batch.batch_size]
        loss = F.cross_entropy(out, batch.y[:batch.batch_size].squeeze())
        loss.backward()
        optimizer.step()
```

---

## References

- Hamilton, W., et al. (2017). "Inductive Representation Learning on Large Graphs." NeurIPS.
- Zeng, H., et al. (2020). "GraphSAINT: Graph Sampling Based Inductive Learning Method." ICLR. [arXiv](https://arxiv.org/abs/1907.04931)
- Chiang, W., et al. (2019). "Cluster-GCN: An Efficient Algorithm for Training Deep and Large GCNs." KDD. [arXiv](https://arxiv.org/abs/1905.07953)

---

**Next:** [Mini-Batching Strategies →](./03-mini-batching.md)
