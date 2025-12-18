# Temporal and Dynamic Graph Neural Networks

Real-world graphs **evolve over time**. Temporal GNNs capture both spatial (graph structure) and temporal (time) dependencies.

## Types of Dynamic Graphs

### 1. Discrete-Time Dynamic Graphs (DTDG)
Graph snapshots at regular intervals: G₁, G₂, G₃, ...

```
t=1: A---B    t=2: A---B---C    t=3: A---B---C
     |             |                 |   |
     D             D                 D---E
```

### 2. Continuous-Time Dynamic Graphs (CTDG)
Events with exact timestamps: (u, v, t, features)

```
Events:
(A, B, 10:00, "message")
(B, C, 10:05, "like")
(A, D, 10:10, "follow")
...
```

## Approaches to Temporal GNNs

### Approach 1: Snapshot + RNN

Process each graph snapshot with a GNN, then model temporal evolution with RNN.

```python
class SnapshotGRU(torch.nn.Module):
    """GNN + GRU for discrete-time graphs."""
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_nodes):
        super().__init__()
        self.gnn = GCNConv(in_channels, hidden_channels)
        self.gru = torch.nn.GRU(hidden_channels, hidden_channels, batch_first=True)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)
        self.num_nodes = num_nodes
    
    def forward(self, snapshots):
        """
        snapshots: list of (x, edge_index) tuples for each time step
        """
        temporal_embeddings = []
        
        for x, edge_index in snapshots:
            h = F.relu(self.gnn(x, edge_index))  # [N, hidden]
            temporal_embeddings.append(h)
        
        # Stack: [T, N, hidden] -> [N, T, hidden]
        h_seq = torch.stack(temporal_embeddings, dim=0).permute(1, 0, 2)
        
        # GRU over time for each node
        out, _ = self.gru(h_seq)  # [N, T, hidden]
        
        # Use last time step
        return self.lin(out[:, -1, :])
```

### Approach 2: EvolveGCN

**Paper:** "EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs" (2020)

Update GNN weights over time using RNN.

```python
from torch_geometric.nn import GCNConv

class EvolveGCN(torch.nn.Module):
    """GCN weights evolve over time."""
    
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        
        # RNN to evolve GCN weights
        self.weight_gru = torch.nn.GRU(
            in_channels * hidden_channels,
            in_channels * hidden_channels,
            batch_first=True
        )
        
        self.lin = torch.nn.Linear(hidden_channels, out_channels)
    
    def forward(self, snapshots):
        batch_size = snapshots[0][0].size(0)
        h_weight = None
        
        all_outputs = []
        for x, edge_index in snapshots:
            # Get evolved weights
            if h_weight is None:
                weight = torch.randn(x.size(1), self.hidden_channels)
            else:
                weight = h_weight.view(x.size(1), self.hidden_channels)
            
            # Apply GCN with evolved weights
            h = F.relu(x @ weight)  # Simplified
            
            # Update weight RNN
            weight_flat = weight.flatten().unsqueeze(0).unsqueeze(0)
            _, h_weight = self.weight_gru(weight_flat)
            h_weight = h_weight.squeeze(0)
            
            all_outputs.append(h)
        
        return self.lin(all_outputs[-1])
```

### Approach 3: Temporal Graph Networks (TGN)

**Paper:** "Temporal Graph Networks for Deep Learning on Dynamic Graphs" (2020)

Uses **memory** to track node states over time.

```python
class TGNMemory(torch.nn.Module):
    """Memory module for TGN."""
    
    def __init__(self, num_nodes, memory_dim, message_dim):
        super().__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        
        # Node memory states
        self.memory = torch.zeros(num_nodes, memory_dim)
        self.last_update = torch.zeros(num_nodes)
        
        # Memory updater (GRU)
        self.gru = torch.nn.GRUCell(message_dim, memory_dim)
    
    def get_memory(self, node_ids):
        return self.memory[node_ids]
    
    def update_memory(self, node_ids, messages, timestamps):
        # Update memory with new messages
        current_memory = self.memory[node_ids]
        new_memory = self.gru(messages, current_memory)
        self.memory[node_ids] = new_memory
        self.last_update[node_ids] = timestamps


class TGN(torch.nn.Module):
    """Simplified Temporal Graph Network."""
    
    def __init__(self, num_nodes, node_dim, edge_dim, memory_dim, time_dim):
        super().__init__()
        self.memory = TGNMemory(num_nodes, memory_dim, node_dim + edge_dim + time_dim)
        
        # Temporal attention
        self.attention = torch.nn.MultiheadAttention(memory_dim, num_heads=4)
        
        # Prediction
        self.lin = torch.nn.Linear(memory_dim, 1)
    
    def forward(self, src, dst, t, edge_features):
        # Get memory for source and destination
        src_memory = self.memory.get_memory(src)
        dst_memory = self.memory.get_memory(dst)
        
        # Compute attention over temporal neighbors
        # (Simplified - real TGN uses temporal neighborhood sampling)
        combined = torch.stack([src_memory, dst_memory], dim=1)
        attn_out, _ = self.attention(combined, combined, combined)
        
        # Update memories
        messages = torch.cat([src_memory, edge_features, 
                             self.time_encode(t)], dim=1)
        self.memory.update_memory(src, messages, t)
        
        return self.lin(attn_out.mean(dim=1))
```

## Spatio-Temporal GNNs for Applications

### Traffic Prediction: ST-GCN

**Paper:** "Spatio-Temporal Graph Convolutional Networks" (2018)

Graph structure is fixed (road network), but features evolve over time.

```python
class STGCN(torch.nn.Module):
    """Spatio-Temporal GCN for traffic prediction."""
    
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 num_nodes, seq_len, pred_len):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Spatial convolution
        self.spatial_conv = GCNConv(in_channels, hidden_channels)
        
        # Temporal convolution (1D conv over time)
        self.temporal_conv = torch.nn.Conv1d(
            hidden_channels, hidden_channels, 
            kernel_size=3, padding=1
        )
        
        # Output
        self.output = torch.nn.Linear(hidden_channels * seq_len, pred_len)
    
    def forward(self, x, edge_index):
        """
        x: [batch, seq_len, num_nodes, features]
        """
        batch_size, seq_len, num_nodes, _ = x.shape
        
        # Process each time step spatially
        h_spatial = []
        for t in range(seq_len):
            h_t = self.spatial_conv(x[:, t].view(-1, x.size(-1)), edge_index)
            h_t = h_t.view(batch_size, num_nodes, -1)
            h_spatial.append(h_t)
        
        # Stack: [batch, num_nodes, seq_len, hidden]
        h = torch.stack(h_spatial, dim=2)
        
        # Temporal conv: [batch * num_nodes, hidden, seq_len]
        h = h.view(batch_size * num_nodes, -1, seq_len)
        h = F.relu(self.temporal_conv(h))
        
        # Flatten and predict
        h = h.view(batch_size, num_nodes, -1)
        return self.output(h)
```

## PyTorch Geometric Temporal

The `torch_geometric_temporal` library provides ready-to-use temporal GNNs.

```python
# pip install torch-geometric-temporal

from torch_geometric_temporal.nn.recurrent import GConvGRU, DCRNN, A3TGCN

# GConvGRU: GCN + GRU
model = GConvGRU(in_channels=8, out_channels=32, K=3)

# DCRNN: Diffusion Convolutional RNN
model = DCRNN(in_channels=8, out_channels=32, K=3)

# A3TGCN: Attention-based Temporal GCN
model = A3TGCN(in_channels=8, out_channels=32, periods=12)
```

## Comparison of Temporal GNN Methods

| Method | Graph Type | Temporal Modeling | Best For |
|--------|------------|-------------------|----------|
| **Snapshot+RNN** | Discrete | RNN over embeddings | Simple dynamics |
| **EvolveGCN** | Discrete | RNN over weights | Structure changes |
| **TGN** | Continuous | Memory + attention | Event streams |
| **ST-GCN** | Fixed structure | 1D temporal conv | Traffic/sensors |

---

## References

- Pareja, A., et al. (2020). "EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs." AAAI. [arXiv](https://arxiv.org/abs/1902.10191)
- Rossi, E., et al. (2020). "Temporal Graph Networks for Deep Learning on Dynamic Graphs." ICML Workshop. [arXiv](https://arxiv.org/abs/2006.10637)
- Yu, B., et al. (2018). "Spatio-Temporal Graph Convolutional Networks." IJCAI. [arXiv](https://arxiv.org/abs/1709.04875)

---

**Next:** [Graph Transformers →](./04-graph-transformers.md)
