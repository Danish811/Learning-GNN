# 👥 Project 5: Social Network Analysis

**Difficulty:** ⭐⭐⭐ Advanced  
**Time:** 4-5 hours  
**Goal:** Analyze and classify users in a large social network

---

## 📖 Background

Social networks are MASSIVE:
- Facebook: 3 billion users
- Twitter: 500 million users
- Twitch: 140 million users

**Your Mission:** Work with a real social network (Twitch) to:
1. Classify users
2. Detect communities
3. Scale to large graphs!

---

## 🧠 Key Challenge: Scalability

Previous projects: 2,000-8,000 nodes (fits in memory easily)

This project: **168,000 nodes, 6.8 million edges**!

```
Can't do: Full graph in one forward pass 💥
Solution: Neighbor sampling (mini-batch training)!
```

---

## 🚀 Task 1: Load and Explore Twitch Data

### About Twitch Dataset:
| Property | Value |
|----------|-------|
| Users (nodes) | ~168,000 |
| Friendships (edges) | ~6.8 million |
| Features per user | 7 |
| Task | Predict if user streams mature content |

### 🧩 Your Task:
```python
from torch_geometric.datasets import Twitch

# Load English Twitch network
dataset = Twitch(root='./data', name='EN')
data = dataset[0]

print(f"Users: {???}")
print(f"Friendships: {???}")
print(f"Features: {???}")
print(f"Classes: {???}")

# Compute average degree
degrees = ???  # Hint: torch.bincount on edge_index
print(f"Avg connections per user: {degrees.float().mean():.1f}")
```

### 🤔 Think About It:

**Q: Why can't we just use a regular GCN on this graph?**

<details>
<summary>Answer</summary>

Memory! A 2-layer GCN needs to aggregate from ALL neighbors. If average degree is 80 and we have 168K nodes:

```
Layer 1: 168K nodes × 80 neighbors = 13.4M feature vectors
Layer 2: 168K nodes × 80² neighbors = 1 BILLION feature lookups
```

Even with fancy optimizations, this won't fit!
</details>

<details>
<summary>✅ Full Solution</summary>

```python
from torch_geometric.datasets import Twitch
import torch

dataset = Twitch(root='./data', name='EN')
data = dataset[0]

print(f"Users: {data.num_nodes:,}")
print(f"Friendships: {data.num_edges:,}")
print(f"Features: {data.num_node_features}")
print(f"Classes: {data.y.unique().shape[0]}")

degrees = torch.bincount(data.edge_index[0])
print(f"Avg connections per user: {degrees.float().mean():.1f}")
print(f"Max connections: {degrees.max()}")
```
</details>

---

## 🚀 Task 2: Set Up Neighbor Sampling

### The Key Idea:
Instead of using ALL neighbors, **sample a fixed number**:

```
Full neighborhood: 500 friends
Sampled (k=25):    25 friends (randomly chosen)

Result: Fixed memory per node! ✅
```

### 🧩 Implement NeighborLoader:
```python
from torch_geometric.loader import NeighborLoader

# Create train/val/test splits
num_nodes = data.num_nodes
perm = torch.randperm(num_nodes)

train_idx = perm[:int(0.7 * num_nodes)]
val_idx = perm[int(0.7 * num_nodes):int(0.85 * num_nodes)]
test_idx = perm[int(0.85 * num_nodes):]

# Build loaders with neighbor sampling
train_loader = NeighborLoader(
    data,
    input_nodes=???,
    num_neighbors=???,  # [layer1_samples, layer2_samples]
    batch_size=???,
    shuffle=True
)

# Check batch structure
batch = next(iter(train_loader))
print(f"Batch center nodes: {batch.batch_size}")
print(f"Batch total nodes: {batch.num_nodes}")  # Includes sampled neighbors!
```

### 🤔 Design Questions:

**Q1: Why is `batch.num_nodes` > `batch.batch_size`?**

<details>
<summary>Answer</summary>

`batch_size` = the TARGET nodes we want to classify
`num_nodes` = target nodes + their sampled neighbors!

We need neighbors for message passing, even though we only predict on targets.
</details>

**Q2: How do `num_neighbors=[25, 10]` work?**

<details>
<summary>Answer</summary>

- Layer 1: Sample 25 neighbors for each target node
- Layer 2: Sample 10 neighbors for each of those 25 nodes

Total receptive field per target: up to 25 × 10 = 250 nodes (bounded!)
</details>

<details>
<summary>✅ Full Solution</summary>

```python
from torch_geometric.loader import NeighborLoader

num_nodes = data.num_nodes
perm = torch.randperm(num_nodes)

train_idx = perm[:int(0.7 * num_nodes)]
val_idx = perm[int(0.7 * num_nodes):int(0.85 * num_nodes)]
test_idx = perm[int(0.85 * num_nodes):]

train_loader = NeighborLoader(
    data,
    input_nodes=train_idx,
    num_neighbors=[25, 10],
    batch_size=1024,
    shuffle=True
)

val_loader = NeighborLoader(data, input_nodes=val_idx, num_neighbors=[25, 10], batch_size=1024)
test_loader = NeighborLoader(data, input_nodes=test_idx, num_neighbors=[25, 10], batch_size=1024)

batch = next(iter(train_loader))
print(f"Batch target nodes: {batch.batch_size}")
print(f"Batch total nodes (with neighbors): {batch.num_nodes}")
```
</details>

---

## 🚀 Task 3: Build Model (GraphSAGE)

### Why GraphSAGE here?

<details>
<summary>Answer</summary>

GraphSAGE was **designed** for this! It:
1. Works with sampled neighborhoods
2. Learns an aggregation function (not fixed embeddings)
3. Can handle new users without retraining
</details>

### 🧩 Fill in the Model:
```python
from torch_geometric.nn import SAGEConv

class SocialGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(???, ???)
        self.conv2 = SAGEConv(???, ???)
        self.dropout = 0.5
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = ???  # Activation
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
```

<details>
<summary>✅ Full Solution</summary>

```python
class SocialGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = 0.5
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
```
</details>

---

## 🚀 Task 4: Train with Mini-Batches

### 🧩 Key Training Pattern:
```python
model = SocialGNN(data.num_node_features, 128, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train_epoch():
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        optimizer.zero_grad()
        
        # Forward pass
        out = model(batch.x, batch.edge_index)
        
        # IMPORTANT: Only take predictions for target nodes!
        out = out[:???]  # How many?
        labels = batch.y[:???]
        
        loss = F.cross_entropy(out, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch.batch_size
    
    return total_loss / len(train_loader.dataset)
```

### 🤔 Critical Question:

**Q: Why `out[:batch.batch_size]` instead of all of `out`?**

<details>
<summary>Answer</summary>

The batch contains target nodes AND their sampled neighbors. But neighbors are only there for message passing — we don't have labels for them and shouldn't compute loss on them.

```
Batch structure:
  [Target1, Target2, ..., Target_bs, Neighbor1, Neighbor2, ...]
   ↑                              ↑
   We predict these               These are just for context!
```
</details>

<details>
<summary>✅ Full Solution</summary>

```python
def train_epoch():
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)[:batch.batch_size]
        loss = F.cross_entropy(out, batch.y[:batch.batch_size])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.batch_size
    
    return total_loss / len(train_idx)

# Train
for epoch in range(10):
    loss = train_epoch()
    print(f"Epoch {epoch}: Loss = {loss:.4f}")
```
</details>

---

## 🚀 Task 5: Community Detection (Bonus)

### The Idea:
Use learned embeddings to find **communities** (groups of similar users).

### 🧩 Your Task:
```python
from sklearn.cluster import KMeans

@torch.no_grad()
def get_all_embeddings():
    model.eval()
    # Use a loader that covers ALL nodes
    full_loader = NeighborLoader(
        data, 
        input_nodes=torch.arange(data.num_nodes),
        num_neighbors=[25, 10],
        batch_size=2048
    )
    
    all_embeds = []
    for batch in full_loader:
        embeds = model.conv1(batch.x, batch.edge_index)  # Stop at hidden layer
        embeds = embeds[:batch.batch_size]  # Only targets
        all_embeds.append(embeds.cpu())
    
    return torch.cat(all_embeds, dim=0)

# Get embeddings
embeddings = get_all_embeddings().numpy()

# Cluster into communities
kmeans = KMeans(n_clusters=10, random_state=42)
communities = kmeans.fit_predict(embeddings)

# Analyze communities
for i in range(10):
    members = (communities == i).sum()
    print(f"Community {i}: {members:,} users ({members/len(communities)*100:.1f}%)")
```

### 🤔 Think About:
- Do communities align with the mature/non-mature split?
- What might each community represent?

---

## ✅ Project Checklist

- [ ] Loaded large Twitch dataset (168K nodes)
- [ ] Understood why full-graph processing fails
- [ ] Implemented NeighborLoader for mini-batches
- [ ] Built GraphSAGE model
- [ ] Trained with proper target node handling
- [ ] Achieved test accuracy > 60%
- [ ] (Bonus) Detected communities

---

## 🎓 What You Learned

| Concept | Key Insight |
|---------|-------------|
| **Scalability** | Large graphs need sampling |
| **NeighborLoader** | Sample fixed neighbors per batch |
| **batch_size vs num_nodes** | Targets vs targets+neighbors |
| **GraphSAGE** | Designed for sampled training |
| **Community detection** | Cluster learned embeddings |

---

**Next Challenge:** [Project 6: Recommendation System →](../P6-Recommendation-System/)
