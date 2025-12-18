# 🚀 GraphSAGE (Sample and Aggregate)

> *"GraphSAGE is like learning HOW to make friends, not just memorizing who your current friends are!"*

**Paper:** "Inductive Representation Learning on Large Graphs" (Hamilton et al., 2017)

---

## 📖 The Story So Far

You've learned two amazing architectures:

| Architecture | Superpower | Limitation |
|--------------|------------|------------|
| [GCN](./01-gcn.md) | Simple & elegant | All neighbors equal |
| [GAT](./02-gat.md) | Learned attention | Can't handle new nodes |

**Both have the same fatal flaw:**

```
Training: Learn embeddings for nodes A, B, C, D, E
          ↓
Model: "I know exactly where A, B, C, D, E belong!"
          ↓
New node F joins: "Who's F? I've never seen F! 😱"
          ↓
Solution: Retrain entire model 😩
```

**GraphSAGE's Revolution:** Learn a **function** that works for ANY node, even new ones!

![GraphSAGE: Sampling neighbors for scalable learning](./images/graphsage_sampling.png)

---

## 💡 The Big Idea: Inductive vs Transductive

### The Old Way (Transductive)

```
GCN/GAT approach:

Training → Fixed embedding table:
  Node A → [0.2, 0.8, 0.1, ...]
  Node B → [0.5, 0.3, 0.7, ...]
  Node C → [0.9, 0.1, 0.4, ...]
  
New node D? → ❌ Not in the table! Must retrain!
```

### The GraphSAGE Way (Inductive)

```
GraphSAGE approach:

Training → Learn an AGGREGATION FUNCTION:
  f(node features, neighbor features) → embedding
  
New node D? → ✅ Just apply f() to D's neighbors!
  embedding_D = f(D's features, D's neighbor features)
  
No retraining needed! 🎉
```

---

## 🎯 Real-World Example

### Why This Matters: Pinterest

Pinterest uses GraphSAGE (called "PinSage") for recommendations:

```
Pinterest problem:
- 2+ billion pins (images)
- 100 million new pins per month!
- Can't retrain for every new pin 😅

PinSage solution:
- Learn how to generate embeddings
- New pin → Sample neighbors → Get embedding → Recommend!
- Works instantly for new content ⚡
```

---

## 🔧 How GraphSAGE Works

### The Three Key Steps

```
For each node v:

1️⃣ SAMPLE: Pick a fixed number of neighbors
   (Not all neighbors — just K random ones!)

2️⃣ AGGREGATE: Combine their features
   (Mean, max-pool, or LSTM)

3️⃣ COMBINE: Merge with your own features
   (Concatenate + transform)
```

### Visual Walkthrough

```
Step 1: SAMPLE (k=2 neighbors)

Original neighbors of A:        Sampled:
    B                              B ✓
    C                              C ✓
    D                              D ✗ (not sampled)
    E                              E ✗
    F                              F ✗
    
Now A only considers 2 neighbors — fixed size!
```

```
Step 2: AGGREGATE

Sampled neighbors: B, C
Their features: h_B = [0.2, 0.8], h_C = [0.4, 0.6]

Mean aggregation:
  h_neighbors = mean(h_B, h_C) = [0.3, 0.7]
```

```
Step 3: COMBINE

A's own features: h_A = [0.5, 0.5]
Aggregated neighbors: [0.3, 0.7]

h_new = ReLU( W × concat(h_A, h_neighbors) )
      = ReLU( W × [0.5, 0.5, 0.3, 0.7] )
```

---

## 🔄 Aggregation Functions

GraphSAGE offers THREE ways to aggregate neighbors:

### 1️⃣ Mean Aggregator (Simple & Fast)

```python
def mean_aggregate(neighbor_features):
    return neighbor_features.mean(dim=0)
```
**Good for:** General use, similar to GCN

### 2️⃣ Max-Pool Aggregator (Capture Extremes)

```python
def pool_aggregate(neighbor_features, W, b):
    # Transform each neighbor
    transformed = ReLU(W @ neighbor_features + b)
    # Take element-wise maximum
    return transformed.max(dim=0)
```
**Good for:** When one strong signal matters

### 3️⃣ LSTM Aggregator (Order-Aware)

```python
def lstm_aggregate(neighbor_features):
    # Random order (since graphs have no inherent order)
    shuffled = random_permutation(neighbor_features)
    # Run through LSTM
    _, (h_final, _) = lstm(shuffled)
    return h_final
```
**Good for:** Complex patterns (but slower)

---

## 🎲 Why Sample? (The Secret Sauce)

### The Problem with Full Neighborhoods

```
Social network node with 5000 friends:

Full aggregation:
- Layer 1: 5000 neighbors
- Layer 2: 5000 × 5000 = 25,000,000 nodes!
- 💥 Memory explodes!
```

### Sampling to the Rescue

```
GraphSAGE (sample 25 per layer):

- Layer 1: 25 neighbors
- Layer 2: 25 × 25 = 625 nodes
- ✅ Fixed, manageable size!
```

### Mini-Batch Training

```python
from torch_geometric.loader import NeighborLoader

# Sample neighbors for mini-batches
loader = NeighborLoader(
    data,
    num_neighbors=[25, 10],  # 25 at layer 1, 10 at layer 2
    batch_size=512,
    shuffle=True
)

for batch in loader:
    # batch contains sampled subgraphs
    out = model(batch.x, batch.edge_index)
    # Only 512 target nodes, but with their sampled neighbors!
```

---

## 🐍 Code: GraphSAGE Implementation

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_features, hidden, out_features):
        super().__init__()
        self.conv1 = SAGEConv(in_features, hidden)
        self.conv2 = SAGEConv(hidden, out_features)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
```

### Full Training with Sampling

```python
from torch_geometric.loader import NeighborLoader
from torch_geometric.datasets import Reddit

# Reddit dataset: 233K nodes, 114M edges (BIG!)
dataset = Reddit(root='./data')
data = dataset[0]

# Create sampled mini-batches
train_loader = NeighborLoader(
    data,
    num_neighbors=[25, 10],
    batch_size=1024,
    input_nodes=data.train_mask,
    shuffle=True
)

model = GraphSAGE(dataset.num_features, 256, dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(10):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)[:batch.batch_size]
        loss = F.cross_entropy(out, batch.y[:batch.batch_size])
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} done!")

# Can now handle 233K+ nodes! 🎉
```

---

## ✨ The Inductive Magic

### Handling New Nodes

```python
# New user joins the social network!
new_user_features = torch.randn(1, num_features)  # Their profile
new_edges = [[existing_friend1, existing_friend2], 
             [new_user_id, new_user_id]]  # Their connections

# Just add them to the graph
extended_x = torch.cat([data.x, new_user_features])
extended_edges = torch.cat([data.edge_index, new_edges], dim=1)

# Get embedding for new user — NO RETRAINING!
with torch.no_grad():
    embeddings = model(extended_x, extended_edges)
    new_user_embedding = embeddings[-1]  # Last node = new user

print("New user classified! Welcome aboard! 🎉")
```

---

## ⚖️ The Big Comparison (All Three!)

| Feature | GCN | GAT | GraphSAGE |
|---------|-----|-----|-----------|
| **Aggregation** | Mean (normalized) | Attention-weighted | Flexible (mean/max/LSTM) |
| **Neighbor weights** | By degree | Learned | Fixed (mean) or learned |
| **New nodes** | ❌ Retrain | ❌ Retrain | ✅ Works! |
| **Scalability** | Medium | Medium | ✅ Excellent |
| **Best for** | Small static graphs | Interpretability | Large/dynamic graphs |

---

## 🎓 Key Takeaways

| Concept | Remember This |
|---------|---------------|
| **Inductive** | Learn a function, not fixed embeddings |
| **Sampling** | Fixed-size neighborhoods → scalable |
| **Aggregate + Combine** | Separate neighbor info from self info |
| **Mini-batch** | Use NeighborLoader for big graphs |
| **New nodes** | Just apply the learned function! |

---

## 🎮 Quick Quiz

1. **What's "inductive" learning?**
   <details>
   <summary>Answer</summary>
   Learning a FUNCTION that can generalize to new, unseen nodes (vs. memorizing specific node embeddings)
   </details>

2. **Why sample neighbors instead of using all?**
   <details>
   <summary>Answer</summary>
   Scalability! With 2 layers and nodes having 1000 neighbors each, you'd need 1,000,000 node features — sampling keeps it fixed at, say, 25 × 25 = 625
   </details>

3. **Can GraphSAGE classify a brand new node?**
   <details>
   <summary>Answer</summary>
   YES! That's its superpower. Just connect the new node to existing ones and run the model — no retraining needed!
   </details>

---

## 🏆 Which Architecture Should You Use?

```
Decision Tree:

Need new node handling? ──Yes──▶ GraphSAGE
        │ No
        ▼
Need interpretable attention? ──Yes──▶ GAT
        │ No
        ▼
Want simple baseline? ──Yes──▶ GCN
        │ No
        ▼
Large graph (>1M)? ──Yes──▶ GraphSAGE + NeighborLoader
        │ No
        ▼
Just start with GCN, it's a solid baseline! ✅
```

---

**Ready to see them all side by side?**

👉 **[Next: Architecture Comparison →](./04-comparison.md)** 📊

---

*"GraphSAGE: Learn HOW to understand nodes, not just WHICH nodes!"* 🧠
