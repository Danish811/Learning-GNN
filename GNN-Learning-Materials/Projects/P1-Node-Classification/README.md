# 🎯 Project 1: Node Classification

**Difficulty:** ⭐ Beginner  
**Time:** 2-3 hours  
**Goal:** Classify research papers by topic using their citation network

---

## 📖 Background

You have a dataset of scientific papers where:
- Each paper has a **feature vector** (bag-of-words)
- Papers **cite** each other (edges)
- Each paper belongs to one of 7 topics (labels)

**Your Mission:** Build a GNN that predicts the topic of papers using BOTH the paper's content AND its citation relationships!

---

## 🗂️ Dataset: Cora

| Property | Value |
|----------|-------|
| Papers (nodes) | 2,708 |
| Citations (edges) | 5,429 |
| Features per paper | 1,433 (word presence) |
| Classes | 7 topics |

---

## 🚀 Task 1: Load and Explore the Data

### What to Do:
1. Load the Cora dataset using PyTorch Geometric
2. Print the number of nodes, edges, and features
3. Check how the train/validation/test splits are defined

### Starter Code:
```python
from torch_geometric.datasets import Planetoid

# Your code here: Load Cora dataset
dataset = ???
data = ???

# Print statistics
print(f"Nodes: {???}")
print(f"Edges: {???}")
print(f"Features: {???}")
print(f"Classes: {???}")
```

<details>
<summary>💡 Hint 1: How to load</summary>

```python
dataset = Planetoid(root='./data', name='Cora')
data = dataset[0]  # First (only) graph
```
</details>

<details>
<summary>💡 Hint 2: Properties to check</summary>

- `data.num_nodes` — number of nodes
- `data.num_edges` — number of edges  
- `data.num_node_features` — features per node
- `dataset.num_classes` — number of classes
</details>

<details>
<summary>✅ Full Solution</summary>

```python
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='./data', name='Cora')
data = dataset[0]

print(f"Nodes: {data.num_nodes}")
print(f"Edges: {data.num_edges}")
print(f"Features: {data.num_node_features}")
print(f"Classes: {dataset.num_classes}")
print(f"Train nodes: {data.train_mask.sum()}")
print(f"Val nodes: {data.val_mask.sum()}")
print(f"Test nodes: {data.test_mask.sum()}")
```
</details>

### 🤔 Think About It:
1. **Why is the number of training nodes so small?** (~5% of data)
2. **How can a GNN still learn with so few labels?**

<details>
<summary>Answer</summary>

GNNs leverage the **graph structure**! Even unlabeled nodes participate in message passing, spreading information through the network. This is called **semi-supervised learning**.
</details>

---

## 🚀 Task 2: Build a Simple GCN Model

### What to Do:
1. Create a 2-layer GCN model
2. Use 16 hidden units
3. Apply ReLU activation and dropout

### 🧩 Fill in the Blanks:
```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = ???  # First GCN layer
        self.conv2 = ???  # Second GCN layer
    
    def forward(self, x, edge_index):
        x = self.conv1(???)  # Apply first layer
        x = ???              # Activation function
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(???)  # Apply second layer
        return x
```

### 🤔 Design Questions:

**Q1: Why do we use dropout BETWEEN layers, not after the last layer?**

<details>
<summary>Answer</summary>

Dropout is a regularization technique. After the last layer, we want the final predictions — adding noise there would hurt accuracy. We use dropout during training to prevent overfitting.
</details>

**Q2: Why only 2 layers? What happens with 10 layers?**

<details>
<summary>Answer</summary>

**Over-smoothing!** Each GCN layer averages neighbor features. With too many layers, all nodes start looking the same — their features get "smoothed out" across the entire graph.
</details>

<details>
<summary>💡 Hint: GCNConv signature</summary>

```python
GCNConv(in_channels, out_channels)
```

Called with: `conv(x, edge_index)`
</details>

<details>
<summary>✅ Full Solution</summary>

```python
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
```
</details>

---

## 🚀 Task 3: Train the Model

### What to Do:
1. Create the model, optimizer, and loss function
2. Write a training loop for 200 epochs
3. Only compute loss on **training nodes** (use `train_mask`)

### 🧩 Fill in the Blanks:
```python
model = GCN(
    in_channels=???, 
    hidden_channels=16, 
    out_channels=???
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    
    out = model(???, ???)  # Forward pass
    loss = F.cross_entropy(out[???], data.y[???])  # Loss on train nodes only
    
    loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
```

### 🤔 Design Questions:

**Q1: Why do we compute loss ONLY on training nodes?**

<details>
<summary>Answer</summary>

This is semi-supervised learning! We only have labels for a few nodes. The unlabeled nodes still participate in message passing (the forward pass), but we can only compute loss where we have ground truth.
</details>

**Q2: What would happen if we used ALL nodes for loss?**

<details>
<summary>Answer</summary>

We'd be cheating! The test nodes' labels should be hidden during training. Plus, validation nodes are held out to tune hyperparameters.
</details>

<details>
<summary>✅ Full Solution</summary>

```python
model = GCN(
    in_channels=dataset.num_features,
    hidden_channels=16,
    out_channels=dataset.num_classes
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    
    loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
```
</details>

---

## 🚀 Task 4: Evaluate on Test Set

### What to Do:
1. Switch model to evaluation mode
2. Get predictions using `argmax`
3. Compute accuracy on test nodes only

### 🧩 Your Code:
```python
model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index)
    pred = ???  # Get predicted class (hint: argmax)
    
    correct = ???  # Count correct predictions on test set
    accuracy = ???
    
print(f"Test Accuracy: {accuracy:.4f}")
```

### 🎯 Expected Result:
- GCN should achieve ~**79-82%** accuracy
- If much lower, check your implementation!

<details>
<summary>✅ Full Solution</summary>

```python
model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    accuracy = int(correct) / int(data.test_mask.sum())
    
print(f"Test Accuracy: {accuracy:.4f}")
```
</details>

---

## 🚀 Bonus Challenge: Beat GCN with GAT!

### Mission:
Replace GCN with GAT (Graph Attention Network) and see if you can get better accuracy!

### Changes Needed:
1. Import `GATConv` instead of `GCNConv`
2. Add `heads=8` for multi-head attention
3. Adjust hidden dimensions (GAT concatenates heads!)

### 🤔 Think About:
- **Why might GAT perform better?**
- **What's the tradeoff?**

<details>
<summary>💡 Hint: GAT changes</summary>

```python
from torch_geometric.nn import GATConv

# Layer 1: 8 heads, each outputs 8 features → total 64
self.conv1 = GATConv(in_channels, 8, heads=8, dropout=0.6)

# Layer 2: Input is 64 (8 heads × 8 features)
self.conv2 = GATConv(64, out_channels, heads=1, concat=False)
```
</details>

<details>
<summary>✅ Full GAT Solution</summary>

```python
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, 
                             concat=False, dropout=0.6)
    
    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# Expected: ~83% accuracy (better than GCN!)
```
</details>

---

## ✅ Project Checklist

- [ ] Loaded and explored Cora dataset
- [ ] Built a 2-layer GCN model
- [ ] Trained with proper train/test split
- [ ] Achieved ~80% accuracy
- [ ] (Bonus) Beat GCN with GAT

---

## 🎓 What You Learned

| Concept | Key Insight |
|---------|-------------|
| **Semi-supervised** | Learn from few labels + graph structure |
| **GCN layer** | Aggregates neighbor information |
| **Over-smoothing** | Why we use 2-3 layers, not more |
| **Train mask** | Only compute loss on labeled nodes |
| **GAT advantage** | Learned attention > fixed weights |

---

**Next Challenge:** [Project 2: Link Prediction →](../P2-Link-Prediction/)
