# 🧬 Project 3: Graph Classification

**Difficulty:** ⭐⭐ Intermediate  
**Time:** 3-4 hours  
**Goal:** Classify entire molecules as toxic or non-toxic

---

## 📖 Background

So far you've classified **nodes**. Now classify **entire graphs**!

**Real-world uses:**
- 💊 Drug toxicity prediction
- 🧪 Molecule property prediction
- 🔬 Protein function classification
- 📄 Document classification (as word graphs)

**Your Mission:** Build a GNN that reads a molecule and predicts: Toxic or Safe?

---

## 🧠 The Key Challenge

For node classification, each node gets a prediction. But for graph classification:

```
Input:  Many graphs, each with different numbers of nodes
Output: ONE label per graph

Challenge: How do you get a SINGLE vector from a graph with 50 nodes?
```

**The answer: POOLING** 🏊

---

## 🚀 Task 1: Load and Explore MUTAG Dataset

### About MUTAG:
| Property | Value |
|----------|-------|
| Molecules (graphs) | 188 |
| Classes | 2 (mutagenic or not) |
| Avg. nodes per graph | ~18 atoms |
| Node features | 7 (atom types) |

### What to Do:
1. Load MUTAG dataset
2. Print statistics
3. Visualize one molecule

### 🧩 Starter Code:
```python
from torch_geometric.datasets import TUDataset

dataset = TUDataset(root='./data', name='MUTAG')

print(f"Number of graphs: {???}")
print(f"Number of classes: {???}")
print(f"Number of node features: {???}")

# Look at one graph
graph = dataset[0]
print(f"\nFirst graph:")
print(f"  Nodes: {???}")
print(f"  Edges: {???}")
print(f"  Label: {???}")
```

### 🤔 Think About It:

**Q: Why do different graphs have different numbers of nodes?**

<details>
<summary>Answer</summary>

Each graph is a different molecule! A water molecule (H₂O) has 3 atoms, while a drug molecule might have 50+ atoms. This is fundamentally different from images, which always have fixed dimensions.
</details>

<details>
<summary>✅ Full Solution</summary>

```python
from torch_geometric.datasets import TUDataset

dataset = TUDataset(root='./data', name='MUTAG')

print(f"Number of graphs: {len(dataset)}")
print(f"Number of classes: {dataset.num_classes}")
print(f"Number of node features: {dataset.num_node_features}")

graph = dataset[0]
print(f"\nFirst graph:")
print(f"  Nodes: {graph.num_nodes}")
print(f"  Edges: {graph.num_edges}")
print(f"  Label: {graph.y.item()}")
```
</details>

---

## 🚀 Task 2: Create Train/Test Split

### The Challenge:
Split **graphs** (not nodes!) into train and test sets.

### 🧩 Your Code:
```python
from torch_geometric.loader import DataLoader
import torch

# Shuffle dataset
torch.manual_seed(42)
dataset = dataset.shuffle()

# Split: 80% train, 20% test
n = len(dataset)
train_dataset = ???
test_dataset = ???

# Create DataLoaders (batch multiple graphs!)
train_loader = DataLoader(train_dataset, batch_size=???, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=???, shuffle=False)

print(f"Training graphs: {len(train_dataset)}")
print(f"Test graphs: {len(test_dataset)}")
```

### 🤔 Key Question:

**Q: How can we "batch" graphs with different sizes?**

<details>
<summary>Answer</summary>

PyTorch Geometric creates a **mega-graph**! It combines all graphs in the batch into one large disconnected graph, with a `batch` tensor to track which node belongs to which graph.

```
Batch of 3 graphs → One big graph with 3 disconnected components
batch tensor: [0,0,0,0, 1,1,1, 2,2,2,2,2]  # Which graph each node belongs to
```
</details>

<details>
<summary>✅ Full Solution</summary>

```python
torch.manual_seed(42)
dataset = dataset.shuffle()

n = len(dataset)
train_dataset = dataset[:int(0.8 * n)]
test_dataset = dataset[int(0.8 * n):]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Training graphs: {len(train_dataset)}")
print(f"Test graphs: {len(test_dataset)}")
```
</details>

---

## 🚀 Task 3: Build Graph Classification Model

### The Architecture:
```
Nodes → [GNN Layers] → Node Embeddings → [POOLING] → Graph Embedding → [MLP] → Class
```

### 🧩 Fill in the Blanks:

```python
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F

class GraphClassifier(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super().__init__()
        # GNN layers
        self.conv1 = ???
        self.conv2 = ???
        self.conv3 = ???
        
        # Final classifier
        self.classifier = torch.nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x, edge_index, batch):
        # 1. Node embeddings (3 GNN layers)
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        
        # 2. Pool: aggregate all nodes in each graph
        x = ???(x, batch)  # Which pooling function?
        
        # 3. Classify
        x = self.classifier(x)
        return x
```

### 🤔 Design Questions:

**Q1: What does `global_mean_pool(x, batch)` do?**

<details>
<summary>Answer</summary>

It averages all node embeddings within each graph:
```
Graph 0: nodes [0,1,2] → mean([x_0, x_1, x_2]) → graph_embedding_0
Graph 1: nodes [3,4]   → mean([x_3, x_4]) → graph_embedding_1
```
</details>

**Q2: Why do we need the `batch` tensor?**

<details>
<summary>Answer</summary>

When we batch multiple graphs, they're combined into one big graph. The `batch` tensor tells us which nodes belong to which original graph, so we can pool them correctly.
</details>

**Q3: What if we used `global_max_pool` instead?**

<details>
<summary>Answer</summary>

`max_pool` takes the maximum value per dimension across all nodes. This captures the "strongest signal" in each graph. 

- `mean_pool`: Good for overall properties
- `max_pool`: Good for detecting presence of specific features
- You can even combine both!
</details>

<details>
<summary>✅ Full Solution</summary>

```python
from torch_geometric.nn import GCNConv, global_mean_pool

class GraphClassifier(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = torch.nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        
        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        return x
```
</details>

---

## 🚀 Task 4: Training Loop

### 🧩 Fill in the Blanks:

```python
model = GraphClassifier(
    num_features=dataset.num_node_features,
    hidden_dim=64,
    num_classes=dataset.num_classes
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train_epoch():
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        optimizer.zero_grad()
        
        # Forward pass - notice we pass batch.batch!
        out = model(???, ???, ???)
        
        # Loss
        loss = F.cross_entropy(out, ???)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(train_loader)
```

### 🤔 Common Mistake:

**Q: What's `batch.batch` vs `batch`?**

<details>
<summary>Answer</summary>

Confusing naming, but important!

- `batch` (the variable): A Batch object containing all data for multiple graphs
- `batch.batch` (the attribute): A tensor telling which graph each node belongs to

```python
# batch = Batch object with x, edge_index, y, batch...
# batch.batch = tensor([0,0,0, 1,1, 2,2,2,2, ...])
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
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

# Train
for epoch in range(100):
    loss = train_epoch()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")
```
</details>

---

## 🚀 Task 5: Evaluate

### 🧩 Your Implementation:

```python
@torch.no_grad()
def test(loader):
    model.eval()
    correct = 0
    total = 0
    
    for batch in loader:
        out = model(batch.x, batch.edge_index, batch.batch)
        pred = ???  # Get predicted class
        correct += ???  # Count correct
        total += ???    # Count total
    
    return correct / total

train_acc = test(train_loader)
test_acc = test(test_loader)
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
```

### 🎯 Expected Results:
- Test Accuracy: ~75-85%
- Higher with better architecture!

<details>
<summary>✅ Full Solution</summary>

```python
@torch.no_grad()
def test(loader):
    model.eval()
    correct = 0
    total = 0
    
    for batch in loader:
        out = model(batch.x, batch.edge_index, batch.batch)
        pred = out.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total += batch.y.size(0)
    
    return correct / total

train_acc = test(train_loader)
test_acc = test(test_loader)
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
```
</details>

---

## 🚀 Bonus: Try GIN for Better Results!

GIN (Graph Isomorphism Network) is designed for graph classification!

### 🤔 Why GIN?

<details>
<summary>Answer</summary>

GIN is **maximally powerful** among message passing GNNs. It uses sum aggregation (not mean), which preserves more structural information. This matters for distinguishing different molecular structures!
</details>

### Challenge: Swap GCNConv for GINConv

<details>
<summary>💡 Hint</summary>

```python
from torch_geometric.nn import GINConv

# GIN needs an MLP inside each layer
mlp = torch.nn.Sequential(
    torch.nn.Linear(in_dim, hidden),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden, hidden)
)
conv = GINConv(mlp)
```
</details>

---

## ✅ Project Checklist

- [ ] Loaded and explored MUTAG dataset
- [ ] Created graph-level train/test splits
- [ ] Built model with GNN + pooling
- [ ] Understood batching of variable-size graphs
- [ ] Achieved ~80% test accuracy
- [ ] (Bonus) Tried GIN for better results

---

## 🎓 What You Learned

| Concept | Key Insight |
|---------|-------------|
| **Graph classification** | One label per entire graph |
| **Pooling** | Aggregate node features → graph feature |
| **Batching graphs** | Combine into one mega-graph |
| **batch tensor** | Tracks which node → which graph |
| **GIN** | Best for graph-level tasks |

---

**Next Challenge:** [Project 4: Molecular Property Prediction →](../P4-Molecular-Properties/)
