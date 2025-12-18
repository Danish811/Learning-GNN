# 🔗 Project 2: Link Prediction

**Difficulty:** ⭐⭐ Intermediate  
**Time:** 3-4 hours  
**Goal:** Predict which papers will cite each other in the future

---

## 📖 Background

**Link prediction** answers: "Will these two nodes connect?"

Real-world uses:
- 👥 Friend recommendations (Facebook)
- 🛒 Product recommendations (Amazon)
- 📚 Citation prediction (Academia)
- 💊 Drug-protein interaction (Pharma)

**Your Mission:** Build a GNN that learns node embeddings, then predict whether two nodes should be connected!

---

## 🧠 The Key Insight

Link prediction is different from node classification:

```
Node Classification: Node → Label
Link Prediction:     (Node A, Node B) → Connected? Yes/No
```

**The approach:**
1. Learn good node embeddings with a GNN
2. For any pair (A, B), combine their embeddings
3. Predict: Are they connected?

---

## 🚀 Task 1: Understand the Data Split

### The Challenge:
In link prediction, we need to hide some edges during training!

```
Original Graph:        Training Graph:        Test:
A ─── B                A ─── B               Does C──D exist?
│     │                │     │               Does A──C exist?
C ─── D                C     D               
                       (edge hidden!)
```

### What to Do:
1. Load the Cora dataset
2. Use `RandomLinkSplit` to create train/val/test edge splits
3. Print how many edges are in each split

### 🧩 Starter Code:
```python
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import RandomLinkSplit

dataset = Planetoid(root='./data', name='Cora')
data = dataset[0]

# Split edges into train/val/test
transform = RandomLinkSplit(
    num_val=???,      # Fraction for validation
    num_test=???,     # Fraction for test
    is_undirected=True,
    add_negative_train_samples=???  # Do we need fake "non-edges"?
)

train_data, val_data, test_data = transform(data)

print(f"Training edges: {???}")
print(f"Test edges: {???}")
```

### 🤔 Think About It:

**Q1: Why do we need "negative samples" (non-edges)?**

<details>
<summary>Answer</summary>

We're training a classifier! We need examples of:
- ✅ **Positive**: Real edges (connected nodes)
- ❌ **Negative**: Non-edges (unconnected nodes)

Otherwise the model would just predict "connected" for everything!
</details>

**Q2: Why must test edges be HIDDEN during training?**

<details>
<summary>Answer</summary>

If the model sees test edges during message passing, it's **cheating**! The model could learn to just memorize the graph structure instead of learning generalizable patterns.
</details>

<details>
<summary>💡 Hint</summary>

```python
transform = RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    is_undirected=True,
    add_negative_train_samples=True
)
```

Each split has:
- `edge_index` — edges for message passing
- `edge_label_index` — edges to predict
- `edge_label` — 1 (positive) or 0 (negative)
</details>

<details>
<summary>✅ Full Solution</summary>

```python
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import RandomLinkSplit

dataset = Planetoid(root='./data', name='Cora')
data = dataset[0]

transform = RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    is_undirected=True,
    add_negative_train_samples=True
)

train_data, val_data, test_data = transform(data)

print(f"Training edges for message passing: {train_data.edge_index.shape}")
print(f"Training edges to predict: {train_data.edge_label_index.shape}")
print(f"Test edges to predict: {test_data.edge_label_index.shape}")
```
</details>

---

## 🚀 Task 2: Build the Encoder (GNN)

### What to Do:
Create a GNN that produces node embeddings. This is the "encoder" part.

### 🧩 Fill in the Blanks:
```python
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class GNNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = ???
        self.conv2 = ???
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = ???  # Activation
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x  # These are our node embeddings!
```

### 🤔 Design Question:

**Why don't we apply softmax at the end (like in node classification)?**

<details>
<summary>Answer</summary>

In node classification, we output class probabilities.

In link prediction, we output **embeddings** — continuous vectors that capture node properties. We'll use these embeddings to THEN predict edges.
</details>

<details>
<summary>✅ Full Solution</summary>

```python
class GNNEncoder(torch.nn.Module):
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

## 🚀 Task 3: Build the Decoder (Link Predictor)

### The Challenge:
Given embeddings for nodes A and B, predict if they're connected.

### Common Approaches:

| Method | Formula | Pros/Cons |
|--------|---------|-----------|
| Dot product | z_A · z_B | Simple, fast |
| Concatenate + MLP | MLP([z_A; z_B]) | Learnable, expressive |
| Hadamard + MLP | MLP(z_A ⊙ z_B) | Good balance |

### 🧩 Your Task:
Implement a simple dot-product decoder:

```python
def decode(z, edge_index):
    """
    Predict link probability for given edges.
    
    Args:
        z: Node embeddings [num_nodes, embed_dim]
        edge_index: Edges to predict [2, num_edges]
    
    Returns:
        Link probabilities [num_edges]
    """
    src = edge_index[0]  # Source nodes
    dst = edge_index[1]  # Destination nodes
    
    # Get embeddings for source and destination
    z_src = ???
    z_dst = ???
    
    # Dot product (element-wise multiply, then sum)
    return ???
```

### 🤔 Think About It:

**Q: Why does dot product make sense for link prediction?**

<details>
<summary>Answer</summary>

Dot product measures **similarity**. If two nodes have similar embeddings (pointing same direction), their dot product is HIGH. 

The assumption: Similar nodes should connect!
</details>

<details>
<summary>✅ Full Solution</summary>

```python
def decode(z, edge_index):
    src = edge_index[0]
    dst = edge_index[1]
    
    z_src = z[src]
    z_dst = z[dst]
    
    # Dot product per edge
    return (z_src * z_dst).sum(dim=-1)
```
</details>

---

## 🚀 Task 4: Training Loop

### What to Do:
1. Encode nodes to get embeddings
2. Decode edge pairs to get predictions
3. Use binary cross-entropy loss
4. Train!

### 🧩 Fill in the Blanks:
```python
encoder = GNNEncoder(dataset.num_features, 128, 64)
optimizer = torch.optim.Adam(encoder.parameters(), lr=0.01)

def train_epoch():
    encoder.train()
    optimizer.zero_grad()
    
    # 1. Get embeddings using TRAINING graph
    z = encoder(train_data.x, ???)
    
    # 2. Decode: predict edges
    pred = decode(z, ???)
    
    # 3. Ground truth labels
    labels = ???
    
    # 4. Binary cross-entropy loss
    loss = F.binary_cross_entropy_with_logits(pred, labels.float())
    
    loss.backward()
    optimizer.step()
    
    return loss.item()
```

### 🤔 Key Question:

**Q: Why use `edge_index` for encoding but `edge_label_index` for decoding?**

<details>
<summary>Answer</summary>

- `edge_index`: The edges used for **message passing** (training graph)
- `edge_label_index`: The edges we're trying to **predict** (includes hidden test edges)

We propagate messages on known edges, but predict on potentially new edges!
</details>

<details>
<summary>✅ Full Solution</summary>

```python
def train_epoch():
    encoder.train()
    optimizer.zero_grad()
    
    z = encoder(train_data.x, train_data.edge_index)
    pred = decode(z, train_data.edge_label_index)
    labels = train_data.edge_label
    
    loss = F.binary_cross_entropy_with_logits(pred, labels.float())
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

# Train for 100 epochs
for epoch in range(100):
    loss = train_epoch()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")
```
</details>

---

## 🚀 Task 5: Evaluate with AUC

### What to Do:
Compute Area Under ROC Curve (AUC) on test edges.

### 🧩 Your Implementation:
```python
from sklearn.metrics import roc_auc_score

@torch.no_grad()
def evaluate(data):
    encoder.eval()
    
    z = encoder(data.x, data.edge_index)
    pred = decode(z, data.edge_label_index)
    pred = ???  # Convert logits to probabilities (hint: sigmoid)
    
    labels = data.edge_label
    
    auc = roc_auc_score(???, ???)  # (true labels, predictions)
    return auc

test_auc = evaluate(test_data)
print(f"Test AUC: {test_auc:.4f}")
```

### 🎯 Expected Results:
- AUC > 0.85 is good
- AUC > 0.90 is excellent

<details>
<summary>✅ Full Solution</summary>

```python
@torch.no_grad()
def evaluate(data):
    encoder.eval()
    
    z = encoder(data.x, data.edge_index)
    pred = decode(z, data.edge_label_index)
    pred = torch.sigmoid(pred)
    
    labels = data.edge_label.cpu()
    pred = pred.cpu()
    
    auc = roc_auc_score(labels, pred)
    return auc

test_auc = evaluate(test_data)
print(f"Test AUC: {test_auc:.4f}")
```
</details>

---

## 🚀 Bonus Challenge: Beat the Baseline!

### Ideas to Try:
1. **Better encoder**: Use GAT instead of GCN
2. **Better decoder**: MLP instead of dot product
3. **More embeddings dimensions**: Try 128 or 256
4. **Different negative sampling ratio**

### 🤔 Think:
- Which improvement has the biggest impact?
- What's the tradeoff of a more complex decoder?

---

## ✅ Project Checklist

- [ ] Understood train/test edge splits
- [ ] Built GNN encoder for embeddings
- [ ] Implemented dot-product decoder
- [ ] Trained with proper loss function
- [ ] Evaluated with AUC metric
- [ ] (Bonus) Tried improvements

---

## 🎓 What You Learned

| Concept | Key Insight |
|---------|-------------|
| **Link prediction** | Predicting connections, not labels |
| **Encoder-decoder** | GNN encodes, decoder predicts |
| **Negative sampling** | Need non-edges for training |
| **Edge splits** | Hide test edges during training |
| **AUC metric** | Better than accuracy for imbalanced data |

---

**Next Challenge:** [Project 3: Graph Classification →](../P3-Graph-Classification/)
