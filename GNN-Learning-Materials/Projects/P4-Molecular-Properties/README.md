# 💊 Project 4: Molecular Property Prediction

**Difficulty:** ⭐⭐⭐ Advanced  
**Time:** 4-5 hours  
**Goal:** Predict toxicity of drug molecules across multiple assays

---

## 📖 Background

In drug discovery, predicting molecular properties is CRITICAL:
- 🚫 Will this molecule be toxic?
- 💧 Will it dissolve in water?
- 🧠 Will it cross the blood-brain barrier?

**Your Mission:** Build a GNN that predicts **multiple toxicity properties** for drug-like molecules!

---

## 🧠 Key Challenge: Multi-Task Learning

Unlike previous projects with ONE label, we predict **12 different properties** per molecule:

```
Molecule X:
  - NR-AR:        Toxic? → 1
  - NR-AhR:       Toxic? → 0
  - NR-ER:        Toxic? → ?  (MISSING!)
  - SR-MMP:       Toxic? → 1
  ...12 total assays
```

**Challenge:** Some labels are MISSING (experiments not done).

---

## 🚀 Task 1: Load Tox21 Dataset

### About Tox21:
| Property | Value |
|----------|-------|
| Molecules | 8,014 |
| Toxicity assays | 12 |
| Task type | Multi-label classification |
| Missing labels | ~75%! |

### 🧩 Your Task:
```python
from torch_geometric.datasets import MoleculeNet

dataset = MoleculeNet(root='./data', name='Tox21')

print(f"Number of molecules: {???}")
print(f"Number of tasks: {???}")
print(f"Node features: {???}")

# Check one molecule
mol = dataset[0]
print(f"\nFirst molecule:")
print(f"  Atoms: {???}")
print(f"  Bonds: {???}")
print(f"  Labels: {mol.y}")  # Notice the shape!
```

### 🤔 Think About It:

**Q: Why do molecular graphs have EDGE features (unlike social networks)?**

<details>
<summary>Answer</summary>

In molecules:
- Nodes = atoms (with properties like element type)
- Edges = chemical bonds (with properties like bond type: single, double, aromatic)

Bond types MATTER for chemistry! A single bond vs double bond changes the molecule entirely.
</details>

<details>
<summary>✅ Full Solution</summary>

```python
from torch_geometric.datasets import MoleculeNet

dataset = MoleculeNet(root='./data', name='Tox21')

print(f"Number of molecules: {len(dataset)}")
print(f"Number of tasks: {dataset[0].y.shape[1]}")
print(f"Node features: {dataset.num_node_features}")

mol = dataset[0]
print(f"\nFirst molecule:")
print(f"  Atoms: {mol.num_nodes}")
print(f"  Bonds: {mol.num_edges}")
print(f"  Labels: {mol.y}")
```
</details>

---

## 🚀 Task 2: Handle Missing Labels

### The Challenge:
Many labels are `NaN` (experiment not done). You can't compute loss on missing values!

### 🧩 Implement Masked Loss:
```python
import torch
import torch.nn.functional as F

def masked_bce_loss(predictions, targets):
    """
    Binary cross-entropy that IGNORES NaN labels.
    
    Args:
        predictions: Model output [batch, 12]
        targets: True labels with NaN [batch, 12]
    
    Returns:
        Loss (only on valid labels)
    """
    # Create mask: True where label is NOT NaN
    mask = ???
    
    # If no valid labels, return 0
    if mask.sum() == 0:
        return torch.tensor(0.0)
    
    # Compute loss only on valid entries
    loss = F.binary_cross_entropy_with_logits(
        predictions[mask],
        targets[mask]
    )
    return loss
```

### 🤔 Design Question:

**Q: Why use `binary_cross_entropy` instead of `cross_entropy`?**

<details>
<summary>Answer</summary>

This is **multi-label**, not multi-class!

- Multi-class: One label from N choices (use cross_entropy)
- Multi-label: Multiple independent yes/no labels (use BCE for each)

A molecule can be toxic in MULTIPLE assays simultaneously.
</details>

<details>
<summary>✅ Full Solution</summary>

```python
def masked_bce_loss(predictions, targets):
    mask = ~torch.isnan(targets)
    
    if mask.sum() == 0:
        return torch.tensor(0.0, device=predictions.device)
    
    loss = F.binary_cross_entropy_with_logits(
        predictions[mask],
        targets[mask]
    )
    return loss
```
</details>

---

## 🚀 Task 3: Build Molecular GNN

### Architecture Overview:
```
Atoms → [GIN Layers] → Atom Embeddings → [Pool] → Molecule Embedding → [MLP] → 12 Predictions
```

### Why GIN for Molecules?

<details>
<summary>Answer</summary>

GIN (Graph Isomorphism Network) is the most **expressive** standard GNN. For molecules, we need to distinguish subtle structural differences. GIN's sum aggregation preserves this information better than mean.
</details>

### 🧩 Fill in the Blanks:
```python
from torch_geometric.nn import GINConv, global_add_pool
import torch.nn as nn

class MoleculeGNN(nn.Module):
    def __init__(self, num_features, hidden_dim, num_tasks):
        super().__init__()
        
        # GIN layers (need MLPs inside!)
        self.convs = nn.ModuleList()
        
        # First layer
        mlp1 = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.convs.append(GINConv(mlp1))
        
        # More layers (you add 2-3 more!)
        ???
        
        # Classifier: molecule embedding → 12 predictions
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_tasks)
        )
    
    def forward(self, x, edge_index, batch):
        # Apply GIN layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        
        # Pool atoms → molecule
        x = ???(x, batch)  # Which pooling for molecules?
        
        # Predict
        return self.classifier(x)
```

### 🤔 Design Question:

**Q: Why `global_add_pool` instead of `global_mean_pool` for molecules?**

<details>
<summary>Answer</summary>

Molecule size matters! Larger molecules have more atoms = more opportunities for toxicity. Using SUM keeps this size information. MEAN would normalize it away.
</details>

<details>
<summary>✅ Full Solution</summary>

```python
class MoleculeGNN(nn.Module):
    def __init__(self, num_features, hidden_dim, num_tasks):
        super().__init__()
        
        self.convs = nn.ModuleList()
        
        # Layer 1
        mlp1 = nn.Sequential(nn.Linear(num_features, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.convs.append(GINConv(mlp1))
        
        # Layers 2-4
        for _ in range(3):
            mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            self.convs.append(GINConv(mlp))
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_tasks)
        )
    
    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        
        x = global_add_pool(x, batch)
        return self.classifier(x)
```
</details>

---

## 🚀 Task 4: Train and Evaluate with ROC-AUC

### Why AUC for Toxicity?

<details>
<summary>Answer</summary>

Toxicity data is **imbalanced** — most molecules are NOT toxic. Accuracy would be misleading (predict "safe" always = high accuracy!). AUC measures ranking quality regardless of threshold.
</details>

### 🧩 Implement Evaluation:
```python
from sklearn.metrics import roc_auc_score
import numpy as np

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    for batch in loader:
        out = model(batch.x, batch.edge_index, batch.batch)
        preds = torch.sigmoid(out)  # Convert to probabilities
        
        all_preds.append(preds.cpu())
        all_labels.append(batch.y.cpu())
    
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    
    # Compute AUC per task, ignoring NaN
    aucs = []
    for i in range(labels.shape[1]):
        mask = ~np.isnan(labels[:, i])
        if mask.sum() > 10:  # Need enough samples
            auc = roc_auc_score(???, ???)
            aucs.append(auc)
    
    return np.mean(aucs)
```

### 🎯 Expected Results:
- Mean AUC > 0.75 is good
- Mean AUC > 0.80 is excellent
- State-of-the-art: ~0.85

<details>
<summary>✅ Full Solution</summary>

```python
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    for batch in loader:
        out = model(batch.x, batch.edge_index, batch.batch)
        preds = torch.sigmoid(out)
        all_preds.append(preds.cpu())
        all_labels.append(batch.y.cpu())
    
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    
    aucs = []
    for i in range(labels.shape[1]):
        mask = ~np.isnan(labels[:, i])
        if mask.sum() > 10:
            try:
                auc = roc_auc_score(labels[mask, i], preds[mask, i])
                aucs.append(auc)
            except:
                pass
    
    return np.mean(aucs) if aucs else 0.0
```
</details>

---

## 🚀 Bonus Challenges

### Challenge 1: Use Edge Features
Molecules have bond types! Try incorporating `edge_attr`.

### Challenge 2: Per-Task Analysis
Which toxicity assays are hardest to predict? Why might that be?

### Challenge 3: Virtual Node
Add a "super node" connected to all atoms — often helps molecular GNNs!

---

## ✅ Project Checklist

- [ ] Loaded and explored Tox21 dataset
- [ ] Implemented masked loss for missing labels
- [ ] Built GIN-based molecular GNN
- [ ] Evaluated with ROC-AUC per task
- [ ] Achieved mean AUC > 0.75
- [ ] (Bonus) Analyzed per-task performance

---

## 🎓 What You Learned

| Concept | Key Insight |
|---------|-------------|
| **Multi-task** | Predict multiple properties at once |
| **Missing labels** | Mask NaN in loss computation |
| **GIN** | Most expressive for molecular graphs |
| **global_add_pool** | Preserves molecule size information |
| **ROC-AUC** | Better than accuracy for imbalanced data |

---

**Next Challenge:** [Project 5: Social Network Analysis →](../P5-Social-Network-Analysis/)
