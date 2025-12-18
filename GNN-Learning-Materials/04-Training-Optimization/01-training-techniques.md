# GNN Training Techniques

Best practices for training Graph Neural Networks effectively.

## Loss Functions

### Node Classification

```python
import torch.nn.functional as F

# Standard cross-entropy
loss = F.cross_entropy(out[train_mask], labels[train_mask])

# With class weights for imbalanced data
class_weights = torch.tensor([1.0, 5.0, 2.0])  # Weight rare classes higher
loss = F.cross_entropy(out[train_mask], labels[train_mask], weight=class_weights)
```

### Link Prediction

```python
def link_prediction_loss(pos_pred, neg_pred):
    """Binary cross-entropy for link prediction."""
    pos_loss = F.binary_cross_entropy(pos_pred, torch.ones_like(pos_pred))
    neg_loss = F.binary_cross_entropy(neg_pred, torch.zeros_like(neg_pred))
    return pos_loss + neg_loss

# Or: Margin-based loss
def margin_loss(pos_scores, neg_scores, margin=1.0):
    return torch.clamp(margin - pos_scores + neg_scores, min=0).mean()
```

### Graph Classification

```python
# Standard classification
loss = F.cross_entropy(graph_pred, graph_labels)

# Multi-label (e.g., Tox21)
loss = F.binary_cross_entropy_with_logits(pred, labels)

# Regression (e.g., QM9)
loss = F.mse_loss(pred, targets)
# Or L1 for outlier robustness
loss = F.l1_loss(pred, targets)
```

### Contrastive Learning

```python
def contrastive_loss(z1, z2, temperature=0.5):
    """NT-Xent loss for graph contrastive learning."""
    batch_size = z1.size(0)
    
    z = torch.cat([z1, z2], dim=0)  # [2B, D]
    z = F.normalize(z, dim=1)
    
    sim = z @ z.t() / temperature  # [2B, 2B]
    
    # Mask out self-similarity
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim.masked_fill_(mask, float('-inf'))
    
    # Positive pairs: (i, i+B) and (i+B, i)
    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size),
        torch.arange(batch_size)
    ]).to(z.device)
    
    return F.cross_entropy(sim, labels)
```

## Regularization

### Dropout Strategies

```python
class GCNWithDropout(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
    
    def forward(self, x, edge_index):
        # Input dropout
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        # Hidden dropout
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv2(x, edge_index)
        return x
```

### DropEdge

```python
from torch_geometric.utils import dropout_edge

def train_with_dropedge(model, data, optimizer, drop_rate=0.5):
    model.train()
    optimizer.zero_grad()
    
    # Randomly drop edges
    edge_index, _ = dropout_edge(data.edge_index, p=drop_rate, training=True)
    
    out = model(data.x, edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()
```

### Weight Decay

```python
# L2 regularization via weight_decay
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
```

### Label Smoothing

```python
def label_smoothing_loss(pred, target, smoothing=0.1):
    n_classes = pred.size(-1)
    one_hot = F.one_hot(target, n_classes).float()
    smooth_target = one_hot * (1 - smoothing) + smoothing / n_classes
    log_prob = F.log_softmax(pred, dim=-1)
    return -(smooth_target * log_prob).sum(dim=-1).mean()
```

## Normalization

### Batch Normalization

```python
class GCNWithBN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x
```

### Layer Normalization

```python
class GCNWithLN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x
```

### GraphNorm

```python
from torch_geometric.nn import GraphNorm

class GCNWithGraphNorm(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.gn1 = GraphNorm(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
    
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.gn1(x, batch)  # Normalize per graph
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x
```

## Learning Rate Scheduling

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

# Reduce on plateau (good for node classification)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
# Usage: scheduler.step(val_accuracy)

# Cosine annealing (good for graph classification)
scheduler = CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)
# Usage: scheduler.step()

# Warmup + Decay
def warmup_lr_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

## Early Stopping

```python
class EarlyStopping:
    def __init__(self, patience=20, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.early_stop

# Usage
early_stopping = EarlyStopping(patience=50)
for epoch in range(1000):
    train(...)
    val_acc = evaluate(...)
    if early_stopping(val_acc):
        print(f"Early stopping at epoch {epoch}")
        break
```

## Training Template

```python
import torch
from torch_geometric.loader import NeighborLoader

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    total_examples = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        out = model(batch.x, batch.edge_index)[:batch.batch_size]
        loss = F.cross_entropy(out, batch.y[:batch.batch_size])
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()
        
        total_loss += loss.item() * batch.batch_size
        total_examples += batch.batch_size
    
    return total_loss / total_examples

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)[:batch.batch_size]
        pred = out.argmax(dim=-1)
        correct += (pred == batch.y[:batch.batch_size]).sum().item()
        total += batch.batch_size
    
    return correct / total

# Main training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNN(...).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=10)

best_val = 0
for epoch in range(500):
    loss = train_epoch(model, train_loader, optimizer, device)
    val_acc = evaluate(model, val_loader, device)
    test_acc = evaluate(model, test_loader, device)
    
    scheduler.step(val_acc)
    
    if val_acc > best_val:
        best_val = val_acc
        torch.save(model.state_dict(), 'best_model.pt')
    
    print(f"Epoch {epoch:03d}: Loss={loss:.4f}, Val={val_acc:.4f}, Test={test_acc:.4f}")
```

---

**Next:** [Scalability Techniques →](./02-scalability.md)
