# GNN Applications: Knowledge Graphs & Recommendation

Using GNNs for knowledge graph reasoning and recommendation systems.

## Knowledge Graph Completion

### The Task

Given a knowledge graph with missing edges, predict which triples (head, relation, tail) should exist.

```
Known: (Einstein, born_in, Germany)
       (Einstein, field, Physics)
       
Predict: (Einstein, awarded, Nobel_Prize) ?
```

### R-GCN for Knowledge Graphs

```python
from torch_geometric.nn import RGCNConv

class RGCN(torch.nn.Module):
    def __init__(self, num_entities, num_relations, hidden_dim):
        super().__init__()
        self.entity_embed = torch.nn.Embedding(num_entities, hidden_dim)
        self.conv1 = RGCNConv(hidden_dim, hidden_dim, num_relations)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations)
        self.relation_embed = torch.nn.Embedding(num_relations, hidden_dim)
    
    def forward(self, edge_index, edge_type):
        x = self.entity_embed.weight
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = self.conv2(x, edge_index, edge_type)
        return x
    
    def score(self, head, relation, tail):
        """DistMult scoring function."""
        h = self.entity_embed(head)
        r = self.relation_embed(relation)
        t = self.entity_embed(tail)
        return (h * r * t).sum(dim=-1)
```

### Training for Link Prediction

```python
def train_kg(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    
    # Sample negative triples
    neg_tail = torch.randint(0, num_entities, (data.edge_index.size(1),))
    
    # Positive scores
    pos_score = model.score(
        data.edge_index[0], 
        data.edge_type, 
        data.edge_index[1]
    )
    
    # Negative scores
    neg_score = model.score(
        data.edge_index[0],
        data.edge_type,
        neg_tail
    )
    
    # Margin loss
    loss = F.relu(1 - pos_score + neg_score).mean()
    loss.backward()
    optimizer.step()
    return loss.item()
```

## GNN-based Recommendation Systems

### Why GNNs for Recommendations?

Recommendation can be viewed as link prediction on a bipartite graph:
- **User nodes** ↔ **Item nodes**
- **Edges**: Past interactions (purchases, ratings, clicks)

### LightGCN

**Paper:** "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation" (2020)

Simplified GCN without feature transformations.

```python
class LightGCN(torch.nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, num_layers=3):
        super().__init__()
        self.num_users = num_users
        self.num_layers = num_layers
        
        # Learnable embeddings
        self.user_embed = torch.nn.Embedding(num_users, embedding_dim)
        self.item_embed = torch.nn.Embedding(num_items, embedding_dim)
        
        torch.nn.init.normal_(self.user_embed.weight, std=0.1)
        torch.nn.init.normal_(self.item_embed.weight, std=0.1)
    
    def forward(self, edge_index):
        # Combine user and item embeddings
        x = torch.cat([self.user_embed.weight, self.item_embed.weight])
        
        # Collect embeddings from all layers
        all_embeddings = [x]
        
        for _ in range(self.num_layers):
            # Simple aggregation (no weights, no activation)
            x = self.propagate(x, edge_index)
            all_embeddings.append(x)
        
        # Mean of all layers
        x = torch.stack(all_embeddings, dim=0).mean(dim=0)
        
        users, items = torch.split(x, [self.num_users, x.size(0) - self.num_users])
        return users, items
    
    def propagate(self, x, edge_index):
        """Simple symmetric normalization."""
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        out = torch.zeros_like(x)
        out.scatter_add_(0, col.unsqueeze(1).expand_as(x[row]), norm.unsqueeze(1) * x[row])
        return out
    
    def predict(self, user_ids, item_ids):
        users, items = self.forward(self.edge_index)
        user_emb = users[user_ids]
        item_emb = items[item_ids]
        return (user_emb * item_emb).sum(dim=-1)
```

### NGCF: Neural Graph Collaborative Filtering

```python
from torch_geometric.nn import MessagePassing

class NGCFConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.lin1 = torch.nn.Linear(in_channels, out_channels)
        self.lin2 = torch.nn.Linear(in_channels, out_channels)
    
    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)
    
    def message(self, x_i, x_j):
        return self.lin1(x_j) + self.lin2(x_i * x_j)


class NGCF(torch.nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, num_layers=3):
        super().__init__()
        self.num_users = num_users
        
        self.user_embed = torch.nn.Embedding(num_users, embedding_dim)
        self.item_embed = torch.nn.Embedding(num_items, embedding_dim)
        
        self.convs = torch.nn.ModuleList([
            NGCFConv(embedding_dim, embedding_dim) for _ in range(num_layers)
        ])
    
    def forward(self, edge_index):
        x = torch.cat([self.user_embed.weight, self.item_embed.weight])
        embeddings = [x]
        
        for conv in self.convs:
            x = F.leaky_relu(conv(x, edge_index))
            x = F.dropout(x, training=self.training)
            embeddings.append(x)
        
        x = torch.cat(embeddings, dim=-1)
        users, items = torch.split(x, [self.num_users, x.size(0) - self.num_users])
        return users, items
```

### Training Recommenders

```python
def bpr_loss(user_emb, pos_item_emb, neg_item_emb):
    """Bayesian Personalized Ranking loss."""
    pos_scores = (user_emb * pos_item_emb).sum(dim=-1)
    neg_scores = (user_emb * neg_item_emb).sum(dim=-1)
    return -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()

def train_recommender(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    
    users, items = model(data.edge_index)
    
    # Get positive pairs from training edges
    user_ids = data.train_edge_index[0]
    pos_item_ids = data.train_edge_index[1] - model.num_users
    
    # Sample negative items
    neg_item_ids = torch.randint(0, items.size(0), (user_ids.size(0),))
    
    user_emb = users[user_ids]
    pos_emb = items[pos_item_ids]
    neg_emb = items[neg_item_ids]
    
    loss = bpr_loss(user_emb, pos_emb, neg_emb)
    
    # L2 regularization
    reg_loss = 0.001 * (user_emb.norm(2) + pos_emb.norm(2) + neg_emb.norm(2))
    
    (loss + reg_loss).backward()
    optimizer.step()
    return loss.item()
```

## Evaluation Metrics

```python
def compute_metrics(model, data, k=10):
    """Compute Recall@K and NDCG@K."""
    users, items = model(data.edge_index)
    
    # For each user, get scores for all items
    scores = users @ items.t()
    
    # Get top-k items
    _, top_k = torch.topk(scores, k, dim=-1)
    
    # Compare with ground truth
    recall = []
    ndcg = []
    for user_id, user_topk in enumerate(top_k):
        true_items = data.test_items[user_id]
        hits = set(user_topk.tolist()) & true_items
        
        recall.append(len(hits) / min(len(true_items), k))
        
        # NDCG calculation
        dcg = sum([1/np.log2(i+2) for i, item in enumerate(user_topk) if item in true_items])
        idcg = sum([1/np.log2(i+2) for i in range(min(len(true_items), k))])
        ndcg.append(dcg / idcg if idcg > 0 else 0)
    
    return np.mean(recall), np.mean(ndcg)
```

## Datasets

| Dataset | Users | Items | Interactions | Domain |
|---------|-------|-------|--------------|--------|
| **MovieLens-1M** | 6,040 | 3,706 | 1M | Movies |
| **Amazon-Book** | 52,643 | 91,599 | 2.98M | Books |
| **Gowalla** | 29,858 | 40,981 | 1.03M | Check-ins |
| **Yelp** | 45,478 | 30,709 | 1.78M | Businesses |

```python
# Example: MovieLens
from torch_geometric.datasets import MovieLens

dataset = MovieLens('./data/ml-1m', 'ml-1m')
```

---

## References

- He, X., et al. (2020). "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." SIGIR. [arXiv](https://arxiv.org/abs/2002.02126)
- Wang, X., et al. (2019). "Neural Graph Collaborative Filtering." SIGIR. [arXiv](https://arxiv.org/abs/1905.08108)
- Schlichtkrull, M., et al. (2018). "Modeling Relational Data with Graph Convolutional Networks." ESWC.

---

**Congratulations!** You've completed the GNN learning materials. 🎉

Ready for **[Practical Projects →](../Projects/)**
