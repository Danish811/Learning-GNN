# 🎬 Project 6: GNN-based Recommendation System

**Difficulty:** ⭐⭐⭐ Advanced  
**Time:** 5-6 hours  
**Goal:** Build a movie recommendation system using graph neural networks

---

## 📖 Background

Recommendation systems are EVERYWHERE:
- 🎬 Netflix: "You might like..."
- 🛒 Amazon: "Customers also bought..."
- 🎵 Spotify: "Discover Weekly"

**Your Mission:** Build a GNN-based recommender using collaborative filtering on a user-item graph!

---

## 🧠 The Key Insight: Bipartite Graphs

Recommendations are naturally a **graph problem**:

```
User-Item Bipartite Graph:

    👤 Alice ─────┬───── 🎬 Inception
                  │
    👤 Bob ───────┼───── 🎬 Avatar
                  │
    👤 Carol ─────┴───── 🎬 Titanic
    
Edges = "User watched/rated movie"

Question: Should Alice watch Titanic?
GNN Answer: Check what similar users liked!
```

---

## 🚀 Task 1: Understanding the Problem

### Collaborative Filtering Explained:

```
Traditional ML:   User Features + Item Features → Prediction
GNN Approach:     User-Item Graph Structure → Embeddings → Prediction

Key insight: You don't NEED explicit features!
             Graph structure IS the feature!
```

### 🤔 Think About It:

**Q: Why is this called "collaborative" filtering?**

<details>
<summary>Answer</summary>

Users "collaborate" through their interactions!

If Alice and Bob both liked Inception and Avatar, they likely have similar taste. So Bob's love for Titanic suggests Alice might like it too.

The "collaboration" happens through the graph structure!
</details>

---

## 🚀 Task 2: Load and Prepare MovieLens Data

### About MovieLens 100K:
| Property | Value |
|----------|-------|
| Users | 943 |
| Movies | 1,682 |
| Ratings | 100,000 |
| Sparsity | 93.7% (most user-item pairs unknown) |

### 🧩 Your Task: Create User-Item Graph
```python
import pandas as pd
import torch
from torch_geometric.data import Data

# Load ratings (you'll need to download MovieLens 100K)
# From: https://grouplens.org/datasets/movielens/100k/

ratings = pd.read_csv('ml-100k/u.data', sep='\t',
                      names=['user_id', 'item_id', 'rating', 'timestamp'])

# Convert to 0-indexed
ratings['user_id'] = ratings['user_id'] - 1
ratings['item_id'] = ratings['item_id'] - 1

num_users = ???
num_items = ???

print(f"Users: {num_users}, Movies: {num_items}")
print(f"Ratings: {len(ratings)}")

# Create edge index
# Users: nodes 0 to num_users-1
# Items: nodes num_users to num_users+num_items-1

edge_user = torch.tensor(ratings['user_id'].values)
edge_item = torch.tensor(ratings['item_id'].values) + ???  # Shift item IDs!

# Undirected graph (user→item AND item→user)
edge_index = torch.stack([
    torch.cat([edge_user, edge_item]),
    torch.cat([edge_item, edge_user])
])

print(f"Edge index shape: {edge_index.shape}")
```

### 🤔 Key Question:

**Q: Why do we make the graph UNDIRECTED (add reverse edges)?**

<details>
<summary>Answer</summary>

Message passing goes both ways!

- User → Item: "What movies did this user like?"
- Item → User: "Who liked this movie?"

Both directions help build better embeddings!
</details>

<details>
<summary>✅ Full Solution</summary>

```python
import pandas as pd
import torch

ratings = pd.read_csv('ml-100k/u.data', sep='\t',
                      names=['user_id', 'item_id', 'rating', 'timestamp'])

ratings['user_id'] = ratings['user_id'] - 1
ratings['item_id'] = ratings['item_id'] - 1

num_users = ratings['user_id'].nunique()
num_items = ratings['item_id'].nunique()

print(f"Users: {num_users}, Movies: {num_items}")
print(f"Ratings: {len(ratings)}")

edge_user = torch.tensor(ratings['user_id'].values)
edge_item = torch.tensor(ratings['item_id'].values) + num_users

edge_index = torch.stack([
    torch.cat([edge_user, edge_item]),
    torch.cat([edge_item, edge_user])
])

print(f"Edge index shape: {edge_index.shape}")
```
</details>

---

## 🚀 Task 3: Build LightGCN Model

### Why LightGCN?

<details>
<summary>Answer</summary>

LightGCN (He et al., 2020) found that for recommendations:
- ❌ Non-linear activations HURT performance
- ❌ Feature transformations HURT performance
- ✅ Just light aggregation works BEST!

It's literally just averaging neighbor embeddings!
</details>

### 🧩 Implement LightGCN:
```python
import torch.nn as nn
from torch_geometric.utils import degree

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, num_layers=3):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers
        
        # Only learnable parameters: embeddings!
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)
        
        # Initialize
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
    
    def get_embeddings(self, edge_index):
        """Light graph convolution - NO activation, NO transformation!"""
        # Combine user and item embeddings
        x = torch.cat([
            self.user_embedding.weight,
            self.item_embedding.weight
        ])
        
        # Compute normalization (like GCN)
        row, col = edge_index
        deg = degree(col, x.size(0))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # Light propagation
        all_embeddings = [x]
        for _ in range(self.num_layers):
            # Message passing: just weighted sum!
            x_new = ???  # Aggregate neighbor embeddings with normalization
            all_embeddings.append(x_new)
            x = x_new
        
        # Final embedding = average of all layers
        final = torch.stack(all_embeddings).mean(dim=0)
        
        return final[:self.num_users], final[self.num_users:]
```

### 🤔 Design Questions:

**Q1: Why average embeddings across ALL layers (not just the last)?**

<details>
<summary>Answer</summary>

Different layers capture different information:
- Layer 0: Original preferences
- Layer 1: Direct neighbors' preferences
- Layer 2: 2-hop preferences

Averaging keeps ALL perspectives!
</details>

**Q2: Why NO activation functions?**

<details>
<summary>Answer</summary>

Experiments show activations add noise for recommendation! The model just needs to learn "who's similar to whom" — non-linearity doesn't help and actually hurts.
</details>

<details>
<summary>✅ Full Solution</summary>

```python
def get_embeddings(self, edge_index):
    x = torch.cat([
        self.user_embedding.weight,
        self.item_embedding.weight
    ])
    
    row, col = edge_index
    deg = degree(col, x.size(0), dtype=x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    
    all_embeddings = [x]
    for _ in range(self.num_layers):
        # Sparse message passing
        x_new = torch.zeros_like(x)
        x_new.scatter_add_(0, col.unsqueeze(1).expand_as(x[row]), 
                          norm.unsqueeze(1) * x[row])
        all_embeddings.append(x_new)
        x = x_new
    
    final = torch.stack(all_embeddings).mean(dim=0)
    return final[:self.num_users], final[self.num_users:]
```
</details>

---

## 🚀 Task 4: BPR Loss (Bayesian Personalized Ranking)

### Why Not Cross-Entropy?

<details>
<summary>Answer</summary>

We don't have explicit "negative" labels! A user not rating a movie means:
1. They haven't seen it yet, OR
2. They wouldn't like it

BPR solves this by saying: "Prefer rated items over unrated ones"
</details>

### BPR Formula:
```
For each user u:
  - pos_item = an item they DID rate
  - neg_item = an item they did NOT rate
  
  Loss = -log(sigmoid(score(u, pos) - score(u, neg)))
  
  Goal: Make score(u, pos) > score(u, neg)
```

### 🧩 Implement BPR Loss:
```python
def bpr_loss(user_emb, pos_item_emb, neg_item_emb):
    """
    Bayesian Personalized Ranking loss.
    
    Args:
        user_emb: User embeddings [batch]
        pos_item_emb: Positive item embeddings [batch]
        neg_item_emb: Negative item embeddings [batch]
    """
    # Score = dot product
    pos_scores = (user_emb * pos_item_emb).sum(dim=-1)
    neg_scores = (user_emb * neg_item_emb).sum(dim=-1)
    
    # BPR: prefer positive over negative
    loss = -torch.log(torch.sigmoid(??? - ???) + 1e-10).mean()
    return loss
```

<details>
<summary>✅ Full Solution</summary>

```python
def bpr_loss(user_emb, pos_item_emb, neg_item_emb):
    pos_scores = (user_emb * pos_item_emb).sum(dim=-1)
    neg_scores = (user_emb * neg_item_emb).sum(dim=-1)
    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()
    return loss
```
</details>

---

## 🚀 Task 5: Training and Evaluation

### 🧩 Implement Metrics:
```python
@torch.no_grad()
def evaluate(model, edge_index, test_users, test_items, k=10):
    model.eval()
    user_emb, item_emb = model.get_embeddings(edge_index)
    
    recalls = []
    for user, true_items in test_data:  # Group by user
        # Score all items for this user
        scores = (user_emb[user] @ item_emb.t())
        
        # Remove training items (they're already known!)
        scores[train_items[user]] = -float('inf')
        
        # Get top-K recommendations
        _, top_k = scores.topk(k)
        
        # Recall@K: what fraction of true items are in top-K?
        hits = len(set(top_k.tolist()) & set(true_items))
        recall = hits / min(len(true_items), k)
        recalls.append(recall)
    
    return sum(recalls) / len(recalls)
```

### 🎯 Expected Results:
- Recall@10 > 0.10 is good
- Recall@10 > 0.15 is excellent

---

## 🚀 Bonus: Generate Recommendations!

### 🧩 Your Task:
```python
def recommend_for_user(model, user_id, already_watched, top_k=10):
    """Generate recommendations for a specific user."""
    model.eval()
    user_emb, item_emb = model.get_embeddings(edge_index)
    
    # Score all items
    scores = ???
    
    # Exclude already watched
    for item in already_watched:
        scores[item] = -float('inf')
    
    # Get top-K
    top_k_scores, top_k_items = scores.topk(top_k)
    
    return list(zip(top_k_items.tolist(), top_k_scores.tolist()))

# Try it!
recs = recommend_for_user(model, user_id=0, already_watched=[...])
print("Top 10 recommendations for User 0:")
for movie_id, score in recs:
    print(f"  Movie {movie_id}: score = {score:.3f}")
```

---

## ✅ Project Checklist

- [ ] Understood user-item bipartite graphs
- [ ] Created edge index from ratings
- [ ] Built LightGCN model
- [ ] Implemented BPR loss
- [ ] Evaluated with Recall@K
- [ ] Generated actual recommendations!

---

## 🎓 What You Learned

| Concept | Key Insight |
|---------|-------------|
| **Bipartite graph** | Users and items as separate node types |
| **LightGCN** | No activations, no transformations — just aggregation |
| **BPR loss** | Prefer rated over unrated items |
| **Recall@K** | Measure recommendation quality |
| **Collaborative filtering** | Learn from the graph structure |

---

## 🎉 Congratulations!

You've completed all 6 projects! You now have hands-on experience with:

1. ✅ Node classification
2. ✅ Link prediction
3. ✅ Graph classification
4. ✅ Molecular property prediction
5. ✅ Large-scale social networks
6. ✅ Recommendation systems

**Ready for the capstone?** [Capstone Project →](../Capstone-Molecular-Property-Prediction/)
