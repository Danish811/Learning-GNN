# GNN Applications: Social Networks

Applying Graph Neural Networks to social network analysis tasks.

## Why GNNs for Social Networks?

Social networks are inherently graphs:
- **Nodes**: Users, posts, groups
- **Edges**: Friendships, follows, interactions
- **Features**: Demographics, interests, activity

| Task | Description | Example |
|------|-------------|---------|
| **Node Classification** | Categorize users | Bot detection, influence prediction |
| **Link Prediction** | Predict connections | Friend recommendations |
| **Community Detection** | Find groups | Interest groups, echo chambers |
| **Influence Propagation** | Track spread | Viral content, misinformation |

## User Classification (Bot Detection)

### Dataset: Twitter Bot Detection

```python
import torch
from torch_geometric.data import Data

# Typical features for Twitter users
user_features = [
    'followers_count',
    'following_count',
    'tweet_count',
    'account_age_days',
    'avg_tweets_per_day',
    'has_profile_image',
    'verified',
    # ... more features
]

# Build graph from follow relationships
# edge_index[0] = followers, edge_index[1] = followed
```

### Model

```python
from torch_geometric.nn import SAGEConv

class BotDetector(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.classifier = torch.nn.Linear(hidden_channels, 2)  # Bot or Human
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return self.classifier(x)
```

## Friend Recommendation (Link Prediction)

```python
from torch_geometric.nn import GAE

class FriendRecommender(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            SAGEConv(in_channels, hidden_channels),
            torch.nn.ReLU(),
            SAGEConv(hidden_channels, out_channels)
        )
    
    def encode(self, x, edge_index):
        for layer in self.encoder:
            if hasattr(layer, 'forward') and 'edge_index' in layer.forward.__code__.co_varnames:
                x = layer(x, edge_index)
            else:
                x = layer(x)
        return x
    
    def decode(self, z, edge_index):
        """Inner product decoder for link prediction."""
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
    
    def forward(self, x, edge_index, pos_edge, neg_edge):
        z = self.encode(x, edge_index)
        pos_score = self.decode(z, pos_edge)
        neg_score = self.decode(z, neg_edge)
        return pos_score, neg_score
```

## Community Detection

Unsupervised GNN for finding communities.

```python
from torch_geometric.nn import GCNConv
from sklearn.cluster import KMeans

class CommunityGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_communities):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.num_communities = num_communities
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x
    
    def detect_communities(self, x, edge_index):
        """Get embeddings and cluster."""
        embeddings = self.forward(x, edge_index).detach().numpy()
        kmeans = KMeans(n_clusters=self.num_communities)
        return kmeans.fit_predict(embeddings)

# Self-supervised training with modularity loss
def modularity_loss(embeddings, edge_index, num_nodes):
    """Encourage embeddings to reflect community structure."""
    adj = torch.zeros(num_nodes, num_nodes)
    adj[edge_index[0], edge_index[1]] = 1
    
    degree = adj.sum(dim=1)
    m = edge_index.size(1) / 2
    
    # Modularity matrix B = A - (d_i * d_j) / (2m)
    B = adj - torch.outer(degree, degree) / (2 * m)
    
    # Soft community assignment
    S = F.softmax(embeddings, dim=1)
    
    # Modularity Q = trace(S^T B S)
    Q = torch.trace(S.t() @ B @ S) / (2 * m)
    
    return -Q  # Maximize modularity
```

## Influence Propagation

Modeling how information spreads through a network.

```python
class InfluenceGNN(torch.nn.Module):
    """Predict influence cascade."""
    
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.time_pred = torch.nn.Linear(hidden_channels, 1)  # Time to activate
        self.prob_pred = torch.nn.Linear(hidden_channels, 1)  # Probability
    
    def forward(self, x, edge_index, activated_mask):
        """
        x: node features
        activated_mask: which nodes are already activated
        """
        # Add activation status to features
        x = torch.cat([x, activated_mask.float().unsqueeze(1)], dim=1)
        
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        
        activation_prob = torch.sigmoid(self.prob_pred(x))
        activation_time = F.softplus(self.time_pred(x))
        
        return activation_prob, activation_time
```

## Real-World Datasets

| Dataset | Nodes | Edges | Task |
|---------|-------|-------|------|
| **Twitch** | 168K | 6.8M | Node classification |
| **Reddit** | 233K | 114M | Node classification |
| **Facebook** | 4K | 88K | Link prediction |
| **Twitter** | Varies | Varies | Bot detection |

```python
from torch_geometric.datasets import Twitch, Reddit

# Twitch gamer social network
dataset = Twitch(root='./data', name='EN')

# Reddit post network (large)
dataset = Reddit(root='./data/Reddit')
```

---

## References

- Zhang, M., & Chen, Y. (2018). "Link Prediction Based on Graph Neural Networks." NeurIPS.
- Fan, W., et al. (2019). "Graph Neural Networks for Social Recommendation." WWW.

---

**Next:** [Molecular Graphs →](./02-molecular-graphs.md)
