# GNN Datasets Guide

A comprehensive guide to datasets commonly used for GNN research and projects.

## Quick Reference

| Dataset | Task | Nodes | Edges | Classes | Domain |
|---------|------|-------|-------|---------|--------|
| Cora | Node | 2,708 | 10,556 | 7 | Citation |
| CiteSeer | Node | 3,327 | 9,104 | 6 | Citation |
| PubMed | Node | 19,717 | 88,648 | 3 | Citation |
| MUTAG | Graph | 188 graphs | ~18/graph | 2 | Molecules |
| PROTEINS | Graph | 1,113 graphs | ~39/graph | 2 | Biology |
| QM9 | Graph | 134K molecules | - | Regression | Chemistry |

---

## Citation Networks

Best for getting started with node classification.

### Cora

| Property | Value |
|----------|-------|
| Nodes | 2,708 papers |
| Edges | 10,556 citations |
| Features | 1,433 (bag of words) |
| Classes | 7 topics |
| Train/Val/Test | 140/500/1000 |

```python
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='./data', name='Cora')
data = dataset[0]

print(f"Nodes: {data.num_nodes}")
print(f"Edges: {data.num_edges}")
print(f"Features: {data.num_node_features}")
print(f"Classes: {dataset.num_classes}")
```

### CiteSeer

Similar to Cora but with 6 classes. Good for comparison experiments.

```python
dataset = Planetoid(root='./data', name='CiteSeer')
```

### PubMed

Larger citation network with diabetes-related papers.

```python
dataset = Planetoid(root='./data', name='PubMed')
# 19,717 nodes, 88,648 edges, 3 classes
```

---

## Social Networks

### Reddit

Large-scale social network for inductive learning.

| Property | Value |
|----------|-------|
| Nodes | 232,965 posts |
| Edges | 114,615,892 |
| Classes | 41 subreddits |

```python
from torch_geometric.datasets import Reddit

dataset = Reddit(root='./data/Reddit')
# Warning: Large dataset, needs sufficient RAM
```

### Twitch

Gamer social networks from different regions.

```python
from torch_geometric.datasets import Twitch

dataset = Twitch(root='./data/Twitch', name='EN')  # English
# Also: DE, ES, FR, PT, RU
```

---

## Molecular Datasets

### QM9

Quantum chemistry properties of small molecules.

| Property | Value |
|----------|-------|
| Molecules | 134,000 |
| Atoms/molecule | ≤9 heavy atoms |
| Targets | 19 quantum properties |

```python
from torch_geometric.datasets import QM9

dataset = QM9(root='./data/QM9')
data = dataset[0]

print(f"Atoms: {data.x.shape[0]}")
print(f"Bonds: {data.edge_index.shape[1] // 2}")
print(f"Target properties: {data.y.shape[1]}")  # 19
```

### ZINC

Subset of ZINC database for graph regression.

```python
from torch_geometric.datasets import ZINC

train_dataset = ZINC(root='./data/ZINC', subset=True, split='train')
val_dataset = ZINC(root='./data/ZINC', subset=True, split='val')
test_dataset = ZINC(root='./data/ZINC', subset=True, split='test')
```

### MoleculeNet

Collection of molecular datasets for various tasks.

```python
from torch_geometric.datasets import MoleculeNet

# HIV activity prediction
hiv = MoleculeNet(root='./data', name='HIV')

# Blood-brain barrier penetration
bbbp = MoleculeNet(root='./data', name='BBBP')

# Toxicity
tox21 = MoleculeNet(root='./data', name='Tox21')
```

---

## TU Datasets (Graph Classification)

Collection of benchmark datasets for graph-level tasks.

### MUTAG

Mutagenicity prediction of chemical compounds.

```python
from torch_geometric.datasets import TUDataset

dataset = TUDataset(root='./data', name='MUTAG')
print(f"Graphs: {len(dataset)}")  # 188
print(f"Classes: {dataset.num_classes}")  # 2
```

### PROTEINS

Protein structure classification.

```python
dataset = TUDataset(root='./data', name='PROTEINS')
# 1,113 graphs, 2 classes (enzyme or not)
```

### Other TU Datasets

```python
# Social networks
imdb = TUDataset(root='./data', name='IMDB-BINARY')
reddit = TUDataset(root='./data', name='REDDIT-BINARY')

# Chemical compounds
ptc = TUDataset(root='./data', name='PTC_MR')
nci1 = TUDataset(root='./data', name='NCI1')
```

---

## Knowledge Graphs

### FB15k-237

Subset of Freebase for link prediction.

```python
from torch_geometric.datasets import FB15k_237

dataset = FB15k_237(root='./data/FB15k-237')
data = dataset[0]

print(f"Entities: {data.num_nodes}")  # 14,541
print(f"Relations: {data.edge_type.max().item() + 1}")  # 237
print(f"Triplets: {data.edge_index.shape[1]}")
```

### WordNet18RR

Lexical knowledge graph.

```python
from torch_geometric.datasets import WordNet18RR

dataset = WordNet18RR(root='./data/WordNet18RR')
```

---

## Heterogeneous Graphs

### OGB (Open Graph Benchmark)

Large-scale benchmarks with standardized splits.

```python
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.linkproppred import PygLinkPropPredDataset

# Node property prediction
ogbn_arxiv = PygNodePropPredDataset(name='ogbn-arxiv')
ogbn_products = PygNodePropPredDataset(name='ogbn-products')

# Graph property prediction
ogbg_molhiv = PygGraphPropPredDataset(name='ogbg-molhiv')

# Link prediction
ogbl_collab = PygLinkPropPredDataset(name='ogbl-collab')
```

### IMDB

Heterogeneous movie database.

```python
from torch_geometric.datasets import IMDB

dataset = IMDB(root='./data/IMDB')
# Node types: movie, director, actor
# Edge types: movie-director, movie-actor
```

---

## Creating Custom Datasets

### From NetworkX

```python
import networkx as nx
from torch_geometric.utils import from_networkx

# Create NetworkX graph
G = nx.karate_club_graph()

# Add node features
for node in G.nodes():
    G.nodes[node]['x'] = [node]  # Simple feature

# Convert to PyG
data = from_networkx(G)
```

### From CSV/Edge List

```python
import pandas as pd
from torch_geometric.data import Data

# Load edges
edges_df = pd.read_csv('edges.csv')
edge_index = torch.tensor([
    edges_df['source'].values,
    edges_df['target'].values
], dtype=torch.long)

# Load node features
nodes_df = pd.read_csv('nodes.csv')
x = torch.tensor(nodes_df.drop('id', axis=1).values, dtype=torch.float)

# Create data object
data = Data(x=x, edge_index=edge_index)
```

### Custom Dataset Class

```python
from torch_geometric.data import Dataset, Data
import os

class MyDataset(Dataset):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)
    
    @property
    def raw_file_names(self):
        return ['edges.csv', 'nodes.csv']
    
    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def download(self):
        pass  # Implement if needed
    
    def process(self):
        # Load and process raw data
        data = ...  # Your processing logic
        torch.save(data, self.processed_paths[0])
    
    def len(self):
        return 1
    
    def get(self, idx):
        return torch.load(self.processed_paths[0])
```

---

## Dataset Selection Guide

| Task | Beginner | Intermediate | Advanced |
|------|----------|--------------|----------|
| Node Classification | Cora | PubMed | ogbn-arxiv |
| Link Prediction | Cora | FB15k-237 | ogbl-collab |
| Graph Classification | MUTAG | PROTEINS | ogbg-molhiv |
| Molecular | QM9 | ZINC | MoleculeNet |
| Social | Twitch | Reddit | ogbn-products |

---

## References

- [PyTorch Geometric Datasets](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html)
- [Open Graph Benchmark](https://ogb.stanford.edu/)
- [TU Datasets](https://chrsmrrs.github.io/datasets/)
- [MoleculeNet](https://moleculenet.org/)
