# 🧪 Capstone Project: Molecular Property Prediction for Drug Discovery

**A comprehensive GNN project applying everything you've learned to real-world drug discovery.**

---

## 📋 Project Overview

| Property | Details |
|----------|---------|
| **Difficulty** | ⭐⭐⭐ Advanced |
| **Duration** | 2-3 weeks |
| **Domain** | Computational Chemistry / Drug Discovery |
| **Task Type** | Graph Regression & Classification |
| **Key Skills** | GNN architectures, molecular graphs, multi-task learning |

### Objective

Build a production-quality GNN system that predicts molecular properties (toxicity, solubility, drug-likeness) from chemical structures—a critical task in drug discovery pipelines.

### Why This Project?

- **Real-world impact**: Drug discovery is a $1.5T industry; ML can reduce costs by 70%
- **Industry-relevant**: Used at Pfizer, Novartis, and startups like Recursion
- **Comprehensive**: Combines graph theory, chemistry, and deep learning
- **Portfolio-worthy**: Demonstrates practical ML engineering skills

---

## 📊 Datasets

### Primary: MoleculeNet Benchmarks

| Dataset | Task | Molecules | Metric | Description |
|---------|------|-----------|--------|-------------|
| **Tox21** | Classification | 8,014 | ROC-AUC | 12 toxicity assays |
| **BBBP** | Classification | 2,039 | ROC-AUC | Blood-brain barrier penetration |
| **ESOL** | Regression | 1,128 | RMSE | Water solubility |
| **Lipophilicity** | Regression | 4,200 | RMSE | Lipophilicity (logD) |
| **FreeSolv** | Regression | 642 | RMSE | Solvation free energy |

### Secondary: QM9 (Quantum Chemistry)

| Property | Description |
|----------|-------------|
| Molecules | 134,000 small organic molecules |
| Features | 19 quantum chemical properties |
| Task | Multi-target regression |

### Data Source

```python
# PyTorch Geometric
from torch_geometric.datasets import MoleculeNet, QM9

# MoleculeNet datasets
tox21 = MoleculeNet(root='./data', name='Tox21')
bbbp = MoleculeNet(root='./data', name='BBBP')
esol = MoleculeNet(root='./data', name='ESOL')

# QM9
qm9 = QM9(root='./data/qm9')
```

---

## 🏗️ Project Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     MOLECULAR PROPERTY PREDICTOR                 │
├─────────────────────────────────────────────────────────────────┤
│  INPUT: SMILES String                                           │
│    "CC(=O)OC1=CC=CC=C1C(=O)O" (Aspirin)                        │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────┐                │
│  │         MOLECULAR GRAPH CONSTRUCTION         │                │
│  │  • Atoms → Nodes (with features)            │                │
│  │  • Bonds → Edges (with features)            │                │
│  │  • RDKit for parsing                        │                │
│  └─────────────────────────────────────────────┘                │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────┐                │
│  │            GNN ENCODER                       │                │
│  │  • Message Passing Layers (GIN/MPNN)        │                │
│  │  • Edge Features Integration                 │                │
│  │  • Virtual Node (optional)                   │                │
│  └─────────────────────────────────────────────┘                │
│                              │                                   │
│  ┌─────────────────────────────────────────────┐                │
│  │         GRAPH POOLING                        │                │
│  │  • Global Mean/Sum Pooling                   │                │
│  │  • Set2Set / Attention Pooling (advanced)   │                │
│  └─────────────────────────────────────────────┘                │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────┐                │
│  │       PREDICTION HEAD                        │                │
│  │  • Classification: Toxicity (12 tasks)      │                │
│  │  • Regression: Solubility, Lipophilicity    │                │
│  └─────────────────────────────────────────────┘                │
│                              │                                   │
│                              ▼                                   │
│  OUTPUT: Property Predictions + Confidence                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
MolecularPropertyPredictor/
├── README.md
├── requirements.txt
├── setup.py
│
├── data/
│   ├── download_data.py
│   └── data_statistics.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── molecule_dataset.py      # Custom dataset class
│   │   ├── featurizer.py            # Atom/bond feature extraction
│   │   └── transforms.py            # Data augmentation
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── gcn.py                   # GCN baseline
│   │   ├── gin.py                   # Graph Isomorphism Network
│   │   ├── mpnn.py                  # Message Passing NN
│   │   ├── attentivefp.py           # Attentive FP model
│   │   └── layers.py                # Custom layers
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py               # Training loop
│   │   ├── metrics.py               # Evaluation metrics
│   │   └── schedulers.py            # LR schedulers
│   │
│   └── utils/
│       ├── __init__.py
│       ├── chemistry.py             # Chemistry utilities
│       └── visualization.py         # Molecule visualization
│
├── configs/
│   ├── tox21.yaml
│   ├── esol.yaml
│   └── qm9.yaml
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_models.ipynb
│   ├── 03_advanced_models.ipynb
│   ├── 04_interpretability.ipynb
│   └── 05_deployment_demo.ipynb
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
└── tests/
    ├── test_data.py
    └── test_models.py
```

---

## 🚀 Implementation Phases

### Phase 1: Setup & Data (Days 1-3) ✅

**Goals:**
- [ ] Set up development environment
- [ ] Download and explore datasets
- [ ] Implement molecular graph construction
- [ ] Create data loading pipeline

**Key Code:**
```python
# featurizer.py - Extract atom and bond features
from rdkit import Chem

ATOM_FEATURES = {
    'atomic_num': list(range(1, 119)),      # Periodic table
    'chirality': ['CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW'],
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-2, -1, 0, 1, 2],
    'hybridization': ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2'],
    'is_aromatic': [False, True],
    'num_hs': [0, 1, 2, 3, 4],
}

def get_atom_features(atom):
    """Convert RDKit atom to feature vector."""
    return [
        safe_index(ATOM_FEATURES['atomic_num'], atom.GetAtomicNum()),
        safe_index(ATOM_FEATURES['degree'], atom.GetTotalDegree()),
        safe_index(ATOM_FEATURES['formal_charge'], atom.GetFormalCharge()),
        safe_index(ATOM_FEATURES['hybridization'], str(atom.GetHybridization())),
        1 if atom.GetIsAromatic() else 0,
        safe_index(ATOM_FEATURES['num_hs'], atom.GetTotalNumHs()),
    ]
```

### Phase 2: Baseline Models (Days 4-7)

**Goals:**
- [ ] Implement GCN baseline
- [ ] Implement GIN model
- [ ] Set up training infrastructure
- [ ] Achieve baseline performance

**Key Code:**
```python
# gin.py - Graph Isomorphism Network
class GINMolecule(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_dim, 
                 num_layers, num_tasks, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        
        # Node embedding
        self.node_encoder = AtomEncoder(hidden_dim)
        self.edge_encoder = BondEncoder(hidden_dim)
        
        # GIN layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        for _ in range(num_layers):
            mlp = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, 2 * hidden_dim),
                torch.nn.BatchNorm1d(2 * hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(2 * hidden_dim, hidden_dim)
            )
            self.convs.append(GINEConv(mlp))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        
        # Prediction head
        self.pred_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, num_tasks)
        )
    
    def forward(self, data):
        x = self.node_encoder(data.x)
        edge_attr = self.edge_encoder(data.edge_attr)
        
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, data.edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
        
        # Global pooling
        x = global_add_pool(x, data.batch)
        
        return self.pred_head(x)
```

### Phase 3: Advanced Models (Days 8-11)

**Goals:**
- [ ] Implement AttentiveFP (attention-based)
- [ ] Add virtual node for global information
- [ ] Implement multi-task learning
- [ ] Hyperparameter tuning

**Key Code:**
```python
# Virtual Node enhancement
class VirtualNodeGNN(torch.nn.Module):
    def __init__(self, base_gnn, hidden_dim):
        super().__init__()
        self.gnn = base_gnn
        self.virtual_node_embedding = torch.nn.Embedding(1, hidden_dim)
        self.virtual_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 2 * hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * hidden_dim, hidden_dim)
        )
    
    def forward(self, data):
        batch_size = data.batch.max().item() + 1
        virtual_node = self.virtual_node_embedding.weight.expand(batch_size, -1)
        
        for layer in self.gnn.convs:
            # Aggregate to virtual node
            x_pooled = global_add_pool(data.x, data.batch)
            virtual_node = virtual_node + self.virtual_mlp(x_pooled)
            
            # Broadcast back to nodes
            data.x = data.x + virtual_node[data.batch]
            
            # Regular GNN layer
            data.x = layer(data.x, data.edge_index)
        
        return self.gnn.pred_head(global_add_pool(data.x, data.batch))
```

### Phase 4: Evaluation & Analysis (Days 12-14)

**Goals:**
- [ ] Comprehensive evaluation on all benchmarks
- [ ] Model interpretability (attention visualization)
- [ ] Error analysis
- [ ] Comparison with literature baselines

**Expected Performance:**

| Dataset | Metric | Our Model | Literature SOTA |
|---------|--------|-----------|-----------------|
| Tox21 | ROC-AUC | ~0.82 | 0.85 |
| BBBP | ROC-AUC | ~0.92 | 0.93 |
| ESOL | RMSE | ~0.55 | 0.50 |
| Lipophilicity | RMSE | ~0.58 | 0.55 |

### Phase 5: Deployment (Days 15-17)

**Goals:**
- [ ] Create inference pipeline
- [ ] Build simple web demo (Gradio/Streamlit)
- [ ] Docker containerization
- [ ] Documentation and README

---

## 📚 References & Tutorials

### Must-Read Papers
1. Gilmer et al. (2017) - "Neural Message Passing for Quantum Chemistry" [arXiv](https://arxiv.org/abs/1704.01212)
2. Xiong et al. (2019) - "AttentiveFP" [J. Med. Chem.](https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959)
3. Hu et al. (2020) - "Strategies for Pre-training Graph Neural Networks" [arXiv](https://arxiv.org/abs/1905.12265)

### Tutorials
- [TeachOpenCADD Tutorial T025](https://projects.volkamerlab.org/) - GNN for molecular property prediction
- [DeepChem Documentation](https://deepchem.io/) - Drug discovery ML library
- [PyG Molecular Tutorial](https://pytorch-geometric.readthedocs.io/en/latest/notes/colabs.html)

### Datasets
- [MoleculeNet Benchmark](https://moleculenet.org/)
- [QM9 Database](http://quantum-machine.org/datasets/)
- [ChEMBL Database](https://www.ebi.ac.uk/chembl/)

---

## ✅ Success Criteria

| Criterion | Target |
|-----------|--------|
| Tox21 ROC-AUC | > 0.80 |
| BBBP ROC-AUC | > 0.90 |
| Training reproducibility | Seeds documented |
| Code quality | Modular, tested |
| Documentation | README, notebooks |
| Demo | Working web app |

---

## 🔧 Tech Stack

| Component | Technology |
|-----------|------------|
| Framework | PyTorch + PyTorch Geometric |
| Chemistry | RDKit, DeepChem |
| Experiment Tracking | Weights & Biases / MLflow |
| Visualization | Matplotlib, RDKit drawing |
| Deployment | Gradio / Streamlit + Docker |

---

## 📝 Getting Started

```bash
# Clone and setup
git clone <your-repo>
cd MolecularPropertyPredictor

# Create environment
conda create -n mol-gnn python=3.10
conda activate mol-gnn

# Install dependencies
pip install torch torch-geometric rdkit-pypi deepchem
pip install wandb gradio jupyter

# Download data
python data/download_data.py

# Run training
python scripts/train.py --config configs/tox21.yaml

# Launch demo
python scripts/demo.py
```

---

**Good luck with your capstone project! 🚀**
