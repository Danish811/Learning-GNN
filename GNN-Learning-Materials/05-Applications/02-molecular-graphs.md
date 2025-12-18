# GNN Applications: Molecular Graphs

Applying GNNs to molecules for drug discovery and materials science.

## Why GNNs for Molecules?

Molecules are naturally graphs:
- **Nodes**: Atoms (C, N, O, H, ...)
- **Edges**: Chemical bonds (single, double, triple, aromatic)
- **Features**: Atomic number, charge, hybridization, etc.

| Task | Description | Application |
|------|-------------|-------------|
| **Property Prediction** | Predict molecular properties | Drug screening |
| **Toxicity Prediction** | Predict harmful effects | Safety assessment |
| **Solubility** | Predict water solubility | Formulation |
| **Binding Affinity** | Drug-target interaction | Drug design |

## Molecular Featurization

### Atom Features

```python
from rdkit import Chem

def get_atom_features(atom):
    """Extract features from RDKit atom."""
    return [
        atom.GetAtomicNum(),            # Atomic number
        atom.GetDegree(),               # Number of bonds
        atom.GetFormalCharge(),         # Formal charge
        int(atom.GetHybridization()),   # sp, sp2, sp3
        int(atom.GetIsAromatic()),      # Aromaticity
        atom.GetTotalNumHs(),           # Attached hydrogens
        atom.GetNumRadicalElectrons(),  # Radical electrons
        int(atom.IsInRing()),           # In ring
    ]

def mol_to_graph(smiles):
    """Convert SMILES to PyG Data object."""
    mol = Chem.MolFromSmiles(smiles)
    
    # Node features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))
    x = torch.tensor(atom_features, dtype=torch.float)
    
    # Edge indices and features
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.extend([[i, j], [j, i]])  # Undirected
        bond_type = bond.GetBondTypeAsDouble()
        edge_attr.extend([[bond_type], [bond_type]])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# Example
data = mol_to_graph("CC(=O)OC1=CC=CC=C1C(=O)O")  # Aspirin
```

### Using DeepChem

```python
from deepchem.feat import MolGraphConvFeaturizer

featurizer = MolGraphConvFeaturizer(use_edges=True)
features = featurizer.featurize(["CCO", "CC(=O)O"])  # Ethanol, Acetic acid
```

## Message Passing Neural Network (MPNN)

The original architecture for molecular property prediction.

```python
from torch_geometric.nn import NNConv, global_add_pool

class MPNN(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, out_dim, num_layers=3):
        super().__init__()
        self.node_embed = torch.nn.Linear(node_dim, hidden_dim)
        
        # Edge network: maps edge features to weight matrix
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            edge_nn = torch.nn.Sequential(
                torch.nn.Linear(edge_dim, hidden_dim * hidden_dim),
            )
            self.convs.append(NNConv(hidden_dim, hidden_dim, edge_nn))
        
        self.output = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_embed(x)
        
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_attr))
        
        # Global pooling
        x = global_add_pool(x, batch)
        
        return self.output(x)
```

## AttentiveFP

State-of-the-art for molecular property prediction.

```python
from torch_geometric.nn import AttentiveFP

# Built-in implementation
model = AttentiveFP(
    in_channels=39,           # Atom features
    hidden_channels=64,
    out_channels=1,           # Property to predict
    edge_dim=10,              # Bond features
    num_layers=3,
    num_timesteps=2,
    dropout=0.2
)

# Forward pass
out = model(data.x, data.edge_index, data.edge_attr, data.batch)
```

## Complete Training Example: Solubility Prediction

```python
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader

# Load ESOL dataset (water solubility)
dataset = MoleculeNet(root='./data', name='ESOL')

print(f"Dataset size: {len(dataset)}")
print(f"Num features: {dataset.num_node_features}")
print(f"Num edge features: {dataset.num_edge_features}")

# Split
train_dataset = dataset[:900]
val_dataset = dataset[900:1000]
test_dataset = dataset[1000:]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# Model
class SolubilityPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gnn = AttentiveFP(
            in_channels=dataset.num_node_features,
            hidden_channels=64,
            out_channels=1,
            edge_dim=dataset.num_edge_features,
            num_layers=3,
            num_timesteps=2
        )
    
    def forward(self, batch):
        return self.gnn(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

model = SolubilityPredictor()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

# Training
for epoch in range(100):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        pred = model(batch).squeeze()
        loss = criterion(pred, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    
    # Validation
    model.eval()
    val_mse = 0
    for batch in val_loader:
        pred = model(batch).squeeze()
        val_mse += criterion(pred, batch.y).item() * batch.num_graphs
    
    if epoch % 10 == 0:
        rmse = (total_loss / len(train_dataset)) ** 0.5
        val_rmse = (val_mse / len(val_dataset)) ** 0.5
        print(f"Epoch {epoch}: Train RMSE = {rmse:.4f}, Val RMSE = {val_rmse:.4f}")
```

## Multi-Task Learning (Tox21)

```python
class MultiTaskMoleculeGNN(torch.nn.Module):
    """Predict multiple properties simultaneously."""
    
    def __init__(self, num_tasks=12):
        super().__init__()
        self.backbone = AttentiveFP(
            in_channels=9,
            hidden_channels=64,
            out_channels=64,
            edge_dim=3,
            num_layers=3,
            num_timesteps=2
        )
        self.heads = torch.nn.ModuleList([
            torch.nn.Linear(64, 1) for _ in range(num_tasks)
        ])
    
    def forward(self, batch):
        x = self.backbone(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        return torch.cat([head(x) for head in self.heads], dim=1)

# Handle missing labels in Tox21
def masked_bce_loss(pred, target):
    mask = ~torch.isnan(target)
    return F.binary_cross_entropy_with_logits(pred[mask], target[mask])
```

## Datasets

| Dataset | Molecules | Task | Metric |
|---------|-----------|------|--------|
| **ESOL** | 1,128 | Solubility | RMSE |
| **FreeSolv** | 642 | Solvation energy | RMSE |
| **Lipophilicity** | 4,200 | LogD | RMSE |
| **Tox21** | 8,014 | 12 toxicity assays | ROC-AUC |
| **BBBP** | 2,039 | Blood-brain barrier | ROC-AUC |
| **QM9** | 134K | 19 quantum properties | MAE |

---

## References

- Gilmer, J., et al. (2017). "Neural Message Passing for Quantum Chemistry." ICML.
- Xiong, Z., et al. (2020). "Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph Attention Mechanism." J. Med. Chem.

---

**Next:** [Knowledge Graphs →](./03-knowledge-graphs.md)
