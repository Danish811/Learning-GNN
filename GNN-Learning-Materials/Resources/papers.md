# Must-Read GNN Papers

A curated collection of essential papers organized by topic. Start with foundational papers, then explore based on your interests.

## 🏛️ Foundational Papers

### The Core Architectures

| Year | Paper | Authors | Key Contribution |
|------|-------|---------|------------------|
| 2017 | [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) | Kipf & Welling | GCN - Spectral convolution approximation |
| 2017 | [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216) | Hamilton et al. | GraphSAGE - Inductive learning + sampling |
| 2018 | [Graph Attention Networks](https://arxiv.org/abs/1710.10903) | Veličković et al. | GAT - Attention mechanism for graphs |
| 2017 | [Neural Message Passing for Quantum Chemistry](https://arxiv.org/abs/1704.01212) | Gilmer et al. | MPNN - Unified message passing framework |

### Theoretical Foundations

| Year | Paper | Key Insight |
|------|-------|-------------|
| 2019 | [How Powerful are Graph Neural Networks?](https://arxiv.org/abs/1810.00826) | GNN expressiveness = WL test (GIN) |
| 2020 | [A Survey on The Expressive Power of Graph Neural Networks](https://arxiv.org/abs/2003.04078) | Comprehensive expressiveness analysis |
| 2021 | [How Attentive are Graph Attention Networks?](https://arxiv.org/abs/2105.14491) | Static vs dynamic attention (GATv2) |

## 📊 Deep Architectures & Over-Smoothing

| Year | Paper | Solution/Contribution |
|------|-------|----------------------|
| 2018 | [Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning](https://arxiv.org/abs/1801.07606) | Analysis of over-smoothing |
| 2019 | [DeepGCNs: Can GCNs Go as Deep as CNNs?](https://arxiv.org/abs/1904.03751) | Residual + dense connections |
| 2020 | [DropEdge: Towards Deep Graph Convolutional Networks on Node Classification](https://arxiv.org/abs/1907.10903) | Random edge dropping |
| 2020 | [Simple and Deep Graph Convolutional Networks](https://arxiv.org/abs/2007.02133) | GCNII with initial residual |

## 🔀 Heterogeneous & Knowledge Graphs

| Year | Paper | Focus |
|------|-------|-------|
| 2019 | [Heterogeneous Graph Attention Network](https://arxiv.org/abs/1903.07293) | HAN - Hierarchical attention |
| 2020 | [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103) | R-GCN for knowledge graphs |
| 2021 | [Heterogeneous Graph Transformer](https://arxiv.org/abs/2003.01332) | HGT - Transformer for hetero graphs |

## ⏱️ Temporal & Dynamic Graphs

| Year | Paper | Focus |
|------|-------|-------|
| 2018 | [Spatial Temporal Graph Convolutional Networks](https://arxiv.org/abs/1709.04875) | ST-GCN for traffic/skeleton |
| 2020 | [EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs](https://arxiv.org/abs/1902.10191) | Time-evolving GCN |
| 2020 | [Temporal Graph Networks for Deep Learning on Dynamic Graphs](https://arxiv.org/abs/2006.10637) | TGN - Memory-based temporal |

## 🧬 Applications: Molecules & Drug Discovery

| Year | Paper | Application |
|------|-------|-------------|
| 2017 | [Neural Message Passing for Quantum Chemistry](https://arxiv.org/abs/1704.01212) | Molecular property prediction |
| 2020 | [Strategies for Pre-training Graph Neural Networks](https://arxiv.org/abs/1905.12265) | Pre-training for molecules |
| 2020 | [MoleculeNet: A Benchmark for Molecular Machine Learning](https://arxiv.org/abs/1703.00564) | Benchmark datasets |

## 🎯 Applications: Recommendations

| Year | Paper | System |
|------|-------|--------|
| 2018 | [Graph Convolutional Neural Networks for Web-Scale Recommender Systems](https://arxiv.org/abs/1806.01973) | PinSage (Pinterest) |
| 2019 | [LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation](https://arxiv.org/abs/2002.02126) | Simplified rec GCN |
| 2019 | [KGAT: Knowledge Graph Attention Network for Recommendation](https://arxiv.org/abs/1905.07854) | Knowledge-enhanced rec |

## 🔬 Scalability

| Year | Paper | Technique |
|------|-------|-----------|
| 2019 | [Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks](https://arxiv.org/abs/1905.07953) | Graph clustering |
| 2021 | [GNNAutoScale: Scalable and Expressive Graph Neural Networks via Historical Embeddings](https://arxiv.org/abs/2106.05609) | Historical embeddings |
| 2020 | [GraphSAINT: Graph Sampling Based Inductive Learning Method](https://arxiv.org/abs/1907.04931) | Subgraph sampling |

## 🤖 GNNs + Large Language Models (2023-2024)

| Year | Paper | Innovation |
|------|-------|------------|
| 2023 | [GraphGPT: Graph Instruction Tuning for Large Language Models](https://arxiv.org/abs/2310.13023) | LLM for graph reasoning |
| 2024 | [LLM4GNN: Large Language Models for Graph Neural Networks](https://arxiv.org/abs/2306.08302) | LLM-enhanced node features |
| 2024 | [Talk Like a Graph: Encoding Graphs for Large Language Models](https://arxiv.org/abs/2310.04560) | Graph-to-text for LLMs |

## 📚 Survey Papers

| Year | Paper | Coverage |
|------|-------|----------|
| 2020 | [A Comprehensive Survey on Graph Neural Networks](https://arxiv.org/abs/1901.00596) | Comprehensive overview |
| 2021 | [Graph Neural Networks: A Review of Methods and Applications](https://arxiv.org/abs/1812.08434) | Methods + applications |
| 2022 | [Graph Representation Learning Book](https://www.cs.mcgill.ca/~wlh/grl_book/) | William Hamilton's GRL book |
| 2024 | [A Survey of Graph Neural Networks in Real World](https://arxiv.org/abs/2402.00447) | Real-world challenges |

## 📖 Books & Courses

### Books
- **Graph Representation Learning** - William L. Hamilton ([Free PDF](https://www.cs.mcgill.ca/~wlh/grl_book/))
- **Deep Learning on Graphs** - Yao Ma & Jiliang Tang ([Website](https://dlg4nlp.github.io/))

### University Courses
- **Stanford CS224W** - Machine Learning with Graphs ([Course Site](http://web.stanford.edu/class/cs224w/))
- **MIT 6.S897** - Machine Learning for Healthcare (includes GNNs)

### Online Tutorials
- **PyTorch Geometric Tutorials** ([GitHub](https://github.com/AntonioLonga/PytorchGeometricTutorial))
- **Distill: A Gentle Introduction to GNNs** ([Article](https://distill.pub/2021/gnn-intro/))

## 🏆 Top Conference Papers (2023-2024)

### ICLR 2024
- [GitHub Collection](https://github.com/JEFLBROWN/Awesome-Graph-Research-ICLR2024)

### ICML 2024  
- [GitHub Collection](https://github.com/JEFLBROWN/Awesome-Graph-Research-ICML2024)

### NeurIPS 2024
- Focus areas: GNN explainability, Graph Transformers, LLM integration

---

## Reading Order Recommendation

### Beginner Path (2-3 weeks)
1. GCN (Kipf & Welling, 2017)
2. GAT (Veličković et al., 2018)
3. GraphSAGE (Hamilton et al., 2017)
4. Comprehensive Survey (2020)

### Intermediate Path (1-2 weeks additional)
5. GIN (Xu et al., 2019)
6. Over-smoothing papers
7. Application paper in your domain

### Advanced Path (ongoing)
8. Latest conference papers
9. Scalability techniques
10. GNN + LLM integration
