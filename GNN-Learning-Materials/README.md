# 🧠 Graph Neural Networks: Complete Learning Guide

> *"Imagine if your social media could understand not just your posts, but HOW you're connected to your friends, and their friends, and use that to recommend amazing things!"*

**That's what Graph Neural Networks do — they learn from connections!**

---

## 🎯 What Will You Learn?

By the end of this course, you'll be able to:
- ✅ Understand how networks (social media, molecules, the internet) can be analyzed with AI
- ✅ Build models that learn from connections, not just data points
- ✅ Create real projects: friend recommendations, drug discovery, fraud detection!

---

## 🗺️ Your Learning Journey

```
🏁 START HERE
     │
     ▼
📚 Prerequisites ─────── Got Python & basic ML? Skip ahead!
     │
     ▼
🎓 FOUNDATIONS ──────── "What even is a graph?" (3 lessons)
     │
     ▼
🏗️ CORE ARCHITECTURES ── The big 3: GCN, GAT, GraphSAGE
     │
     ▼
🚀 ADVANCED ─────────── Go deeper: Transformers, Temporal
     │
     ▼
🔬 BUILD STUFF! ─────── 6 hands-on projects
     │
     ▼
🏆 CAPSTONE ─────────── Drug Discovery AI!
```

---

## 📚 Course Modules

| Module | What You'll Learn | Time |
|--------|------------------|------|
| **[00 - Prerequisites](./00-Prerequisites/)** | Python, PyTorch basics | 2-3 hrs |
| **[01 - Foundations](./01-Foundations/)** | Graphs, GNNs, Message Passing | 4-6 hrs |
| **[02 - Core Architectures](./02-Core-Architectures/)** | GCN, GAT, GraphSAGE | 6-8 hrs |
| **[03 - Advanced Concepts](./03-Advanced-Concepts/)** | Deep GNNs, Transformers | 4-6 hrs |
| **[04 - Training](./04-Training-Optimization/)** | Make it work at scale! | 3-4 hrs |
| **[05 - Applications](./05-Applications/)** | Real-world uses | 4-5 hrs |

---

## 🛠️ Hands-On Projects

| # | Project | Difficulty | What You'll Build |
|---|---------|------------|-------------------|
| 🟢 P1 | [Node Classification](./Projects/P1-Node-Classification/) | Beginner | Classify research papers by topic |
| 🟡 P2 | [Link Prediction](./Projects/P2-Link-Prediction/) | Intermediate | Predict who will become friends |
| 🟡 P3 | [Graph Classification](./Projects/P3-Graph-Classification/) | Intermediate | Classify molecules as toxic/safe |
| 🔴 P4 | [Molecular Properties](./Projects/P4-Molecular-Properties/) | Advanced | Predict drug properties |
| 🔴 P5 | [Social Networks](./Projects/P5-Social-Network-Analysis/) | Advanced | Find communities in Twitch |
| 🔴 P6 | [Recommendations](./Projects/P6-Recommendation-System/) | Advanced | Build a movie recommender |
| 🏆 | [Capstone](./Projects/Capstone-Molecular-Property-Prediction/) | Capstone | Full drug discovery pipeline |

---

## 🚀 Quick Start

### 1. Set Up Your Environment (5 minutes)

```bash
# Create a fresh Python environment
python -m venv gnn-env
gnn-env\Scripts\activate  # Windows
# source gnn-env/bin/activate  # Mac/Linux

# Install the magic ✨
pip install torch torch-geometric networkx matplotlib jupyter
```

### 2. Check It Works

```python
import torch
import torch_geometric
print(f"🔥 PyTorch: {torch.__version__}")
print(f"📊 PyG: {torch_geometric.__version__}")
print("✅ Ready to learn GNNs!")
```

### 3. Start Learning!

👉 **Begin with [Graph Theory Basics →](./01-Foundations/01-graph-theory-basics.md)**

---

## 💡 The Big Idea (in 30 seconds)

Traditional AI sees data as independent points:
```
Image 1: 🐱  →  "Cat"
Image 2: 🐕  →  "Dog"
(Each image analyzed separately)
```

**GNNs see connections:**
```
    👤 Alice
   / | \
  👤 👤 👤  ← "Alice is friends with Bob, Charlie, Diana"
   Bob     "What does that tell us about Alice?"
```

GNNs answer: **"You are who your friends are!"** 🤝

---

## 📚 Extra Resources

- **[Papers](./Resources/papers.md)** — 40+ must-read research papers
- **[Datasets](./Resources/datasets.md)** — Where to get graph data
- **[Frameworks](./Resources/frameworks.md)** — PyTorch Geometric & DGL guides

---

## 🤔 Who Is This For?

- 🎓 **Students** curious about cutting-edge AI
- 💻 **Developers** wanting to add GNNs to their toolkit
- 🔬 **Researchers** exploring graph-based learning
- 🎮 **Anyone** who thinks connections between things are cool!

---

**Ready to see AI in a whole new way?**

� **[Start Your Journey →](./01-Foundations/01-graph-theory-basics.md)** 🚀

---

*Made with 💜 for curious minds*
