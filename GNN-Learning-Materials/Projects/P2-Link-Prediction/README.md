# 🔗 Project 2: Link Prediction Challenge

**Difficulty:** ⭐⭐ Intermediate  
**Estimated Time:** 8-12 hours  
**Prerequisites:** Complete Project 1 + Module 2

---

## 🎯 Your Mission

You're working at a research organization that tracks scientific collaborations. They have a network of researchers connected by co-authorship. 

**Your task:** Build a system that predicts **which researchers will collaborate in the future** based on the existing collaboration network.

This is a **link prediction** problem: given a graph, predict missing or future edges.

---

## 🧠 Before You Begin: Conceptual Understanding

**Do NOT write any code until you've thought through these questions:**

### Fundamental Questions

1. What makes link prediction different from node classification?
2. In node classification, each node has a label. In link prediction, what has a label?
3. If you have N nodes, how many possible edges are there? (Undirected graph)
4. If only 1% of possible edges exist, what challenge does this create?

### Research Questions

Before implementing, research and write short answers (2-3 sentences each):

1. What is an "encoder-decoder" architecture for link prediction?
2. What is "negative sampling" and why is it necessary?
3. What evaluation metrics are used for link prediction? Why not just accuracy?
4. What is the difference between "transductive" and "inductive" link prediction?

**Write down your answers before proceeding.**

---

# Phase 1: Problem Setup and Data Preparation (3+ hours)

## Task 1.1: Understanding the Data Challenge

For link prediction, we face a unique challenge: we need to **hide some edges** during training so we can test on them.

### Think Through This:

1. If we train on ALL edges and test on those same edges, what would happen?
2. Why is this different from node classification, where we don't need to hide nodes?
3. Draw a diagram showing:
   - Original graph
   - Training graph (some edges removed)
   - Test edges (the removed ones)

### Questions to Answer:

1. What percentage of edges should you hide for testing? Why?
2. Should we also have a validation set of edges? Why?
3. When doing message passing, should you use the full graph or the training graph? **Think carefully!**

---

## Task 1.2: The Negative Sampling Problem

This is crucial and tricky. Work through it carefully.

### The Problem:

```
Your graph has 1000 nodes and 5000 edges.
Possible edges: 1000 × 999 / 2 = 499,500
Existing edges: 5000
Non-edges: 494,500

If we only predict "connected" for everything, we'd be wrong 5000/5000 = never!
Wait... that seems wrong. What's the actual issue?
```

### Questions to Answer:

1. If you train only on positive edges (real connections), what will your model learn?
2. Where do "negative samples" come from?
3. What is the ratio of negative to positive samples you should use? Why?
4. Is every non-edge a valid negative sample? What could go wrong?

### Design Task:

Write pseudocode (not real code) for a function that:
- Takes a graph and desired number of negative samples
- Returns fake edges that DON'T exist in the graph
- Is efficient (think about how you'd do this for a large graph)

---

## Task 1.3: Edge Splitting Strategy

Now design your data split.

### Questions to Answer:

1. Should you split edges randomly, or does order matter?
2. If you remove too many edges, what happens to your graph? (Think: connectivity)
3. If you remove edges from node A to train on, can you still use node A for message passing?
4. What is "message leakage" in link prediction? How do you avoid it?

### Task:

Research `torch_geometric.transforms.RandomLinkSplit` and understand:
- What parameters does it take?
- What outputs does it produce?
- What does `add_negative_train_samples` do?

**Do not use it yet** — just understand what it does and why.

---

### ✅ Phase 1 Checkpoint

Before moving on:
- [ ] Written answers to all 15+ questions above
- [ ] Hand-drawn diagram of train/test edge splitting
- [ ] Pseudocode for negative sampling
- [ ] Understanding of RandomLinkSplit (without using it yet)

---

# Phase 2: Designing the Architecture (2+ hours)

## Task 2.1: Encoder Design

The "encoder" produces node embeddings. But for link prediction, we need some different considerations.

### Design Questions:

1. What should the output dimension of your encoder be? (This is not obvious!)
2. Should you use the same GCN architecture from Project 1? What might need to change?
3. Do you need a `forward` method that returns class probabilities? Or something else?

### Task:

Draw a diagram of your encoder architecture showing:
- Input shape
- Each layer and its output shape
- Final output shape
- What each output represents

---

## Task 2.2: Decoder Design

The "decoder" takes two node embeddings and predicts if they should connect.

### Research Task:

Investigate and compare these decoder approaches:
1. **Dot product:** Simply `z_u · z_v`
2. **Hadamard product + MLP:** `MLP(z_u ⊙ z_v)`
3. **Concatenation + MLP:** `MLP([z_u; z_v])`
4. **Distance-based:** `1 / (1 + ||z_u - z_v||)`

### Questions to Answer:

1. What are the pros and cons of each approach?
2. Which is fastest at inference time if you have 100,000 nodes?
3. Which can capture the most complex patterns?
4. What is the bias of dot product? (Hint: what does it assume about similarity?)
5. Why might simple methods outperform complex ones?

### Design Decision:

Choose one decoder approach and **justify your choice in writing**. Your justification should be at least 3-4 sentences.

---

## Task 2.3: Loss Function Selection

### Questions to Answer:

1. Is this a classification or regression problem?
2. What is Binary Cross-Entropy loss? Why is it appropriate here?
3. What is BPR (Bayesian Personalized Ranking) loss? How does it differ?
4. For edge prediction, what does each loss function optimize for?

### Task:

Write the mathematical formula for BCE loss. Make sure you understand each term.

---

### ✅ Phase 2 Checkpoint

Before writing any code:
- [ ] Diagram of encoder architecture
- [ ] Comparison table of 4 decoder approaches
- [ ] Written justification for your decoder choice
- [ ] Mathematical understanding of your loss function

---

# Phase 3: Implementation (3+ hours)

Now you may write code. But go slowly and verify each step.

## Task 3.1: Data Preparation

Implement edge splitting using `RandomLinkSplit`.

### Requirements:
1. Create train/val/test splits (research appropriate ratios)
2. Generate negative samples
3. Verify your splits: print statistics to confirm they look right

### Verification Questions:

After implementing, answer:
1. How many positive training edges do you have?
2. How many negative training edges?
3. Does the training graph remain connected?
4. What is the overlap between train and test edges? (Should be 0!)

---

## Task 3.2: Model Implementation

Build your encoder-decoder model.

### Requirements:
1. Encoder: GNN that produces node embeddings
2. Decoder: Your chosen approach to predict edge probability
3. The model should work with your prepared data

### Testing Your Implementation:

Before training, verify:
1. Does your encoder produce the expected output shape?
2. Does your decoder produce a scalar for each edge?
3. Can you run one forward pass without errors?
4. Does the loss compute correctly?

---

## Task 3.3: Training Loop

Implement training.

### Important Considerations:

1. Which edges do you use for message passing in the encoder?
2. Which edges do you use to compute loss?
3. How do you batch positive and negative edges?

### Implement:
1. Training loop for N epochs
2. Validation evaluation every E epochs
3. Early stopping based on validation performance

---

## Task 3.4: Evaluation

### Research Task:

Learn about these metrics for link prediction:
1. AUC-ROC
2. Average Precision (AP)
3. Hits@K
4. Mean Reciprocal Rank (MRR)

**For each metric, write:**
- What it measures (in plain English)
- When it's most appropriate
- Its range and what values are "good"

### Implement:

Choose 2 metrics and implement evaluation on your test set.

---

### ✅ Phase 3 Checkpoint

- [ ] Working data preparation with verified statistics
- [ ] Encoder-decoder model that runs without errors
- [ ] Training loop showing decreasing loss
- [ ] Evaluation on 2+ metrics

---

# Phase 4: Analysis and Improvement (2+ hours)

## Task 4.1: Baseline Comparison

Implement a **non-GNN baseline** to compare against.

### Options:

1. **Common Neighbors:** Predict based on how many neighbors two nodes share
2. **Random:** Random predictions
3. **Degree heuristic:** Predict based on node degrees

### Questions:

1. How does your GNN compare to these simple baselines?
2. If the GNN barely beats common neighbors, what does that tell you?
3. When would simple heuristics be sufficient?

---

## Task 4.2: Error Analysis

Examine where your model fails.

### Tasks:

1. Find 10 false positives (predicted edge, but no real edge)
   - What do these node pairs have in common?
   - Why might your model have predicted an edge?

2. Find 10 false negatives (missed real edges)
   - What makes these edges hard to predict?
   - Do they share few common neighbors?

3. Is there a pattern to the errors? Do they cluster in certain parts of the graph?

---

## Task 4.3: Architecture Ablation

Try at least 2 variations of your architecture:

1. Different encoder (GAT instead of GCN?)
2. Different decoder (switch to a different approach)
3. Different embedding dimension
4. Different number of layers

**Create a table comparing all variations** with your chosen metrics.

---

### ✅ Phase 4 Checkpoint

- [ ] Baseline comparison table
- [ ] Detailed error analysis (20 examples)
- [ ] Architecture ablation table (4+ experiments)
- [ ] Written analysis of what works and why

---

# Phase 5: Final Report (1 hour)

## Deliverables

Create a comprehensive report including:

1. **Introduction:** What is link prediction? Why does it matter?
2. **Methodology:** Your approach, with architecture diagrams
3. **Experiments:** All your comparisons and ablations
4. **Results:** Best metrics achieved
5. **Discussion:** What worked? What didn't? Why?
6. **Conclusion:** Key takeaways

---

## 🏆 Success Criteria

- [ ] AUC > 0.85 on test set
- [ ] Beats common neighbors baseline by 5%+
- [ ] Comprehensive report with all sections
- [ ] All questions answered throughout
- [ ] At least 4 architecture variations tested

---

## 📚 Resources (Only When Truly Stuck)

- [Link Prediction Survey Paper](https://arxiv.org/abs/2010.00906)
- [PyG Link Prediction Tutorial](https://pytorch-geometric.readthedocs.io/en/latest/notes/tutorial.html)

---

**Next Project:** [Graph Classification Challenge →](../P3-Graph-Classification/)
