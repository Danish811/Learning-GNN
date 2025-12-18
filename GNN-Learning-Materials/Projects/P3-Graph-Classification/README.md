# 🧬 Project 3: Graph Classification Challenge

**Difficulty:** ⭐⭐ Intermediate  
**Estimated Time:** 8-12 hours  
**Prerequisites:** Complete Projects 1-2 + Modules 1-2

---

## 🎯 Your Mission

You've been hired by a pharmaceutical company. They have thousands of molecular compounds and need to predict which ones are mutagenic (cause DNA mutations) without expensive lab tests.

**Your task:** Build a GNN that classifies **entire molecules** as mutagenic or non-mutagenic.

This is fundamentally different from Projects 1 and 2: you're classifying **whole graphs**, not nodes or edges.

---

## 🧠 Before You Begin: Conceptual Foundation

**No code until these are answered!**

### The Core Challenge

Think through this carefully:

1. In node classification, each node has features and you predict a label for each node.
2. In graph classification, you have a **whole graph** with many nodes, but only **ONE label** for the entire graph.
3. How do you go from "many node embeddings" to "one graph embedding"?

### Fundamental Questions

1. If Graph A has 10 nodes and Graph B has 50 nodes, how can they both produce embeddings of the same size?
2. What information might be lost when you compress a variable-size graph into a fixed-size vector?
3. What information should definitely be preserved?

### Research Before Starting

Research and write short explanations (3-4 sentences each):

1. What is "graph pooling" or "readout"?
2. What are at least 3 different pooling methods? (Name them and describe what they do)
3. What is the difference between "flat" and "hierarchical" pooling?
4. What makes molecular graphs different from social networks?

---

# Phase 1: Understanding Molecular Graphs (2+ hours)

## Task 1.1: Molecule as Graphs

Before touching any code, understand the domain.

### Questions to Answer:

1. In a molecular graph:
   - What do nodes represent?
   - What do edges represent?
   - What might node features include?
   - What might edge features include?

2. Draw (by hand) the graph representation of:
   - Water (H₂O)
   - Methane (CH₄)
   - Benzene (C₆H₆)

3. Why is molecular representation as a graph natural?

4. What information about a molecule is captured by the graph? What is NOT captured?

---

## Task 1.2: The MUTAG Dataset

You'll use the MUTAG dataset of mutagenic compounds.

### Research Questions:

1. What is mutagenicity? Why is it important to predict?
2. How many molecules are in MUTAG?
3. How are molecules represented (what features)?
4. What is the class balance?

### After Loading the Data:

(You may write code now just to load and inspect)

1. What is the average number of atoms per molecule?
2. What is the range (min and max)?
3. What is the average number of bonds per molecule?
4. Create a histogram of molecule sizes.

---

## Task 1.3: The Variable Size Problem

This is the crux of graph classification.

### Think Through:

```
Molecule A: 17 atoms → 17 node embeddings of size 64 → ??? → 1 class prediction
Molecule B: 45 atoms → 45 node embeddings of size 64 → ??? → 1 class prediction

Both need to produce a fixed-size representation for the classifier!
```

### Questions:

1. If you just concatenate all node embeddings, what problems arise?
2. If you average all node embeddings, what information is lost?
3. If you sum all node embeddings, what changes with molecule size?
4. If you take max over all node embeddings, what is preserved?

### Compare (Write 2-3 sentences for each):

1. **Mean pool:** Average of all node embeddings
2. **Sum pool:** Sum of all node embeddings
3. **Max pool:** Element-wise maximum across nodes
4. **Attention pool:** Weighted sum based on learned attention

For each: What are the pros and cons? When would each be best?

---

### ✅ Phase 1 Checkpoint

- [ ] Hand-drawn molecular graphs (at least 3)
- [ ] Written answers about molecular representation
- [ ] Dataset statistics and histogram
- [ ] Comparison table of 4+ pooling methods

---

# Phase 2: Batching Challenge (2+ hours)

This is trickier than it sounds. Understand it deeply.

## Task 2.1: The Batching Problem

In image classification, batching is easy: stack images (all same size).

In graph classification:
- Graph 1: 10 nodes
- Graph 2: 25 nodes  
- Graph 3: 8 nodes

How do you batch these?

### Questions:

1. Can you pad graphs to the same size? What are the problems with this?
2. What is PyG's approach to batching graphs?
3. What is the "batch" tensor and what does it contain?
4. If you batch 3 graphs with 10, 25, and 8 nodes respectively:
   - How many total nodes in the batch?
   - What is the shape of the batch tensor?
   - What values does the batch tensor contain?

### Critical Understanding:

Draw a diagram showing:
1. Three separate graphs (before batching)
2. The combined "mega-graph" (after batching)
3. Explain how the batch tensor tracks graph membership

---

## Task 2.2: Pooling with Batches

Here's where it gets interesting.

### Question:

If you have a batched mega-graph with 43 nodes, and you apply mean pooling:
- Do you take the mean of ALL 43 nodes? 
- Or the mean within each original graph?

### Task:

Explain (in your own words) how `global_mean_pool(x, batch)` works:
1. What is `x`? What is its shape?
2. What is `batch`? What is its shape?
3. What is the output? What is its shape?
4. Write pseudocode for what this function does.

---

## Task 2.3: Forward Pass Design

Before writing code, design your forward pass on paper.

### For a batch of graphs:

1. Input: What shapes do `x`, `edge_index`, and `batch` have?
2. After GNN layers: What shape is `x`?
3. After pooling: What shape is `x`?
4. After classifier: What shape is the output?

**Draw a complete diagram showing all shapes at each step.**

---

### ✅ Phase 2 Checkpoint

- [ ] Diagram of batching 3 graphs
- [ ] Explanation of batch tensor
- [ ] Pseudocode for global_mean_pool
- [ ] Complete forward pass diagram with shapes

---

# Phase 3: Architecture Design (2+ hours)

## Task 3.1: GNN Backbone Selection

### Questions to Consider:

1. You learned about GCN, GAT, and GraphSAGE. Which might be best for molecules?
2. What is GIN (Graph Isomorphism Network)? Why was it designed?
3. What does "as powerful as the WL test" mean?
4. Why might GIN be particularly good for graph classification?

### Research Task:

Read about the Weisfeiler-Lehman graph isomorphism test:
1. What problem does it solve?
2. How is it related to GNN expressiveness?
3. What is the key difference between sum aggregation and mean aggregation for expressiveness?

---

## Task 3.2: Depth and Over-smoothing

### Questions:

1. For molecular graphs with diameter ~8, how many GNN layers might you need?
2. If you use too many layers, what happens? (Think back to Project 1)
3. What techniques can help with deeper networks?

### Design Decision:

Decide how many GNN layers you'll use. **Write a justification** (3-4 sentences) for your choice.

---

## Task 3.3: Complete Architecture Design

Design your complete architecture on paper:

1. Number and type of GNN layers
2. Hidden dimensions
3. Activations
4. Pooling strategy
5. Classifier (how many layers? dimensions?)
6. Dropout/regularization

**Create a diagram showing all components.**

---

### ✅ Phase 3 Checkpoint

- [ ] Research notes on GIN and WL test
- [ ] Architecture diagram with all components
- [ ] Written justification for each design choice
- [ ] Parameter count estimate

---

# Phase 4: Implementation and Experimentation (3+ hours)

Now code.

## Task 4.1: Implement Base Model

Build your designed model. Do NOT look at tutorials — implement from your design.

### Verify:
1. Model runs without errors on one batch
2. Output shape is [batch_size, num_classes]
3. Loss computes correctly

---

## Task 4.2: Training Pipeline

Implement:
1. Data splits (train/test) — decide on split ratio and justify
2. DataLoader with appropriate batch size
3. Training loop
4. Validation monitoring

### Questions After First Training:

1. What accuracy did you achieve?
2. Is the model overfitting? How do you know?
3. What does the loss curve look like?

---

## Task 4.3: Pooling Ablation

This is important: compare pooling strategies!

Implement and compare:
1. Global mean pooling
2. Global max pooling
3. Global add (sum) pooling
4. Combination (mean + max concatenated)

**Create a table showing accuracy for each.**

Which works best? Why do you think so?

---

## Task 4.4: Architecture Ablation

Experiment with:
1. Different number of layers (2, 3, 4, 5)
2. Different GNN types (GCN vs GIN)
3. Different hidden dimensions

**Create a table with at least 8 experiments.**

---

### ✅ Phase 4 Checkpoint

- [ ] Working model with documented accuracy
- [ ] Pooling comparison table
- [ ] Architecture ablation table
- [ ] Best configuration identified

---

# Phase 5: Advanced Exploration (Optional, 2+ hours)

## Task 5.1: Attention Pooling

Implement a simple attention-based pooling:
- Learn a weight for each node
- Weighted sum instead of simple mean

Does it improve accuracy?

---

## Task 5.2: Edge Features

MUTAG has edge features (bond types). Are you using them?

Investigate:
1. How to incorporate edge features in your GNN
2. Does it improve accuracy?

---

## Task 5.3: Different Dataset

Try your model on a different dataset:
- PROTEINS
- ENZYMES
- NCI1

Does your best architecture transfer?

---

# Phase 6: Final Deliverables

## Report Requirements

1. **Introduction:** Graph classification problem and challenges
2. **Dataset:** Complete analysis of MUTAG
3. **Method:** Your architecture with diagrams
4. **Experiments:** All ablation results with tables
5. **Discussion:** What worked? What didn't?
6. **Conclusion:** Key learnings

---

## 🏆 Success Criteria

- [ ] Test accuracy > 75% (aim for 85%+)
- [ ] At least 2 pooling methods compared
- [ ] At least 8 architecture experiments
- [ ] Complete report with all sections
- [ ] All questions answered throughout project

---

**Next Project:** [Molecular Property Prediction Challenge →](../P4-Molecular-Properties/)
