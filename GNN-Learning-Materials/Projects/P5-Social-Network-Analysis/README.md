# 👥 Project 5: Large-Scale Social Network Analysis Challenge

**Difficulty:** ⭐⭐⭐ Advanced  
**Estimated Time:** 12-16 hours  
**Prerequisites:** Complete Projects 1-4

---

## 🎯 Your Mission

You're building a recommendation system for a streaming platform (like Twitch). You have a massive social network of users connected by mutual friendships, and you need to:

1. Classify users into categories
2. Detect communities
3. **Do this at scale: 168,000+ nodes, 6.8 million edges**

This project focuses on **scalability** — techniques that work on small graphs often fail here.

---

## 🧠 Before You Begin: The Scale Problem

Spend significant time understanding why scale matters.

### Memory Explosion Exercise

Do this calculation by hand:

1. You have a 2-layer GCN with hidden dimension 256
2. You have 168,000 nodes
3. Average node degree is 80

**Calculate:**
1. Size of the node feature matrix after layer 1
2. During a forward pass, how many messages are sent? (Roughly)
3. If each message is 256 floats (4 bytes each), how much memory for messages alone?

### The Reality:

1. Naive GCN on 168K nodes: ~32 GB GPU memory needed
2. Typical GPU: 8-16 GB
3. You CANNOT do a full forward pass

### Questions:

1. What are the two main resources that become bottlenecks at scale?
2. What is the difference between computational bottleneck and memory bottleneck?
3. Which is harder to solve? Why?

---

## Phase 1: Scalability Deep Dive (3+ hours)

## Task 1.1: Mini-Batch Training on Graphs

Research and understand the key scalable training methods:

### Method 1: Neighbor Sampling (GraphSAGE-style)

1. What is the core idea?
2. How are mini-batches constructed?
3. What is a "sampling fan-out"? (e.g., [25, 10])
4. How much does this reduce memory usage compared to full-batch?
5. What is the tradeoff — what do you lose?

### Method 2: Cluster-GCN

1. What is the core idea?
2. How is the graph partitioned?
3. What is the issue with inter-cluster edges?
4. When is this better than neighbor sampling?

### Method 3: GraphSAINT

1. What is different about GraphSAINT's sampling?
2. What are "random walk" samplers?
3. What are "node" vs "edge" samplers?
4. What bias does it correct for?

### Comparison:

Create a table comparing all three methods on:
- Memory efficiency
- Computation efficiency
- Variance of gradients
- Ease of implementation
- Best suited for (what graph types)

---

## Task 1.2: The NeighborLoader Deep Dive

You'll use PyG's NeighborLoader. Understand it deeply.

### Questions:

1. What does `num_neighbors=[25, 10]` mean exactly?
2. If you request batch_size=1024, how many nodes end up in the batch?
3. What is `input_nodes`? Why is it important?
4. What are the target nodes vs. context nodes in a batch?
5. Why does the forward pass use ALL batch nodes, but loss uses only target nodes?

### Draw a Diagram:

For a 2-layer GNN with num_neighbors=[3, 2]:
1. Start with one target node
2. Show its sampled 1-hop neighbors (3 nodes)
3. Show sampled 2-hop neighbors for each (2 nodes each)
4. Label which nodes are target vs. context

---

## Task 1.3: Inductive vs. Transductive

### Questions:

1. In the Twitch dataset, can new users join after training?
2. If you trained on 80% of nodes, can you classify the remaining 20% without retraining?
3. What property of GraphSAGE enables this?
4. Why can't vanilla GCN (as described in the paper) do inductive learning?
5. What's the difference between "learning embeddings" and "learning to generate embeddings"?

---

### ✅ Phase 1 Checkpoint

- [ ] Memory calculation exercise completed
- [ ] Comparison table of 3 scalable training methods
- [ ] NeighborLoader deep dive with all questions answered
- [ ] Sampling diagram (hand-drawn)

---

# Phase 2: Data Exploration (2+ hours)

## Task 2.1: Load and Profile the Graph

Load the Twitch dataset and answer:

### Basic Statistics:
1. Number of nodes and edges
2. Is the graph directed or undirected?
3. Node feature dimension
4. Number of classes and their distribution

### Graph Structure:
1. Average, min, max, median degree
2. Degree distribution (plot histogram)
3. Is it power-law? (many low-degree, few high-degree)
4. Graph diameter (or estimate if too expensive)
5. Number of connected components

### Feature Analysis:
1. What do the 7 node features represent? (Research!)
2. Are features normalized?
3. Any missing values?
4. Feature correlation with class labels

---

## Task 2.2: Sampling Behavior

Before training, understand how sampling works on YOUR data:

1. Create a NeighborLoader with num_neighbors=[25, 10], batch_size=1024
2. For one batch:
   - How many total nodes?
   - How many target nodes?
   - What is the ratio?
3. For a node with degree 200:
   - What fraction of neighbors are sampled?
4. For a node with degree 5:
   - Are ALL neighbors included?
5. Does sampling introduce bias toward low-degree nodes?

---

### ✅ Phase 2 Checkpoint

- [ ] Complete graph statistics
- [ ] Degree distribution visualization
- [ ] Feature analysis
- [ ] Sampling behavior analysis

---

# Phase 3: Model Implementation (3+ hours)

## Task 3.1: Architecture Decision

### Consider:

1. Why is GraphSAGE the natural choice here?
2. Could you use GAT with sampling? What changes?
3. How many layers should you use? (Relate to graph diameter)
4. What aggregator should you use? (mean vs pool)

**Write a design document (1 page) with your choices and justification.**

---

## Task 3.2: Implementation

Build your system with:

1. GraphSAGE model (experiment with 2-3 layers)
2. NeighborLoader for training
3. NeighborLoader for validation (with full neighborhood or sampled?)
4. Proper train/val/test splits

### Key Challenge:

For evaluation, you want accurate (not sampled) predictions. How do you do this efficiently?

Options:
1. Sample at evaluation too (fast but approximate)
2. Use all neighbors (accurate but slow)
3. Use larger sample counts (middle ground)

Research and implement a solution. Justify your choice.

---

## Task 3.3: Training

Train your model with:

1. Appropriate learning rate (schedule?)
2. Appropriate batch size (experiment!)
3. Early stopping based on validation performance
4. Track training time per epoch

### Measure:

1. Time per epoch
2. Peak GPU memory usage
3. Validation accuracy over training

---

### ✅ Phase 3 Checkpoint

- [ ] Design document with architecture justification
- [ ] Working model with sampled training
- [ ] Evaluation strategy implemented
- [ ] Training time and memory metrics

---

# Phase 4: Scaling Experiments (3+ hours)

## Task 4.1: Batch Size Experiments

Experiment with batch sizes: 256, 512, 1024, 2048, 4096

For each, measure:
1. Training time per epoch
2. Memory usage
3. Final accuracy

Create a plot showing tradeoffs.

---

## Task 4.2: Sampling Depth Experiments

Try different sampling configurations:
- [10]
- [25]
- [25, 10]
- [25, 10, 5]
- [50, 25, 10]

For each:
1. Training time
2. Accuracy
3. Memory usage

What is the optimal configuration? Why?

---

## Task 4.3: Full vs. Sampled Evaluation

Compare:
1. Sampled evaluation (fast)
2. Full evaluation (if possible, may OOM)
3. High-sample evaluation (middle ground)

How much accuracy do you lose with sampling?

---

## Task 4.4: Compare to Small-Graph Approach

If you have enough memory:
1. Try full-batch GCN on the entire graph
2. Compare accuracy to sampled approach
3. Compare memory and time

If you can't fit it:
1. Explain why
2. Estimate what hardware would be needed
3. Describe real-world constraints this creates

---

### ✅ Phase 4 Checkpoint

- [ ] Batch size experiment table and plot
- [ ] Sampling depth experiment table
- [ ] Evaluation comparison
- [ ] Full vs. sampled analysis

---

# Phase 5: Community Detection (2+ hours)

## Task 5.1: Extract Embeddings

After training:
1. Extract embeddings for all 168K nodes
2. You'll need to do this in batches!
3. Save embeddings to disk

---

## Task 5.2: Cluster Embeddings

Apply clustering to find communities:

1. Use K-means with different K values (try 5, 10, 20, 50)
2. Evaluate cluster quality with:
   - Silhouette score
   - Cluster size distribution

---

## Task 5.3: Community Analysis

For your best clustering:

1. Are communities correlated with the class labels?
2. What might communities represent? (Research Twitch user segments)
3. Visualize a sample using t-SNE (subsample if needed)
4. Can you characterize what makes each community distinct?

---

### ✅ Phase 5 Checkpoint

- [ ] All embeddings extracted and saved
- [ ] Clustering results for multiple K
- [ ] Community-label correlation analysis
- [ ] t-SNE visualization

---

# Phase 6: Final Report

## Required Sections:

1. **The Scale Problem:** Why this is fundamentally different from small graphs
2. **Dataset Analysis:** Complete Twitch network statistics
3. **Scalable Training:** Methods used and why
4. **Experiments:** All scaling experiments with plots
5. **Community Detection:** Results and interpretation
6. **Lessons Learned:** What would you do differently?

---

## 🏆 Success Criteria

- [ ] Training on 168K+ nodes successfully
- [ ] Test accuracy > 60%
- [ ] All scaling experiments completed
- [ ] Community detection analysis
- [ ] Training time under 10 minutes per epoch
- [ ] Comprehensive report

---

**Next Project:** [Recommendation System Challenge →](../P6-Recommendation-System/)
