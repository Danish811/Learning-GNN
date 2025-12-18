# 🎬 Project 6: Recommendation System Challenge

**Difficulty:** ⭐⭐⭐ Advanced  
**Estimated Time:** 15-20 hours  
**Prerequisites:** Complete Projects 1-5

---

## 🎯 Your Mission

You're building the recommendation engine for a streaming service. Users interact with content (movies, shows), and you need to predict: **What will each user want to watch next?**

This project combines:
- Graph neural networks
- Recommendation systems
- Real-world evaluation challenges
- Production-level considerations

---

## 🧠 Before You Begin: The Recommendation Landscape

Spend 3+ hours on conceptual understanding before any code.

### The Evolution of RecSys

Research and summarize (one paragraph each):

1. **Collaborative Filtering:** What is it? How does it work?
2. **Matrix Factorization:** How did this improve on early CF?
3. **Deep Learning RecSys:** What neural approaches emerged?
4. **Graph-based RecSys:** How do GNNs fit into this landscape?

### Why Graphs?

Think through these questions:

1. How can you represent user-item interactions as a graph?
2. What type of graph is this? (Hint: bipartite)
3. Draw an example user-item graph with 4 users and 6 items
4. In this graph, what does it mean for two users to be "close"?
5. What does message passing DO in this context?

### The Cold Start Problem

1. What is the cold start problem?
2. How does it manifest for new users? New items?
3. Why are many traditional methods bad at cold start?
4. How might GNNs help (or not help)?

---

# Phase 1: Recommendation Fundamentals (3+ hours)

## Task 1.1: Understand the Problem Formally

### Formalization:

1. Define the recommendation problem mathematically
   - What is the input?
   - What is the output?
   - What is the goal?

2. What is implicit feedback? How is it different from explicit ratings?

3. For MovieLens (your dataset), do you have implicit or explicit feedback?

4. If you have ratings (1-5 stars), should you still treat it as implicit? Why?

---

## Task 1.2: Evaluation in Recommendation

This is subtle and important. Research thoroughly.

### The Problem:

In classification, you have labels. In recommendation:
- You know what users DID interact with
- You don't know what they WOULD interact with if shown
- Absence of interaction ≠ dislike

### Questions:

1. What is the difference between accuracy and ranking metrics?
2. What is Recall@K? Intuitive explanation and formula.
3. What is NDCG@K? What does the "discounted" and "cumulative" mean?
4. What is Hit Rate? How does it differ from precision?
5. What is Mean Reciprocal Rank (MRR)?

### Design Decision:

Which metric is most appropriate for:
- A video streaming service where users watch one thing at a time?
- An e-commerce site where users browse many items?
- A music service with continuous playback?

---

## Task 1.3: Training Objective

### The BPR Paper

Read or skim: "BPR: Bayesian Personalized Ranking from Implicit Feedback"

Answer:
1. What does BPR optimize?
2. What are positive pairs? Negative pairs?
3. Why can't you use cross-entropy loss directly?
4. What is the assumption about unobserved data?
5. Write the BPR loss formula and explain each term.

---

### ✅ Phase 1 Checkpoint

- [ ] Problem formalization written
- [ ] 5 evaluation metrics explained with examples
- [ ] BPR loss understood and derived
- [ ] Cold start problem analyzed

---

# Phase 2: Data Preparation (3+ hours)

## Task 2.1: MovieLens Exploration

Use MovieLens 100K or 1M (choose and justify).

### Basic Statistics:
1. Number of users, items, ratings
2. Rating distribution (histogram)
3. Sparsity (what percentage of user-item pairs have ratings?)

### User Analysis:
1. Distribution of number of ratings per user
2. Who are the most active users?
3. Average, median, min, max ratings per user

### Item Analysis:
1. Distribution of number of ratings per movie
2. What are the most rated movies?
3. What are the least rated? (1 rating only?)

---

## Task 2.2: Build the Graph

Transform ratings into a bipartite graph.

### Design Decisions:

1. Should you include all ratings, or only positive ones (e.g., ≥4)?
2. How do you encode user IDs and item IDs into a single node space?
3. Should edges be weighted by rating value?
4. Should you add self-loops? Why or why not?

### Implementation:

Create `edge_index` for the user-item bipartite graph.
Verify:
1. Number of edges
2. No duplicate edges
3. Correct ID ranges

---

## Task 2.3: Train/Test Split

This is tricky for recommendation.

### Options:

1. **Random split:** Random 80/20 of interactions
2. **Leave-one-out:** Hold out last item per user
3. **Temporal split:** All interactions before time T for train

### Questions:

1. What is data leakage in recommendation splits?
2. If you randomly split, what information leaks?
3. Why is temporal split more realistic but harder?

**Choose a split strategy and justify (1 paragraph).**

---

### ✅ Phase 2 Checkpoint

- [ ] Complete data statistics report
- [ ] Bipartite graph constructed
- [ ] Split strategy chosen with justification
- [ ] No data leakage verified

---

# Phase 3: Architecture Design (3+ hours)

## Task 3.1: LightGCN Deep Dive

Read: "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation"

### Questions:

1. What is LightGCN's key insight about simplification?
2. What components of GCN does it remove? Why?
3. How does it aggregate across layers?
4. What is the normalization scheme?
5. Write the LightGCN layer equation.

### Comparison:

Compare to NGCF (Neural Graph Collaborative Filtering):
1. What does NGCF do that LightGCN doesn't?
2. Why does LightGCN still work (or work better)?

---

## Task 3.2: Design Your Model

### Questions to Answer:

1. Number of layers (study the paper's recommendations)
2. Embedding dimension
3. How to predict score for a user-item pair?
4. How to generate top-K recommendations?

### Key Design:

Draw the complete architecture:
- User embeddings initialization
- Item embeddings initialization
- Message passing layers
- How to get final embeddings
- Scoring mechanism

---

## Task 3.3: Negative Sampling Strategy

### Questions:

1. For each positive (user, item) pair, how many negatives should you sample?
2. Should you sample uniformly? Or popularity-weighted?
3. What is "hard negative mining"?
4. What problems occur if negatives are too easy?

**Design your negative sampling strategy with justification.**

---

### ✅ Phase 3 Checkpoint

- [ ] LightGCN paper summarized
- [ ] Architecture diagram
- [ ] Scoring mechanism defined
- [ ] Negative sampling strategy designed

---

# Phase 4: Implementation (4+ hours)

## Task 4.1: Implement LightGCN

Build from scratch (no copying from tutorials):

1. User and item embeddings
2. Light convolution (no activation, no transformation)
3. Layer aggregation
4. Scoring function

### Verify:

1. Can you run a forward pass?
2. Are embedding shapes correct?
3. Is the graph structure correct?

---

## Task 4.2: Implement BPR Training

1. Positive edge sampling
2. Negative edge sampling
3. BPR loss computation
4. Training loop

### Monitor:

- Loss curve
- Training time per epoch
- Memory usage

---

## Task 4.3: Implement Evaluation

This is complex. Implement carefully:

For each test user:
1. Get their embedding
2. Score ALL items they haven't interacted with
3. Exclude training items from ranking
4. Get top-K items
5. Compute metrics against actual test items

### Implement:
- Recall@K (K = 10, 20, 50)
- NDCG@K (K = 10, 20)
- Hit Rate@K

### Efficiency:

For 10K+ users, computing metrics for all items per user is slow. How can you speed this up?

---

### ✅ Phase 4 Checkpoint

- [ ] LightGCN model runs
- [ ] BPR training works with decreasing loss
- [ ] All evaluation metrics implemented
- [ ] Metrics computed on test set

---

# Phase 5: Experiments and Analysis (3+ hours)

## Task 5.1: Hyperparameter Study

Experiment with:

1. Embedding dimension: 32, 64, 128, 256
2. Number of layers: 1, 2, 3, 4
3. Learning rate: 0.0001, 0.001, 0.01
4. Regularization weight

Create a table with best combination.

---

## Task 5.2: Baseline Comparison

Implement or use existing implementations of:

1. **PopularityBased:** Recommend most popular items
2. **Matrix Factorization (MF):** Basic embedding dot product
3. **NeuMF:** Neural collaborative filtering (if time permits)

Compare to your LightGCN on all metrics.

---

## Task 5.3: Cold Start Analysis

Analyze performance by user activity:

1. Group users by number of training interactions
2. Compute metrics for each group
3. Does LightGCN help cold users more than MF?

---

## Task 5.4: Item-level Analysis

1. Which items are recommended most often?
2. Are popular items over-represented?
3. What is the "coverage" of your recommendations?
4. Is there a popularity bias issue?

---

### ✅ Phase 5 Checkpoint

- [ ] Hyperparameter table
- [ ] Baseline comparison table
- [ ] Cold start analysis
- [ ] Popularity bias analysis

---

# Phase 6: Production Considerations (2+ hours)

## Task 6.1: Scalability

1. How does training time scale with users/items?
2. At what scale does your implementation fail?
3. What would you change for 1M users?

---

## Task 6.2: Serving

1. How would you generate recommendations in real-time?
2. Can you pre-compute all recommendations? Space requirements?
3. Approximate nearest neighbor? (FAISS, ScaNN)

---

## Task 6.3: Updating

1. New user joins - how to generate recommendations?
2. User watches something new - how to update?
3. Full retraining vs. incremental update - tradeoffs?

---

# Phase 7: Final Report

## Required Sections:

1. **Introduction:** Recommendation systems and GNNs
2. **Background:** LightGCN and related work
3. **Dataset:** Complete MovieLens analysis
4. **Methods:** Your implementation details
5. **Experiments:** All comparisons and ablations
6. **Analysis:** Cold start, popularity bias, etc.
7. **Production Considerations:** Scalability discussion
8. **Conclusion:** Key takeaways

---

## 🏆 Success Criteria

- [ ] LightGCN implemented from scratch
- [ ] BPR training working
- [ ] Recall@20 > 0.20 on test set
- [ ] Beats MF baseline by 10%+
- [ ] Cold start analysis completed
- [ ] All 100+ questions answered
- [ ] Comprehensive final report

---

## 🎉 Congratulations!

You've completed all 6 projects! You now have deep practical experience with:

- Node classification at scale
- Link prediction
- Graph classification
- Molecular property prediction
- Large-scale social networks
- Production recommendation systems

**Final Challenge:** [Capstone Project →](../Capstone-Molecular-Property-Prediction/)
