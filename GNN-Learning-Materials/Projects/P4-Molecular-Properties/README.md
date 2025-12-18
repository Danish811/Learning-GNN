# 💊 Project 4: Molecular Property Prediction Challenge

**Difficulty:** ⭐⭐⭐ Advanced  
**Estimated Time:** 12-16 hours  
**Prerequisites:** Complete Projects 1-3

---

## 🎯 Your Mission

You're a machine learning researcher at a drug discovery company. The company screens thousands of molecules annually for multiple toxicity endpoints.

**Your task:** Build a multi-task GNN that predicts **12 different toxicity properties** simultaneously for drug-like molecules.

This is a real-world challenge with:
- Multi-task learning
- Severely missing labels
- Imbalanced classes
- Molecular domain knowledge

---

## 🧠 Before You Begin: Deep Conceptual Understanding

This project requires significant preparation. Spend at least 2-3 hours on this section before any coding.

### Drug Discovery Context

Research and write answers (5+ sentences each):

1. What is the drug discovery pipeline? Where does computational screening fit?
2. What is "toxicity" in the context of drug development?
3. Why would a company want to predict toxicity before synthesis?
4. What is the cost of a failed drug in clinical trials due to toxicity?

### Multi-Task Learning

Research and answer:

1. What is multi-task learning? How does it differ from training separate models?
2. What might be the advantages of predicting multiple properties jointly?
3. What might be the disadvantages?
4. How do tasks "help" each other in multi-task learning?
5. When might multi-task learning hurt compared to single-task?

### The Missing Labels Problem

This is crucial for Tox21:

1. In Tox21, approximately 75% of labels are missing. What does this mean?
2. Why are labels missing? (Think about how the data was collected)
3. Can you impute (fill in) missing labels? Why or why not?
4. How should you handle missing labels in your loss function?
5. How do you evaluate when labels are missing?

---

# Phase 1: Domain Understanding (3+ hours)

## Task 1.1: Understanding Molecular Representations

Unlike previous projects, you need domain knowledge.

### Questions to Answer:

1. What is SMILES notation? Convert these to molecular graphs (by hand):
   - "CCO" (ethanol)
   - "c1ccccc1" (benzene)
   
2. In molecular graphs:
   - What properties of atoms become node features?
   - What properties of bonds become edge features?
   - Why is 3D structure often NOT captured?

3. What is the difference between:
   - Constitutional descriptors
   - Topological descriptors
   - Fingerprints
   - Graph neural network embeddings

4. What is the "scaffold" of a molecule? Why does it matter for ML?

---

## Task 1.2: The Tox21 Dataset Deep Dive

### Background Research:

1. What organization created Tox21? Why?
2. What are the 12 toxicity endpoints? (List them all)
3. What laboratory assays were used? (General understanding)
4. How were molecules selected for the dataset?

### Data Analysis (You may now write code to explore):

1. How many molecules in total?
2. For each of the 12 endpoints:
   - How many positive labels?
   - How many negative labels?
   - How many missing labels?
   - Create a visualization

3. What is the average number of labels per molecule?
4. Are some molecules labeled for all 12 endpoints? How many?
5. Are some molecules labeled for only 1 endpoint? How many?

### Molecular Statistics:

1. Distribution of molecule sizes (number of atoms)
2. Distribution of molecule "complexity" (number of rings, etc.)
3. Distribution of atom types
4. Distribution of bond types

---

## Task 1.3: The Imbalance Challenge

Toxicity data is highly imbalanced. Understand this deeply.

### Questions:

1. For the most imbalanced endpoint, what is the positive/negative ratio?
2. If a model predicts "non-toxic" for everything, what accuracy does it get?
3. Why is accuracy a misleading metric here?
4. What metrics are better for imbalanced classification?
5. What is AUC-ROC? What does it measure intuitively?
6. What is AUC-PRC? When is it preferred over ROC?

---

### ✅ Phase 1 Checkpoint

- [ ] Written answers to all 25+ questions above
- [ ] Visualization of label distribution across 12 endpoints
- [ ] Molecular statistics histograms
- [ ] Understanding of evaluation challenges

---

# Phase 2: Architecture Design Decisions (2+ hours)

## Task 2.1: Encoder Selection

### Analysis Question:

For molecular property prediction, compare these architectures:

1. **GCN:** Pros/cons for molecules?
2. **GIN:** Why is it popular for molecules?
3. **MPNN (Message Passing Neural Network):** How does it incorporate edge features?
4. **AttentiveFP:** What does attention add for molecules?

**Write a comparison table and select one with justification (1 paragraph).**

---

## Task 2.2: Multi-Task Head Design

You need to predict 12 properties. Design options:

### Option A: Shared trunk, separate heads
```
Molecules → GNN → Shared embedding → 12 separate MLPs → 12 predictions
```

### Option B: Fully shared
```
Molecules → GNN → Shared embedding → Single MLP → 12 predictions
```

### Option C: Mixed sharing
```
Molecules → GNN → Shared embedding → Grouped MLPs (endpoints that share biology) → predictions
```

### Questions:

1. What are the tradeoffs of each design?
2. Which has more parameters? Fewer?
3. Which allows tasks to help each other most?
4. What is "negative transfer" and how might it manifest here?

**Design your approach and justify it.**

---

## Task 2.3: Loss Function Design

With missing labels, you can't use standard cross-entropy.

### Questions:

1. Write pseudocode for computing loss that ignores missing labels.
2. Should you weight tasks equally? Why or why not?
3. What is "uncertainty weighting" in multi-task learning?
4. What is "focal loss" and might it help here?

---

### ✅ Phase 2 Checkpoint

- [ ] Architecture comparison table
- [ ] Multi-task head design with justification
- [ ] Loss function pseudocode
- [ ] Handling of missing labels explained

---

# Phase 3: Implementation (4+ hours)

## Task 3.1: Data Pipeline

Build a robust data pipeline:

1. Load Tox21 dataset
2. Create appropriate train/val/test splits
3. Handle missing labels properly
4. Set up DataLoader

### Critical Question:

How do you split molecules? Randomly? By scaffold?

Research:
- What is "scaffold split"?
- Why might it give more realistic performance estimates?
- What is data leakage and how does scaffold split prevent it?

---

## Task 3.2: Model Implementation

Implement your designed architecture with:

1. Molecular encoder (GNN layers)
2. Global pooling
3. Multi-task prediction heads
4. Proper handling of missing labels in forward pass

### Verification:

- Does your model run without errors?
- Are shapes correct at each layer?
- Is the loss computation correct for missing labels?

---

## Task 3.3: Training with Care

This is harder than previous projects. Consider:

1. Learning rate scheduling
2. Early stopping for each task?
3. Gradient clipping
4. How to monitor multiple tasks during training

### Questions:

1. How do you decide when to stop training with 12 tasks?
2. If some tasks overfit before others, what do you do?
3. How do you balance gradients from different tasks?

---

## Task 3.4: Per-Task Evaluation

Evaluate each of the 12 endpoints separately:

1. Compute AUC-ROC for each
2. Compute AUC-PRC for each
3. Identify which endpoints are easy/hard
4. Compute mean AUC across tasks

---

### ✅ Phase 3 Checkpoint

- [ ] Complete data pipeline with scaffold split
- [ ] Working multi-task model
- [ ] Training loop with proper monitoring
- [ ] Per-task evaluation table

---

# Phase 4: Advanced Analysis (3+ hours)

## Task 4.1: Baseline Comparisons

Implement and compare against:

1. **Random Forest** on molecular fingerprints
2. **Single-task models** (12 separate GNNs)
3. **Simple MLP** on fingerprints

### Analysis:

1. Does your GNN beat fingerprints? By how much?
2. Does multi-task learning help compared to single-task?
3. For which endpoints is the difference largest?

---

## Task 4.2: Error Analysis

For the hardest endpoint:

1. Examine 20 false positives (predicted toxic, actually not)
2. Examine 20 false negatives (predicted safe, actually toxic)
3. What molecular features correlate with errors?
4. Are certain molecular scaffolds problematic?

---

## Task 4.3: Task Relationship Analysis

Investigate task correlations:

1. Compute pairwise correlation of task labels
2. Which tasks are most correlated?
3. Do correlated tasks have similar model performance?
4. Could you use task relationships to improve predictions?

---

## Task 4.4: Interpretability

For 3 molecules, try to understand the predictions:

1. Visualize attention weights (if using attention)
2. What atoms contribute most to toxicity prediction?
3. Does this make chemical sense? (Research toxic functional groups)

---

### ✅ Phase 4 Checkpoint

- [ ] Baseline comparison table
- [ ] Error analysis document with 40 examples
- [ ] Task correlation matrix
- [ ] Interpretability analysis for 3 molecules

---

# Phase 5: Final Report (2+ hours)

## Required Sections

1. **Abstract:** One paragraph summary
2. **Introduction:** Drug discovery, toxicity prediction, multi-task learning
3. **Related Work:** At least 5 papers you read
4. **Dataset:** Complete Tox21 analysis
5. **Methods:** Architecture, training, evaluation
6. **Experiments:** All comparisons and ablations
7. **Results:** Best performance with confidence intervals
8. **Discussion:** What worked, what didn't, why
9. **Conclusion:** Key takeaways
10. **References:** All papers cited

---

## 🏆 Success Criteria

- [ ] Mean AUC-ROC > 0.75 on test set
- [ ] Multi-task beats single-task (or you explain why not)
- [ ] Complete error analysis
- [ ] Beats fingerprint baseline on at least 8/12 tasks
- [ ] Comprehensive report (10+ pages)
- [ ] All 50+ questions answered throughout

---

**Next Project:** [Social Network Analysis Challenge →](../P5-Social-Network-Analysis/)
