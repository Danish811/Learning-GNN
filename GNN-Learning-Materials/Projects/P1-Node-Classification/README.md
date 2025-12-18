# 🎯 Project 1: Node Classification Challenge

**Difficulty:** ⭐ Beginner  
**Estimated Time:** 6-10 hours  
**Prerequisites:** Complete Modules 1-2

---

## 🎯 Your Mission

You've been hired as a data scientist at an academic research company. They have a large database of scientific papers, and they want to **automatically categorize papers by research topic** based on how they cite each other.

Your task: Build a system that predicts the topic of a paper using:
1. The paper's content (word features)
2. Which papers it cites and is cited by (graph structure)

---

## 📊 The Dataset

You'll work with the **Cora citation network**:
- ~2,700 research papers
- ~5,400 citation links between papers
- Each paper has word-presence features
- 7 research topics to predict

**Your first task:** Load this dataset and understand it thoroughly before writing any model code.

---

## 🧭 Project Phases

This project is divided into 5 phases. **Complete each phase fully before moving to the next.**

---

# Phase 1: Data Exploration (2+ hours)

## Task 1.1: Load and Inspect

Using PyTorch Geometric, load the Cora dataset and answer these questions **in a notebook or document**:

### Questions to Answer:

1. How many nodes (papers) are in the graph?
2. How many edges (citations) are there?
3. What is the dimensionality of the node features?
4. How many classes are there to predict?
5. What does each node feature represent? (Research this!)

<details>
<summary>💡 Hint: Loading the dataset</summary>

Look into `torch_geometric.datasets.Planetoid`
</details>

---

## Task 1.2: Understand the Splits

The dataset comes with predefined train/val/test splits.

### Questions to Answer:

1. How many nodes are in the training set?
2. What percentage of the total nodes is this?
3. Why do you think such a small percentage is used for training?
4. What is "semi-supervised learning" and why does it apply here?

<details>
<summary>💡 Hint: Accessing splits</summary>

Look for `train_mask`, `val_mask`, `test_mask` attributes
</details>

---

## Task 1.3: Analyze the Graph Structure

Before building any model, understand the graph:

### Questions to Answer:

1. What is the **average degree** (connections per node)?
2. What is the **maximum degree**? The minimum?
3. Is this graph **dense or sparse**? What percentage of all possible edges exist?
4. Visualize a small subset of the graph (e.g., a node and its neighbors). What do you observe?

### Reflection Questions:

> **Why do the answers to these questions matter for building a GNN?**
> 
> Write 2-3 sentences explaining how graph sparsity and degree distribution might affect your model design choices.

---

## Task 1.4: Explore Label Distribution

### Questions to Answer:

1. How many papers are in each of the 7 classes?
2. Are the classes balanced or imbalanced?
3. Create a bar chart showing the class distribution.
4. Does the class distribution in training match the overall distribution?

---

## Task 1.5: Feature Analysis

### Questions to Answer:

1. What is the average number of non-zero features per node?
2. Are some features much more common than others?
3. What preprocessing might be helpful? (Think: normalization, feature selection)

---

### ✅ Phase 1 Checkpoint

Before moving on, you should have:
- [ ] A notebook/document with answers to ALL questions above
- [ ] At least one visualization of the graph
- [ ] A bar chart of class distribution
- [ ] Written reflections on what you learned

**Do not proceed until you can explain the dataset to someone else without looking at your notes.**

---

# Phase 2: Baseline Model (2+ hours)

Now that you understand the data, build your first model.

## Task 2.1: Design Decisions

Before writing code, **think through your design**:

### Questions to Answer (Write down your answers!):

1. What is the input to your model? (Shape? Type?)
2. What is the output? (Shape? What does each output represent?)
3. How many GNN layers will you use? **Justify your choice.**
4. What hidden dimension will you use? **Justify your choice.**
5. What activation function? Why?
6. Will you use dropout? Why or why not?

> **Important:** There are no "right" answers here. The goal is to think through your choices and be able to defend them.

---

## Task 2.2: Build the Model

Now implement your design.

**Requirements:**
- Use GCNConv layers from PyTorch Geometric
- The model should be a class that extends `nn.Module`
- It should have a `forward` method that takes node features and edge index

**NO scaffolding code provided.** Build it yourself based on what you learned in Module 2.

<details>
<summary>💡 Hint: Stuck on structure?</summary>

Your model needs:
1. One or more GCNConv layers
2. Activation functions between layers
3. Optionally: dropout, batch normalization

Look at the GCNConv documentation for the expected input/output format.
</details>

---

## Task 2.3: Training Loop

Write your own training loop.

### Requirements:
1. Use an appropriate loss function (think: what kind of problem is this?)
2. Use an optimizer (Adam is a common choice)
3. Train for enough epochs to see convergence
4. Print loss every N epochs
5. Compute accuracy on the validation set periodically

### Questions to Answer:

1. Why do you only compute loss on **training nodes** even though all nodes participate in message passing?
2. What loss function did you choose and why?
3. How did you decide when to stop training?

---

## Task 2.4: Evaluate on Test Set

After training, evaluate on the **test set** (not validation!).

### Questions to Answer:

1. What is your test accuracy?
2. Is it higher or lower than validation accuracy? What might explain this?
3. Are there classes your model does particularly well or poorly on? Why might this be?

---

### ✅ Phase 2 Checkpoint

Before moving on:
- [ ] Working GCN model (your own code, not copied)
- [ ] Training loop that shows decreasing loss
- [ ] Test accuracy of at least 70% (aim for 80%+)
- [ ] Written answers to all design questions

---

# Phase 3: Understanding Your Model (1+ hours)

## Task 3.1: Ablation Study

Now we'll understand **why** your model works.

### Experiment 1: What if we removed the graph?

Create a baseline that ignores the graph structure:
- Just use a simple MLP on node features
- Compare accuracy to your GNN

**Question:** How much did the graph structure help? Why?

### Experiment 2: Number of layers

Try your model with:
- 1 layer
- 2 layers
- 3 layers
- 5 layers
- 10 layers

**Questions:**
1. Which number of layers works best? 
2. What happens with many layers? (This phenomenon has a name — research it!)
3. Why do you think this happens?

### Experiment 3: Hidden dimensions

Try different hidden dimensions: 4, 16, 64, 256

**Question:** What's the tradeoff? Is bigger always better?

---

## Task 3.2: Error Analysis

Look at the nodes your model got **wrong**:

### Questions to Answer:

1. Pick 5 incorrectly classified test nodes. What are their predicted vs true labels?
2. Do these errors have anything in common? (e.g., low degree nodes? Certain classes?)
3. Look at the neighbors of an incorrectly classified node. What are their labels?
4. Can you form a hypothesis about when your model fails?

---

### ✅ Phase 3 Checkpoint

- [ ] Comparison table: GNN vs MLP accuracy
- [ ] Graph showing accuracy vs number of layers
- [ ] Written analysis of at least 5 errors
- [ ] Hypothesis about model failure modes

---

# Phase 4: Improvement Challenge (2+ hours)

Your baseline works. Now make it better!

## Task 4.1: Research Better Architectures

Before implementing improvements, **research** these questions:

1. What is GAT (Graph Attention Networks)? How does it differ from GCN?
2. What is the potential advantage of attention for this problem?
3. What other architectures might help? (GraphSAGE? GIN?)

**Write a short paragraph comparing at least 2 architectures and explaining which you want to try next and why.**

---

## Task 4.2: Implement One Improvement

Choose ONE improvement and implement it:

**Option A:** Replace GCN with GAT
- How do attention heads work?
- How do you handle the output dimension with multiple heads?

**Option B:** Add residual connections
- What are residual connections?
- How do they help with deep networks?

**Option C:** Try a different aggregation function
- What alternatives exist to mean aggregation?
- Why might they help?

### After implementing:
1. Compare test accuracy to your baseline
2. Did it help? By how much?
3. What did you learn about why it did or didn't help?

---

## Task 4.3: Hyperparameter Tuning

Systematically tune at least 3 hyperparameters:
- Learning rate
- Hidden dimension
- Dropout rate
- Number of layers
- Weight decay

**Create a table showing your experiments and results.**

What combination worked best?

---

### ✅ Phase 4 Checkpoint

- [ ] Written comparison of architectures (before implementing)
- [ ] Implemented at least one improvement
- [ ] Hyperparameter tuning table with 10+ experiments
- [ ] Final best accuracy recorded

---

# Phase 5: Documentation and Reflection (1 hour)

## Task 5.1: Write a Report

Create a short report (1-2 pages) containing:

1. **Problem Statement:** What were you trying to solve?
2. **Approach:** What methods did you try?
3. **Results:** What accuracy did you achieve? Include a table.
4. **Key Insights:** What was the most surprising thing you learned?
5. **Future Work:** What would you try if you had more time?

---

## Task 5.2: Reflection Questions

Answer these questions honestly:

1. What was the hardest part of this project?
2. Where did you get stuck? How did you get unstuck?
3. What concept became clearer through implementation?
4. What would you do differently if starting over?

---

## 🏆 Project Complete!

Congratulations! You've completed the node classification project.

**Your deliverables:**
- [ ] Jupyter notebook with all code
- [ ] Written answers to all questions
- [ ] Comparison tables and visualizations
- [ ] Final report (1-2 pages)
- [ ] Best model achieving 80%+ accuracy

---

## 📚 Resources (Use Only When Truly Stuck)

- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [GCN Original Paper](https://arxiv.org/abs/1609.02907)
- [GAT Original Paper](https://arxiv.org/abs/1710.10903)

Remember: Struggling is part of learning! Try to solve problems yourself before looking things up.

---

**Next Project:** [Link Prediction Challenge →](../P2-Link-Prediction/)
