# 📋 Prerequisites: What You Need Before Starting

> *"You don't need a PhD! Just some Python basics and a curious mind."*

---

## ✅ Quick Checklist

| Skill | Required | Level |
|-------|----------|-------|
| 🐍 Python | ✅ Yes | Basic |
| 📐 Math | ✅ Some | High school |
| 🧠 Machine Learning | 🟡 Helpful | Beginner |
| 🔥 PyTorch | 🟡 Helpful | Beginner |

**Don't have all of these? No worries! This guide will help.**

---

## 🐍 Python Basics

### What You Need to Know

```python
# 1. Variables and lists
friends = ["Alice", "Bob", "Charlie"]
nums = [1, 2, 3, 4, 5]

# 2. Loops
for friend in friends:
    print(f"Hi, {friend}!")

# 3. Functions
def greet(name):
    return f"Hello, {name}!"

# 4. Classes (just the basics!)
class Dog:
    def __init__(self, name):
        self.name = name
    
    def bark(self):
        return f"{self.name} says: Woof!"

buddy = Dog("Buddy")
print(buddy.bark())  # "Buddy says: Woof!"
```

### Quick Test 🧪

Can you understand this code?

```python
numbers = [1, 2, 3, 4, 5]
squared = [n ** 2 for n in numbers]
print(squared)  # What's the output?
```

<details>
<summary>Answer</summary>
[1, 4, 9, 16, 25] — It squares each number!
</details>

✅ **If you got it, you're ready for Python!**

---

## 📐 Math You'll Need

### 1. Vectors: Just Lists of Numbers!

```python
# A vector is just a list of numbers
me = [0.8, 0.2, 0.5]  # Maybe: [how much I like cats, dogs, birds]
you = [0.3, 0.9, 0.1]

# We can add them!
combined = [me[i] + you[i] for i in range(3)]
# [1.1, 1.1, 0.6]
```

### 2. Matrices: Grids of Numbers!

Think of it like a spreadsheet:

```python
import numpy as np

# This is a matrix (2 rows, 3 columns)
matrix = np.array([
    [1, 2, 3],  # Row 1
    [4, 5, 6],  # Row 2
])

# Access specific numbers
print(matrix[0, 2])  # Row 0, Column 2 = 3
```

### 3. Matrix Multiplication: The Magic Operation

Don't memorize the formula — just know these tools exist!

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Python does the hard work
result = A @ B  # or np.matmul(A, B)
print(result)
```

**GNNs use matrix multiplication to pass messages efficiently!**

---

## 🔥 PyTorch in 5 Minutes

PyTorch is like NumPy but:
1. 🚀 Can use GPUs (super fast!)
2. 🧠 Can automatically compute gradients (for learning!)

### Tensors = PyTorch's Arrays

```python
import torch

# Create a tensor (just like numpy array)
x = torch.tensor([1.0, 2.0, 3.0])
print(x)  # tensor([1., 2., 3.])

# Math works the same
y = x * 2
print(y)  # tensor([2., 4., 6.])

# Create random tensors
random_matrix = torch.randn(3, 4)  # 3x4 random numbers
```

### Neural Networks Made Easy

```python
import torch.nn as nn

# A simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 5)  # 10 inputs → 5 outputs
        self.layer2 = nn.Linear(5, 2)   # 5 inputs → 2 outputs
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))  # First layer + activation
        x = self.layer2(x)               # Second layer
        return x

# Use it!
model = SimpleNet()
input_data = torch.randn(1, 10)  # 1 sample, 10 features
output = model(input_data)
print(output.shape)  # torch.Size([1, 2])
```

### Training Loop Template

```python
# The classic training loop (you'll use this a lot!)
model = SimpleNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    # 1. Forward pass
    prediction = model(data)
    
    # 2. Calculate loss (how wrong are we?)
    loss = loss_function(prediction, target)
    
    # 3. Backward pass (calculate gradients)
    optimizer.zero_grad()
    loss.backward()
    
    # 4. Update weights
    optimizer.step()
```

---

## 📊 Machine Learning Refresher

### What is Machine Learning?

```
Traditional Programming:
  Data + Rules → Computer → Answers

Machine Learning:
  Data + Answers → Computer → Rules (learned!)
```

### The Three Types

| Type | What It Does | Example |
|------|-------------|---------|
| **Supervised** | Learn from labeled examples | Cat vs Dog photos |
| **Unsupervised** | Find patterns in data | Customer segments |
| **Reinforcement** | Learn by trial and error | Game AI |

**GNNs are usually Supervised Learning!**

### Key Terms

| Term | Plain English |
|------|--------------|
| **Training** | Teaching the model with examples |
| **Testing** | Checking if it actually learned |
| **Loss** | How wrong the model is |
| **Accuracy** | How often it's right |
| **Epoch** | One complete pass through training data |

---

## 🛠️ Setup Your Environment

### Option 1: The Quick Way (pip)

```bash
pip install torch torch-geometric networkx matplotlib jupyter
```

### Option 2: The Safe Way (conda)

```bash
conda create -n gnn-learning python=3.10
conda activate gnn-learning
pip install torch torch-geometric networkx matplotlib jupyter
```

### Verify Everything Works

```python
import torch
import torch_geometric
import networkx as nx
import matplotlib.pyplot as plt

print("🔥 PyTorch:", torch.__version__)
print("📊 PyTorch Geometric:", torch_geometric.__version__)
print("🌐 NetworkX:", nx.__version__)
print("✅ All good! Ready to learn GNNs!")
```

---

## 📚 Still Need Practice?

### Python
- 🆓 [Python for Beginners](https://www.python.org/about/gettingstarted/)
- 🎮 [Codecademy Python](https://www.codecademy.com/learn/learn-python-3)

### PyTorch
- 🆓 [PyTorch in 60 Minutes](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
- 📺 [PyTorch Beginner Series](https://www.youtube.com/playlist?list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4)

### Machine Learning
- 🆓 [Google ML Crash Course](https://developers.google.com/machine-learning/crash-course)
- 📺 [3Blue1Brown - Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) ⭐

---

## 🎯 Ready Check

Can you answer YES to these questions?

- [ ] I can write a Python function with parameters
- [ ] I understand that a matrix is a grid of numbers
- [ ] I know what `torch.tensor([1, 2, 3])` creates
- [ ] I know that ML learns patterns from data

**If you checked at least 3, you're ready!** 🚀

---

## 🚀 Let's Go!

**[Start Learning: Graph Theory Basics →](../01-Foundations/01-graph-theory-basics.md)** 🌐

---

*"Everyone starts somewhere. You're already ahead by being curious!"* ✨
