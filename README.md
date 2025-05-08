# Feature-Selection-Optimizers

## Overview
This project investigates optimization algorithms for efficient feature selection in machine learning models. Specifically, it compares nature-inspired metaheuristic algorithms—focusing on those that mimic the social behaviors of animals versus those that simulate human intelligence and learning processes.

The goal is to evaluate and compare the performance of two distinct algorithms in selecting the most relevant features to improve classification accuracy and reduce model complexity.

So, the problem statement is: "How can we search the space of possible feature subsets to minimize classification error and reduce feature count?"

---

## Why Feature Selection is Important in Machine Learning

Feature selection is a crucial step in the machine learning pipeline that involves identifying the most relevant features (variables) in a dataset. Selecting the right subset of features can lead to several key benefits:

### 1. Improves Model Accuracy
Irrelevant or redundant features can negatively affect model performance. By removing noise and focusing only on the most informative inputs, feature selection can significantly boost predictive accuracy.

### 2. Reduces Overfitting
When too many features are used—especially on small datasets—the model can overfit, meaning it performs well on training data but poorly on unseen data. Feature selection helps improve generalization.

### 3. Decreases Computational Cost
Fewer features mean less data to process, which leads to faster model training and inference. This is especially valuable in real-time systems or large-scale applications.

### 4. Enhances Model Interpretability
Simpler models with fewer features are easier to understand and interpret—especially critical in sensitive domains like healthcare, finance, or legal systems.

### 5. Improves Data Quality
Feature selection can expose irrelevant, noisy, or highly correlated features, helping improve the quality of the dataset itself.

### 6. Essential in High-Dimensional Data
In fields like genomics, text mining, or image processing, the number of features can be in the thousands. Feature selection becomes necessary to prevent the “curse of dimensionality.”

---

## Selected Optimizers

### Grey Wolf Optimizer (GWO)
- Inspired by the leadership and hunting behavior of grey wolves.
- Models a hierarchical decision-making structure with alpha, beta, and delta wolves guiding the search.

### Teaching-Learning-Based Optimization (TLBO)
- Inspired by the educational dynamics between teachers and students in a classroom.
- Operates in two phases: the Teacher Phase (global search) and Learner Phase (local improvement).

We need binary versions of GWO and TLBO because feature selection is a discrete 0/1 problem, and binary adaptations allow the algorithms to explore the binary search space effectively by updating bit selections rather than continuous values.

---

## Fitness function

In feature selection, we have two goals:
Goal 1: high classification accuracy (low error rate)
Goal 2: small number of features (simpler, cheaper model)

But these goals are conflicting:
Selecting more features → may increase accuracy
Selecting fewer features → may reduce accuracy

We need to balance these goals → combine them into one fitness function.

It is defined as:
`Fitness = α × ErrorRate + β × (Number of Selected Features / Total Features)`
where:
- `α`: weight assigned to classification error (e.g., 0.9)
- `β`: weight assigned to feature count penalty (e.g., 0.1)
- `ErrorRate`: classification error on validation data
- `Number of Selected Features`: count of selected features (1s in the binary vector)
- `Total Features`: total number of features in the dataset

---

## Benchmark Classification Datasets

The following datasets are chosen to represent a range of dimensional complexities:

### Low Dimension
- Dataset: Breast Cancer Wisconsin  
- Features: 30  
- Classes: 2  
- [View Dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

### Medium Dimension
- Dataset: Arrhythmia  
- Features: 279  
- Classes: 16  
- [View Dataset](https://archive.ics.uci.edu/dataset/5/arrhythmia)

### High Dimension
- Dataset: Leukemia (Gene Expression)  
- Features: 7129  
- Classes: 2  
- [View Dataset](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE9476)

---

## Goals
- Evaluate the effectiveness of GWO and TLBO for feature selection across different dataset complexities.
- Compare based on:
  - Classification accuracy
  - Number of selected features
  - Computational efficiency
  - Stability across folds

---

## Papers
- https://www.mdpi.com/2076-3417/15/2/489
- https://ieeexplore.ieee.org/document/9108264
- https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cth2.12498
- https://www.sciencedirect.com/science/article/abs/pii/S0169743923001302
- https://arxiv.org/pdf/2402.11839 
---
