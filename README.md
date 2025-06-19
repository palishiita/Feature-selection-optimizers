# Feature-Selection-Optimizers

## Overview
This project investigates optimization algorithms for efficient feature selection in machine learning models. Specifically, it compares nature-inspired metaheuristic algorithms—focusing on those that mimic the social behaviors of animals versus those that simulate human intelligence and learning processes.

The goal is to evaluate and compare the performance of two distinct algorithms in selecting the most relevant features to improve classification accuracy and reduce model complexity.

So, the problem statement is: "How can we search the space of possible feature subsets to minimize classification error and reduce feature count?"

---

## Why Feature Selection is Important in Machine Learning

Feature selection is a crucial step in the machine learning pipeline that involves identifying the most relevant features (variables) in a dataset.

- Improves Model Accuracy
- Reduces Overfitting
- Decreases Computational Cost
- Enhances Model Interpretability
- Improves Data Quality
- Essential in High-Dimensional Data

---

## Selected Optimizers

We need binary versions of GWO and TLBO because feature selection is a discrete 0/1 problem, and binary adaptations allow the algorithms to explore the binary search space effectively by updating bit selections rather than continuous values.

### Grey Wolf Optimizer (GWO)
Inspired by: The social hierarchy and cooperative hunting strategies of grey wolves in nature.

GWO mimics three main behaviors observed in grey wolves:
1. **Hierarchical Leadership**  
   The population is divided into:
    - Alpha (α): The best solution so far (leader).
    - Beta (β) and Delta (δ): Second- and third-best solutions; help guide the search.
    - Omega (ω): The rest of the wolves; follow the top three.  
      This hierarchy helps balance exploitation (refining current good solutions) and exploration (searching new areas).

2. **Encircling the Prey (Optimum)**
    - Wolves surround the prey from different directions.
    - In optimization terms, this means candidate solutions converge from different paths toward the current best solution.

3. **Hunting Behavior**
    - Wolves update their positions based on the positions of α, β, and δ wolves.
    - The new solution is a weighted average of the top three wolves, which helps avoid premature convergence and ensures diverse exploration.

#### Binary GWO for Feature Selection

![BGW](https://github.com/palishiita/Feature-selection-optimizers/blob/main/img/gwo.png)

---

### Teaching-Learning-Based Optimization (TLBO)
Inspired by: Teaching–learning processes in a classroom environment.

TLBO operates in two distinct phases:

1. **Teacher Phase (Global Search)**
    - The teacher is the best-performing solution in the population.
    - The teacher attempts to raise the mean knowledge (fitness) of the class by sharing knowledge with all students.
    - Each learner updates their solution vector by moving closer to the teacher’s knowledge level.
    - This simulates global learning and helps guide the population toward better regions in the search space.

2. **Learner Phase (Local Refinement)**
    - Learners interact and learn from one another.
    - A learner compares itself with a randomly chosen peer.
    - If the peer is better, the learner moves toward them.
    - This ensures peer-to-peer knowledge exchange and introduces local refinements to fine-tune solutions.

Together, these two phases:
- Encourage exploration via the teacher’s influence
- Encourage exploitation through collaborative learning among peers
- Provide a parameter-free optimization framework (no tuning of control parameters like crossover/mutation rates)

#### Binary TLBO for Feature Selection

![BTLBO](https://github.com/palishiita/Feature-selection-optimizers/blob/main/img/tlbo.png)

---

## Fitness function

In feature selection, we have two goals:
- Goal 1: high classification accuracy (low error rate)
- Goal 2: small number of features (simpler, cheaper model)

But these goals are conflicting:
- Selecting more features → may increase accuracy
- Selecting fewer features → may reduce accuracy

We need to balance these goals → combine them into one fitness function.

It is defined as:

![fitness](https://github.com/palishiita/Feature-selection-optimizers/blob/main/img/fitness.png)

where:
- `α`: weight assigned to classification error (e.g., 0.9)
- `β`: weight assigned to feature count penalty (e.g., 0.1)
- `ErrorRate`: classification error on validation data
- `Number of Selected Features`: count of selected features (1s in the binary vector)
- `Total Features`: total number of features in the dataset

The fitness function is used every time a new candidate solution (binary vector) is generated or evaluated. This happens:
1. At the start — to evaluate the initial population (random feature subsets).
2. During each iteration/epoch — whenever a solution is updated in either GWO or TLBO.
3. To rank solutions — so we can identify the best wolves (α, β, δ) or the best learner (teacher).
4. To guide selection — when choosing whether an updated solution is better and should replace the old one.

---

## Benchmark Classification Datasets

The following datasets are chosen to represent a range of dimensional complexities:

| Dataset                                                                 | Dimensionality | Features | Classes |
|-------------------------------------------------------------------------|----------------|----------|---------|
| [Breast Cancer Wisconsin](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) | Low            | 30       | 2       |
| [Arrhythmia](https://archive.ics.uci.edu/dataset/5/arrhythmia)         | Medium         | 279      | 16      |
| [Arcene](https://archive.ics.uci.edu/dataset/19/arcene)                | High           | 10,000   | 2       |

---

## Goals
- Evaluate the effectiveness of GWO and TLBO for feature selection across different dataset complexities.
- Compare based on:
    - Classification accuracy
    - Number of selected features
    - Computational efficiency
    - Stability across folds

---

## Workflow

Below is the typical sequence when running `Main.kt` on a dataset:

1. **Load the raw CSV** via `DataLoader`. Each loader normalizes numeric columns through `DataProcessor.minMaxNormalize` so all features fall in the range `[0, 1]`.
2. **Initialize the optimizer** (`GWO` or `TLBO`) with a population and the desired number of iterations.
3. **Iterative search**
    - For every wolf/learner, `evaluateDetailed` trains a temporary `RandomForestWrapper` on an internal 80/20 split of the *current* feature subset.
    - It returns a `FitnessResult` containing the penalized accuracy and the full set of metrics (accuracy, precision, recall, F1).
    - Statistics for the whole population—best (alpha) fitness, maximum, minimum and average fitness—are appended to `optimizer_log_file.csv` each iteration.
4. **Select the best mask** after the final iteration and apply it to the dataset to keep only the chosen features.
5. **Train the definitive model** on a fresh 80/20 split of the reduced dataset and evaluate the final metrics.
   Results are printed to the console.

`RandomForestWrapper` makes predictions in batches, so large datasets can be processed without exhausting memory.

---

## Papers
- https://www.mdpi.com/2076-3417/15/2/489
- https://ieeexplore.ieee.org/document/9108264
- https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cth2.12498
- https://www.sciencedirect.com/science/article/abs/pii/S0169743923001302
- https://arxiv.org/pdf/2402.11839