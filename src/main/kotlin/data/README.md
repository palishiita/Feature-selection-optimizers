# Dataset Descriptions for Feature Selection Benchmarking

This document provides a comprehensive description of the three benchmark datasets used in the **Feature-Selection-Optimizers** project. These datasets represent a wide range of dimensional complexities—low, medium, and high—and are used to evaluate and compare the effectiveness of feature selection algorithms such as Binary GWO and Binary TLBO.

---

## 1. Breast Cancer Wisconsin (Diagnostic) Dataset

- **Source**: UCI Machine Learning Repository
- **Link**: [Breast Cancer Wisconsin Diagnostic Dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- **Samples**: 569
- **Features**: 30 real-valued input features
- **Target Classes**: 2 (Malignant, Benign)

### Description:
This dataset consists of features computed from digitized images of fine needle aspirate (FNA) of breast masses. The features describe characteristics of the cell nuclei present in the image. It is a widely used dataset for binary classification tasks in medical diagnosis.

### Features (Grouped by Type):
Each measurement has been computed in three forms—mean, standard error (SE), and worst (largest value):

- **Radius**: Mean distance from center to points on the perimeter
- **Texture**: Standard deviation of gray-scale values
- **Perimeter**
- **Area**
- **Smoothness**: Local variation in radius lengths
- **Compactness**: (Perimeter² / Area) - 1.0
- **Concavity**: Severity of concave portions of the contour
- **Concave points**: Number of concave portions of the contour
- **Symmetry**
- **Fractal dimension**: "Coastline approximation" – 1

> Total: 30 numerical features derived from the 10 characteristics above × 3 variations (mean, SE, worst)

### Target Variable:
- `M`: Malignant (Cancerous)
- `B`: Benign (Non-cancerous)
> **Location**: Column 2 of the dataset (i.e., second column in each row)

---

## 2. Arrhythmia Dataset

- **Source**: UCI Machine Learning Repository
- **Link**: [Arrhythmia Dataset](https://archive.ics.uci.edu/dataset/5/arrhythmia)
- **Samples**: 452
- **Features**: 279
- **Target Classes**: 16 (Multiclass Classification)

### Description:
The arrhythmia dataset contains ECG (electrocardiogram) signal data collected from patients. Each record represents one patient’s ECG analysis, and the features capture detailed medical parameters such as wave intervals, amplitudes, heart rates, and diagnostic codes.

This dataset is particularly useful for testing classification models on **imbalanced classes** and medium-to-high dimensional feature spaces.

### Features:
- P-wave, QRS complex, QT interval, PR interval, ST segment
- Heart rate, QRS axis, T axis, P axis
- Amplitude and duration of each wave
- Various diagnostic attributes coded numerically

> Note: Some features have missing values (represented by `?`), requiring preprocessing before modeling.

### Target Variable:
- Class `1`: Normal
- Classes `2–16`: Various arrhythmias and abnormalities (e.g., supraventricular premature beats, left bundle branch block, etc.)
> **Location**: Last column in each row

---

## 3. Leukemia Dataset

- **Source**: scikit-feature repository
- **Link**: [Leukemia Dataset (GSE9476)](https://github.com/jundongl/scikit-feature/blob/master/skfeature/data/leukemia.mat)
- **Samples**: 72
- **Features**: 7070
- **Target Classes**: 2 (AML, ALL)

### Description:
This high-dimensional biomedical dataset is used for binary classification of leukemia subtypes. Each instance represents a patient, with features corresponding to gene expression levels measured through microarray experiments.

### Features:
- 7070 continuous numerical features representing gene expression levels across different genes.
- Data is commonly used to benchmark algorithms that must handle the curse of dimensionality.

### Target Variable:
- Class `0`: Acute Myeloid Leukemia (AML)
- Class `1`: Acute Lymphoblastic Leukemia (ALL)
> **Location**: Stored in the `Y` variable of the `leukemia.mat` file

---

## Summary Table

| Dataset               | # Samples | # Features | Target Type     | Classes                           | Target Variable Location              |
|------------------------|------------|-------------|------------------|------------------------------------|----------------------------------------|
| BCW (Breast Cancer)    | 569        | 30          | Binary           | Malignant, Benign                  | Second column of the dataset           |
| Arrhythmia             | 452        | 279         | Multiclass (16)  | Normal, 15 Arrhythmia types        | Last column of the dataset             |
| Leukemia               | 72         | 7070        | Binary           | AML, ALL                           | `Y` variable in `leukemia.mat` file    |

---

## Notes:
- All datasets have been preprocessed to handle missing values and normalization where applicable.
- Feature selection algorithms are evaluated based on their ability to reduce the number of features while retaining or improving classification accuracy on these datasets.