# Iris Flower Classification Pipeline

An end-to-end Machine Learning workflow built to classify Iris flower species (*Setosa*, *Versicolor*, and *Virginica*) using morphological measurements. This repository demonstrates core data science competencies: avoiding data leakage, training multiple classification algorithms, utilizing dynamic model-evaluation loops, and conducting fine-grained confusion matrix diagnostic analysis.

---

## 📌 Project Overview

Classifying species based on physical characteristics is a classic pattern-recognition problem. This project utilizes the morphometric measurements of Iris flowers—specifically sepal length, sepal width, petal length, and petal width—to train and benchmark three distinct classification models.

By stress-testing these models on an even 50/50 train-test split (75 samples each), this pipeline evaluates classifier behavior on decision boundaries, highlighting how different algorithms handle closely related classes.

---

## 📂 Project Structure

```text
Iris flower/
├── Iris.csv           # Input dataset for the ML pipeline
├── README.md          # Project documentation
├── iris.ipynb         # Main Jupyter Notebook containing the ML pipeline
└── requirements.txt   # Project dependencies

```

---

## ⚙️ Machine Learning Pipeline

### 1. Data Preprocessing & Leakage Prevention

* **Dropping ID Column:** The `Id` feature was dropped immediately (`X = df.drop(["Species", "Id"], axis=1)`) to prevent data leakage, ensuring the models do not falsely learn arbitrary index sequences.
* **Target Encoding:** Utilized `LabelEncoder` to cleanly convert text labels into numeric targets (`0` for Setosa, `1` for Versicolor, and `2` for Virginica).
* **Train-Test Split:** Implemented an even 50/50 partition (`test_size=0.5` with `random_state=42`) yielding exactly 75 training samples and 75 testing samples to benchmark model generalizability rigorously.

### 2. Model Training & Configuration

The pipeline evaluates three diverse classification strategies side-by-side:

* **Logistic Regression:** A linear classifier configured with `max_iter=1000` to ensure complete mathematical convergence.
* **Gaussian Naive Bayes:** A probabilistic classifier assuming features follow a Gaussian distribution.
* **K-Nearest Neighbors (KNN):** A distance-based classifier tuned with $k=5$ neighbors.

### 3. Dynamic Evaluation Loop

All models are dynamically fitted and validated through a single unified loop, calculating exact macro-averaged Precision, Recall, F1-Score, and multi-class Confusion Matrices.

---

## 📈 Model Performance Benchmark

The following evaluation metrics were generated directly from the unseen test split (75 total samples):

| Classifier Model | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) |
| --- | --- | --- | --- | --- |
| **Logistic Regression 🏆** | **100.00%** | **100.00%** | **100.00%** | **100.00%** |
| **Gaussian Naive Bayes** | 98.67% | 98.61% | 98.55% | 98.55% |
| **K-Nearest Neighbors ($k=5$)** | 94.67% | 95.06% | 94.20% | 94.16% |

---

## 🔍 Confusion Matrix Insights

Analyzing the confusion matrices reveals exactly how each algorithm handles decision boundaries:

* **Logistic Regression Matrix:**

$$\begin{bmatrix} 29 & 0 & 0 \\ 0 & 23 & 0 \\ 0 & 0 & 23 \end{bmatrix}$$


* *Insight:* Achieved a flawless diagonal. The linear decision boundaries perfectly separated all 3 classes on the test set.


* **Gaussian Naive Bayes Matrix:**

$$\begin{bmatrix} 29 & 0 & 0 \\ 0 & 23 & 0 \\ 0 & 1 & 22 \end{bmatrix}$$


* *Insight:* Highly robust, misclassifying only a single sample. It incorrectly predicted 1 *Virginica* (Class 2) as *Versicolor* (Class 1).


* **K-Nearest Neighbors ($k=5$) Matrix:**

$$\begin{bmatrix} 29 & 0 & 0 \\ 0 & 23 & 0 \\ 0 & 4 & 19 \end{bmatrix}$$


* *Insight:* Encountered more friction near the decision margins, misclassifying 4 *Virginica* (Class 2) samples as *Versicolor* (Class 1) due to neighborhood overlaps.



> **Key Takeaway:** All three models easily achieved 100% accuracy on *Iris-setosa*, which is linearly separable from the other two species. The marginal differences in performance stem entirely from the slight morphological overlap between *Iris-versicolor* and *Iris-virginica*.

---

## 🚀 How to Run the Project

1. **Clone this repository:**
```bash
git clone https://github.com/AICatalyst890/iris-machine-learning-pipeline.git
cd iris-machine-learning-pipeline

```

2. **Install dependencies:**
```bash
pip install -r requirements.txt

```

3. **Launch JupyterLab:**
```bash
jupyter lab iris.ipynb

```

4. **Run the pipeline:** Execute all notebook cells sequentially to train the models, view live outputs, and render performance plots.