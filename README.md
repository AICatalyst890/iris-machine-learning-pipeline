# Iris Flower Classification Pipeline

An end-to-end Machine Learning workflow built inside a Jupyter Notebook to classify Iris flower species (Setosa, Versicolor, and Virginica) using morphological measurements (sepal length, sepal width, petal length, and petal width).

This project demonstrates core data science competencies: splitting datasets, training multiple classification algorithms, looping through models dynamically, and evaluating model performance using comprehensive classification metrics.

## Project Structure

```text
Iris flower/
├── .gitignore
├── Iris.csv           # Input dataset for the ML pipeline
├── README.md          # Project documentation
├── iris.ipynb         # Main Jupyter Notebook containing the ML pipeline
└── requirements.txt   # Project dependencies

```

## Machine Learning Pipeline

### 1. Data Processing & Preparation

* **Feature Extraction:** Utilizes the classic Iris dataset features to map input measurements to target species labels.
* **Data Splitting:** Uses `train_test_split` from `sklearn.model_selection` to separate the dataset into dedicated training and testing sets, ensuring unbiased model evaluation.

### 2. Model Training

The pipeline evaluates three distinct classification algorithms side-by-side to compare their baseline performance:

* **Logistic Regression:** A linear model optimized with a maximum iteration limit (`max_iter=1000`) for robust convergence.
* **K-Nearest Neighbors (KNN):** A distance-based classifier configured with $k=5$ neighbors using the Minkowski metric.
* **Naive Bayes (Gaussian NB):** A probabilistic classifier that assumes features follow a normal distribution.

### 3. Evaluation Metrics

Models are dynamically scored against the unseen test set using four key evaluation metrics (calculated with a `macro` average to account for class balance):

* **Accuracy:** Overall percentage of correctly predicted samples.
* **Precision:** The ability of the classifier not to label a negative sample as positive.
* **Recall:** The ability of the classifier to find all positive samples.
* **F1-Score:** The harmonic mean of precision and recall.
* **Confusion Matrix:** A breakdown of true vs. predicted classifications to pinpoint exact misclassifications.

---

## Model Performance Summary

The evaluation loop generated the following benchmark results on the test split:

| Classifier Model | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Performance Note |
| --- | --- | --- | --- | --- | --- |
| **Logistic Regression** | **100%** (`1.0`) | `1.0` | `1.0` | `1.0` | **Perfect classification**; zero errors. |
| **Gaussian Naive Bayes** | **98.6%** (`0.9867`) | `0.9861` | `0.9855` | `0.9855` | Misclassified only 1 sample. |
| **K-Nearest Neighbors ($k=5$)** | **94.6%** (`0.9467`) | `0.9506` | `0.9420` | `0.9416` | Misclassified 4 samples. |

### Confusion Matrix Insights

* **Logistic Regression** achieved a flawless diagonal matrix: `[[29, 0, 0], [0, 23, 0], [0, 0, 23]]`.
* **Naive Bayes** struggled with just 1 sample, mistakenly predicting a class 1 instead of class 2.
* **KNN** encountered slightly more friction in the boundary zones, misclassifying 4 instances of class 2 as class 1.

---

## How to Run

1. Open your terminal and navigate to the project folder.
2. Launch your Jupyter environment:
```bash
jupyter lab

```


3. Open `iris.ipynb`.
4. Run all cells sequentially (`Run -> Run All Cells`) to train the models, generate predictions, and view the comparative performance metrics.

```