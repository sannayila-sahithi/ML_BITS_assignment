# ML BITS Assignment– Interactive Model Evaluation

Goal: Implement 6 classification models on one dataset, evaluate them using multiple metrics, build an interactive Streamlit UI, and deploy it on Streamlit Community Cloud.


# 1) Problem Statement

This project implements and compares multiple machine learning classification models on a single dataset. The trained models are evaluated using Accuracy, AUC, Precision, Recall, F1-score, and MCC, and then exposed via an interactive Streamlit web application where users can upload a test CSV, select a model, and view performance metrics and a confusion matrix/classification report.

# 2) Dataset Description

Dataset: Breast Cancer Wisconsin (Diagnostic) Dataset (UCI) (loaded via 'sklearn.datasets') 

Why this dataset?
Instances: 569 (≥ 500 required)
Features: 30 numeric (≥ 12 required)
Task: Binary classification (Malignant vs Benign)
Clean and well-known benchmarking dataset

Target column in app/test CSV: 'diagnosis'
'M' Malignant (positive class, internally encoded as 1)
'B' Benign (negative class, internally encoded as 0)

Train/Test split: 80/20 stratified split (20% test split is saved as a downloadable CSV for the Streamlit app).

# 3) Models Implemented

All models are trained on the same dataset:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

Preprocessing note:
- StandardScaler is applied for Logistic Regression and KNN.
- Tree-based models (Decision Tree, Random Forest, XGBoost) are trained without scaling.

# 4) Evaluation Metrics

For each model, the following metrics are computed:
- Accuracy
- AUC (ROC-AUC)
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)


# 5) Model Comparison Table (Your Run Results)

The following results are computed on the 20% held-out test split.

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|----------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Random Forest | 0.9737 | 0.9929 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |
| XGBoost | 0.9649 | 0.9937 | 1.0000 | 0.9048 | 0.9500 | 0.9258 |


# 6) Observations (Model-wise)


| ML Model | Observation |
|----------|-------------|
| Logistic Regression | Strong AUC and balanced precision/recall indicate the dataset is close to linearly separable; performs reliably with scaling. |
| Decision Tree | Lower overall metrics than ensembles; may overfit without pruning/tuning; performance varies with split depth. |
| KNN | Performs well after scaling; sensitive to feature scaling and value of k; slightly lower recall than top ensembles. |
| Naive Bayes | Very high precision but lower recall; Gaussian independence assumptions may not fully match feature distributions. |
| Random Forest | Best overall performer; ensemble averaging improves generalization and robustness; high MCC indicates strong stability. |
| XGBoost | Competitive with Random Forest; strong AUC and MCC; effective at capturing feature interactions and non-linear decision boundaries. |


## 7) Streamlit App Features

The Streamlit app provides:
- Upload Test Dataset (CSV) (as per free-tier capacity guidance)
- Download button for the 20% test split CSV
- Model selection dropdown
- Display of evaluation metrics
- Confusion matrix and classification report


## 8) Project Structure

ML_BITS_assignment/
├─ app.py
├─ requirements.txt
├─ README.md
├─ data/
│  ├─ breast_cancer_full.csv
│  ├─ test_20_percent.csv
│  ├─ expected_schema.json
│  └─ eda_summary.csv
└─ model/
   ├─ __init__.py
   ├─ utils.py
   ├─ train_all.py
   ├─ logistic_regression.py
   ├─ decision_tree.py
   ├─ knn.py
   ├─ naive_bayes.py
   ├─ random_forest.py
   ├─ xgboost_model.py
   ├─ metrics_comparison.csv
   └─ *.pkl  (saved models + scaler)

# 9) How to Run Locally (BITS VM / Local)

# 9.1 Create and activate venv

python3 -m venv venv
source venv/bin/activate


# 9.2 Install dependencies

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 9.3 Train models and generate artifacts

python -m model.train_all

This generates:
- Trained model '.pkl' files in 'model/'
- 'model/metrics_comparison.csv'
- 'data/test_20_percent.csv' (20% test split for the app)

# 9.4 Run Streamlit app

python -m streamlit run app.py

# 10) Links (Fill these)

- GitHub Repository:
- Live Streamlit App: