# app.py
import os
import json
import io

import streamlit as st
import pandas as pd
import numpy as np
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix, classification_report
)

# --- Paths ---
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")

# --- Page config ---
st.set_page_config(page_title="ML BITS Assignment – Model Evaluation", layout="wide")

st.title("ML BITS Assignment – Interactive Model Evaluation")
st.markdown("""
Upload a **test dataset (CSV)** with the same feature columns and a `diagnosis` column (`M`/`B`).  
Select a model to evaluate. The app will display metrics and confusion matrix / classification report.

> Note: Free Streamlit tier has limited memory, so only **test** data upload is supported (training is done offline).""")

# --- Download test CSV ---
test20_path = os.path.join(DATA_DIR, "test_20_percent.csv")

if os.path.exists(test20_path):
    with open(test20_path, "rb") as f:
        st.download_button(
            label="⬇️ Download 20% Test Data CSV",
            data=f,
            file_name="test_20_percent.csv",
            mime="text/csv"
        )
else:
    st.warning("20% test CSV not found. Run training: python -m model.train_all")

# --- Sidebar: Upload & Model selection ---
st.sidebar.header("1) Upload Test CSV")
uploaded = st.sidebar.file_uploader("Choose CSV file", type=["csv"])

st.sidebar.header("2) Select Model")
model_names = [
    "Logistic Regression", "Decision Tree", "KNN",
    "Naive Bayes", "Random Forest", "XGBoost"
]
choice = st.sidebar.selectbox("Model", model_names)

# Map to model files
MODEL_FILES = {
    "Logistic Regression": "logistic_model.pkl",
    "Decision Tree": "decision_tree_model.pkl",
    "KNN": "knn_model.pkl",
    "Naive Bayes": "naive_bayes_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "XGBoost": "xgboost_model.pkl"
}

# Load scaler for LR & KNN
scaler = None
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)

# Load expected schema
schema_path = os.path.join(DATA_DIR, "expected_schema.json")
expected_features = None
if os.path.exists(schema_path):
    expected_features = json.load(open(schema_path))["features"]

# --- Helper: validate & preprocess ---
def prepare_data(df: pd.DataFrame):
    # Expect 'diagnosis' column with M/B
    if "diagnosis" not in df.columns:
        st.error("Uploaded CSV must include a 'diagnosis' column with values 'M' or 'B'.")
        st.stop()
    # Split features/target
    y_true = df["diagnosis"].map({"M": 1, "B": 0})
    X = df.drop(columns=["diagnosis"])

    # Schema check
    if expected_features is not None:
        missing = [c for c in expected_features if c not in X.columns]
        extra = [c for c in X.columns if c not in expected_features]
        if missing:
            st.error(f"Missing required feature columns: {missing}")
            st.stop()
        if extra:
            st.warning(f"Extra columns ignored: {extra}")
            X = X[expected_features]
    return X, y_true

# --- Main: when file is uploaded & model chosen ---
if uploaded is not None:
    try:
        test_df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    st.subheader("Preview of Uploaded Data")
    st.dataframe(test_df.head())

    X_test, y_true = prepare_data(test_df)

    # Load chosen model
    model_file = os.path.join(MODEL_DIR, MODEL_FILES[choice])
    if not os.path.exists(model_file):
        st.error(f"Model file not found: {MODEL_FILES[choice]}. Please run training.")
        st.stop()
    model = joblib.load(model_file)

    # Scale only for LR & KNN
    if choice in ["Logistic Regression", "KNN"]:
        if scaler is None:
            st.error("Scaler not found. Please run training to generate scaler.pkl.")
            st.stop()
        X_scaled = scaler.transform(X_test)
        X_input = pd.DataFrame(X_scaled, columns=X_test.columns)
    else:
        X_input = X_test

    # Predictions & probabilities
    preds = model.predict(X_input)
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_input)[:, 1]

    # Metrics
    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds)
    rec = recall_score(y_true, preds)
    f1 = f1_score(y_true, preds)
    mcc = matthews_corrcoef(y_true, preds)
    auc = roc_auc_score(y_true, proba if proba is not None else preds)

    # Display metrics
    st.subheader(f"Performance Metrics – {choice}")
    mdf = pd.DataFrame({
        "Metric": ["Accuracy","AUC","Precision","Recall","F1","MCC"],
        "Value": [acc, auc, prec, rec, f1, mcc]
    })
    mdf["Value"] = mdf["Value"].map(lambda x: f"{x:.4f}")
    st.table(mdf)

    # Confusion matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, preds)
    fig, ax = plt.subplots(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticklabels(["Benign (0)", "Malignant (1)"])
    ax.set_yticklabels(["Benign (0)", "Malignant (1)"])
    st.pyplot(fig)

    # Classification report
    st.subheader("Classification Report")
    report = classification_report(y_true, preds, target_names=["Benign(0)","Malignant(1)"])
    st.text(report)
else:
    st.info("Upload a test CSV to begin.")
