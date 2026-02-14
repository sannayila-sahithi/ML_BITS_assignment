# model/utils.py
import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix, classification_report
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODEL_DIR = os.path.dirname(__file__)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def load_or_make_dataset():
    """
    Loads UCI Breast Cancer dataset via sklearn, writes a CSV for reproducibility,
    and returns X, y DataFrames. Target encoded: M=1, B=0.
    """
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    # sklearn target: 0=malignant, 1=benign; we convert to M=1, B=0 for clarity:
    # Map: malignant -> 1 (positive), benign -> 0 (negative)
    y = pd.Series(np.where(data.target == 0, 1, 0), name="diagnosis")
    # Save a full CSV copy (features + target) for transparency
    full = X.copy()
    # Convert target to original letters for CSV compatibility with common versions (M/B)
    full["diagnosis"] = y.map({1: "M", 0: "B"})
    full_csv = os.path.join(DATA_DIR, "breast_cancer_full.csv")
    full.to_csv(full_csv, index=False)
    return X, y  # y is numeric (1/0) in memory

def train_test_split_stratified(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def eda_report(X, y):
    """
    Generates a minimal EDA report (printable), returns dict for README if needed.
    """
    report = {}
    report["shape"] = {"rows": int(X.shape[0]), "features": int(X.shape[1])}
    report["target_balance"] = y.value_counts().to_dict()
    report["missing_values_per_column"] = X.isna().sum().to_dict()
    desc = X.describe().T
    # Save a short EDA CSV
    desc_csv = os.path.join(DATA_DIR, "eda_summary.csv")
    desc.to_csv(desc_csv)
    return report

def prepare_scaler(X_train):
    scaler = StandardScaler()
    scaler.fit(X_train)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    return scaler

def load_scaler():
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    return joblib.load(scaler_path)

def compute_metrics(y_true, y_pred, y_proba=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    if y_proba is not None:
        auc = roc_auc_score(y_true, y_proba)
    else:
        auc = roc_auc_score(y_true, y_pred)  # fallback (less ideal)
    return {
        "Accuracy": acc, "AUC": auc, "Precision": prec, "Recall": rec, "F1": f1, "MCC": mcc
    }

def persist_expected_schema(X_columns):
    """
    Save expected feature schema so the app can validate uploaded CSV columns.
    """
    schema_path = os.path.join(DATA_DIR, "expected_schema.json")
    json.dump({"features": list(X_columns)}, open(schema_path, "w"))

def save_test_20_percent_csv(X_test, y_test, filename="test_20_percent.csv"):
    """
    Save the FULL 20% held-out test split to data/test_20_percent.csv
    Format: all feature columns + diagnosis column (M/B)
    """
    out = X_test.copy()
    out["diagnosis"] = y_test.map({1: "M", 0: "B"})
    out_path = os.path.join(DATA_DIR, filename)
    out.to_csv(out_path, index=False)
    return out_path

def confusion_and_report(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["Benign(0)", "Malignant(1)"])
    return cm, report
