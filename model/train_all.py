# model/train_all.py
import os
import pandas as pd
from model.utils import (
    load_or_make_dataset, train_test_split_stratified, eda_report,
    prepare_scaler, load_scaler, persist_expected_schema, save_test_20_percent_csv
)

from model.logistic_regression import train_lr, evaluate_and_save_lr
from model.decision_tree import train_dt, evaluate_and_save_dt
from model.knn import train_knn, evaluate_and_save_knn
from model.naive_bayes import train_nb, evaluate_and_save_nb
from model.random_forest import train_rf, evaluate_and_save_rf
from model.xgboost_model import train_xgb, evaluate_and_save_xgb

THIS_DIR = os.path.dirname(__file__)
MODEL_DIR = THIS_DIR
DATA_DIR = os.path.join(os.path.dirname(THIS_DIR), "data")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

if __name__ == "__main__":
    # 1) Load data and split
    X, y = load_or_make_dataset()
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y, test_size=0.2, random_state=42)

    # 2) EDA: write short report and save schema
    report = eda_report(X, y)
    persist_expected_schema(X.columns)

    # 3) Prepare scaler for LR & KNN
    scaler = prepare_scaler(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled_df  = pd.DataFrame(X_test_scaled,  columns=X_test.columns,  index=X_test.index)

    # 4) Train + evaluate each model
    rows = []

    # Logistic Regression (scaled)
    lr = train_lr(X_train_scaled_df, y_train)
    m_lr = evaluate_and_save_lr(lr, X_test_scaled_df, y_test, MODEL_DIR)
    rows.append(["Logistic Regression", *[m_lr[k] for k in ["Accuracy","AUC","Precision","Recall","F1","MCC"]]])

    # Decision Tree (raw)
    dt = train_dt(X_train, y_train)
    m_dt = evaluate_and_save_dt(dt, X_test, y_test, MODEL_DIR)
    rows.append(["Decision Tree", *[m_dt[k] for k in ["Accuracy","AUC","Precision","Recall","F1","MCC"]]])

    # KNN (scaled)
    knn = train_knn(X_train_scaled_df, y_train, k=5)
    m_knn = evaluate_and_save_knn(knn, X_test_scaled_df, y_test, MODEL_DIR)
    rows.append(["KNN", *[m_knn[k] for k in ["Accuracy","AUC","Precision","Recall","F1","MCC"]]])

    # Naive Bayes (raw)
    nb = train_nb(X_train, y_train)
    m_nb = evaluate_and_save_nb(nb, X_test, y_test, MODEL_DIR)
    rows.append(["Naive Bayes", *[m_nb[k] for k in ["Accuracy","AUC","Precision","Recall","F1","MCC"]]])

    # Random Forest (raw)
    rf = train_rf(X_train, y_train, n_estimators=100)
    m_rf = evaluate_and_save_rf(rf, X_test, y_test, MODEL_DIR)
    rows.append(["Random Forest", *[m_rf[k] for k in ["Accuracy","AUC","Precision","Recall","F1","MCC"]]])

    # XGBoost (raw)
    xgb = train_xgb(X_train, y_train)
    m_xgb = evaluate_and_save_xgb(xgb, X_test, y_test, MODEL_DIR)
    rows.append(["XGBoost", *[m_xgb[k] for k in ["Accuracy","AUC","Precision","Recall","F1","MCC"]]])

    # 5) Persist metrics comparison table
    cols = ["Model","Accuracy","AUC","Precision","Recall","F1","MCC"]
    metrics_df = pd.DataFrame(rows, columns=cols)
    metrics_path = os.path.join(THIS_DIR, "metrics_comparison.csv")
    metrics_df.to_csv(metrics_path, index=False)

    # 6) Create test_20_percent.csv
    test20_path = save_test_20_percent_csv(X_test, y_test, "test_20_percent.csv")
    print("Saved:", metrics_path)
    print("Saved FULL 20% test split CSV:", test20_path)
    print("Saved models in:", MODEL_DIR)
