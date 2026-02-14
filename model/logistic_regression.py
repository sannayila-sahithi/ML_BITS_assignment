# model/logistic_regression.py
import os
import joblib
from model.utils import compute_metrics

from sklearn.linear_model import LogisticRegression

def train_lr(X_train_scaled, y_train):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)   # works with DataFrame too
    return model

def evaluate_and_save_lr(model, X_test_scaled, y_test, out_dir):
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_proba)
    joblib.dump(model, os.path.join(out_dir, "logistic_model.pkl"))
    return metrics
