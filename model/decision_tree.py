# model/decision_tree.py
import os
import joblib
from model.utils import compute_metrics
from sklearn.tree import DecisionTreeClassifier

def train_dt(X_train, y_train):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_and_save_dt(model, X_test, y_test, out_dir):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_proba)
    joblib.dump(model, os.path.join(out_dir, "decision_tree_model.pkl"))
    return metrics
