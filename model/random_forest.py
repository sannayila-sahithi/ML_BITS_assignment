# model/random_forest.py
import os
import joblib
from model.utils import compute_metrics
from sklearn.ensemble import RandomForestClassifier

def train_rf(X_train, y_train, n_estimators=100):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_and_save_rf(model, X_test, y_test, out_dir):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_proba)
    joblib.dump(model, os.path.join(out_dir, "random_forest_model.pkl"))
    return metrics
