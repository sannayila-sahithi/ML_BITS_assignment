# model/knn.py
import os
import joblib
from model.utils import compute_metrics
from sklearn.neighbors import KNeighborsClassifier

def train_knn(X_train_scaled, y_train, k=5):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    return model

def evaluate_and_save_knn(model, X_test_scaled, y_test, out_dir):
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_proba)
    joblib.dump(model, os.path.join(out_dir, "knn_model.pkl"))
    return metrics
