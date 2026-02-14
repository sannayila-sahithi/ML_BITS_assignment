# model/naive_bayes.py
import os
import joblib
from model.utils import compute_metrics
from sklearn.naive_bayes import GaussianNB

def train_nb(X_train, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

def evaluate_and_save_nb(model, X_test, y_test, out_dir):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_proba)
    joblib.dump(model, os.path.join(out_dir, "naive_bayes_model.pkl"))
    return metrics
