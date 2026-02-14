# model/xgboost_model.py
import os
import joblib
from model.utils import compute_metrics
from xgboost import XGBClassifier

def train_xgb(X_train, y_train, random_state=42):
    model = XGBClassifier(
        random_state=random_state,
        eval_metric='logloss',
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4
    )
    model.fit(X_train, y_train)
    return model

def evaluate_and_save_xgb(model, X_test, y_test, out_dir):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_proba)
    joblib.dump(model, os.path.join(out_dir, "xgboost_model.pkl"))
    return metrics
