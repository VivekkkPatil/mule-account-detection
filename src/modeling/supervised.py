import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    precision_recall_curve, auc, roc_auc_score,
    precision_score, recall_score, f1_score, confusion_matrix
)
from xgboost import XGBClassifier
import joblib

from src import config


def train_cv(X_processed, y, n_folds=None):
    """
    Train XGBoost with stratified K-fold CV.
    Returns out-of-fold predictions (probabilities) for the whole dataset,
    plus the list of trained fold models.
    """
    n_folds = n_folds or config.N_FOLDS
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.RANDOM_STATE)

    oof_preds = np.zeros(len(y))
    models = []

    scale_pos_weight = (y == 0).sum() / (y == 1).sum()

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_processed, y)):
        X_train, X_val = X_processed[train_idx], X_processed[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            random_state=config.RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        models.append(model)

        print(f"Fold {fold + 1}/{n_folds} done.")

    return oof_preds, models


def evaluate_predictions(y_true, y_pred_proba, threshold=0.5):
    """Print imbalance-aware evaluation metrics."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    roc_auc = roc_auc_score(y_true, y_pred_proba)

    y_pred = (y_pred_proba >= threshold).astype(int)

    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Precision @ {threshold}: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall @ {threshold}: {recall_score(y_true, y_pred):.4f}")
    print(f"F1 @ {threshold}: {f1_score(y_true, y_pred):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    return {"pr_auc": pr_auc, "roc_auc": roc_auc}


def train_final_model(X_processed, y):
    """Train one final model on ALL data (for deployment/SHAP/inference)."""
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=config.RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=-1,
    )
    model.fit(X_processed, y)
    return model


def save_model(model, path=None):
    path = path or (config.MODELS_DIR / "xgb_model.pkl")
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Saved model to {path}")