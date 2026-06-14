import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

from src import config


def train_isolation_forest(X_processed, contamination=None):
    """
    Train Isolation Forest on the full dataset.
    contamination = expected proportion of outliers (mules).
    We use the actual positive rate from our data.
    """
    contamination = contamination or round(81/9082, 4)  # 0.0089

    print(f"Training Isolation Forest (contamination={contamination})...")
    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    iso.fit(X_processed)
    print("Isolation Forest trained.")
    return iso


def get_anomaly_scores(iso, X_processed):
    """
    Returns anomaly scores normalized to 0-1 range.
    Higher score = more anomalous = more likely a mule.
    """
    # Raw scores from sklearn are negative (more negative = more anomalous)
    raw_scores = iso.decision_function(X_processed)

    # Flip and normalize to 0-1
    flipped = -raw_scores
    normalized = (flipped - flipped.min()) / (flipped.max() - flipped.min())
    return normalized


def save_iso_model(iso, path=None):
    path = path or (config.MODELS_DIR / "isolation_forest.pkl")
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(iso, path)
    print(f"Saved Isolation Forest to {path}")