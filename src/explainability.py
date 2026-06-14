import numpy as np
import pandas as pd
import shap
import joblib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config


def compute_shap_values(model, X_processed, feature_names):
    """
    Compute SHAP values for all accounts using the trained XGBoost model.
    Returns a DataFrame of SHAP values (one row per account, one col per feature).
    """
    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_processed)

    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    print(f"SHAP values computed for {len(shap_df)} accounts.")
    return shap_df, explainer


def get_top_drivers(shap_df, account_idx, top_n=5):
    """
    For a single account, return the top N features driving its risk score,
    with direction (pushing score up = towards mule, or down = towards normal).
    """
    row = shap_df.iloc[account_idx]
    top = row.abs().nlargest(top_n)
    drivers = []
    for feat in top.index:
        drivers.append({
            "feature": feat,
            "shap_value": round(row[feat], 4),
            "direction": "↑ risk" if row[feat] > 0 else "↓ risk"
        })
    return drivers


def plot_summary(shap_df, X_processed, feature_names, save_path=None):
    """Global SHAP summary plot — shows which features matter most overall."""
    save_path = save_path or (config.OUTPUTS_DIR / "shap_summary.png")
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure()
    shap.summary_plot(
        shap_df.values,
        X_processed,
        feature_names=feature_names,
        show=False,
        max_display=20
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP summary plot saved to {save_path}")


def save_shap_values(shap_df, path=None):
    path = path or (config.OUTPUTS_DIR / "shap_values.csv")
    shap_df.to_csv(path, index=False)
    print(f"SHAP values saved to {path}")