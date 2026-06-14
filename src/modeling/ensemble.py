import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config


def blend_scores(xgb_proba, anomaly_scores, xgb_weight=0.8):
    """
    Blend XGBoost probability and Isolation Forest anomaly score
    into a single 0-1 risk score per account.
    Higher = more likely a mule.
    """
    iso_weight = 1 - xgb_weight
    risk_scores = (xgb_weight * xgb_proba) + (iso_weight * anomaly_scores)
    return risk_scores


def assign_risk_tier(risk_scores):
    """
    Assign human-readable risk tier to each account.
    Green / Amber / Red based on risk score thresholds.
    """
    tiers = pd.cut(
        risk_scores,
        bins=[-np.inf, 0.3, 0.6, np.inf],
        labels=["Green", "Amber", "Red"]
    )
    return tiers


def build_risk_table(X_raw, y, oof_preds, anomaly_scores, feature_names_selected):
    """
    Build the final risk table: one row per account with
    risk score, tier, true label, and top raw feature values.
    """
    risk_scores = blend_scores(oof_preds, anomaly_scores)
    tiers = assign_risk_tier(risk_scores)

    risk_df = pd.DataFrame({
        "account_idx": range(len(y)),
        "true_label": y.values,
        "xgb_proba": oof_preds,
        "anomaly_score": anomaly_scores,
        "risk_score": risk_scores,
        "risk_tier": tiers,
    })

    return risk_df


def plot_risk_distribution(risk_df, save_path=None):
    """
    Histogram of risk scores, split by true label.
    Shows how well mules and normals separate on the blended score.
    """
    save_path = save_path or (config.OUTPUTS_DIR / "risk_score_distribution.png")
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(
        risk_df[risk_df["true_label"] == 0]["risk_score"],
        bins=50, alpha=0.6, color="steelblue", label="Normal"
    )
    ax.hist(
        risk_df[risk_df["true_label"] == 1]["risk_score"],
        bins=50, alpha=0.8, color="crimson", label="Mule"
    )

    ax.axvline(0.3, color="orange", linestyle="--", linewidth=1.5, label="Amber threshold (0.3)")
    ax.axvline(0.6, color="red", linestyle="--", linewidth=1.5, label="Red threshold (0.6)")

    ax.set_xlabel("Risk Score")
    ax.set_ylabel("Number of Accounts")
    ax.set_title("Risk Score Distribution: Mule vs Normal Accounts")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Risk distribution plot saved to {save_path}")


def save_risk_scores(risk_df, path=None):
    path = path or config.RISK_SCORES_PATH
    risk_df.to_csv(path, index=False)
    print(f"Risk scores saved to {path}")