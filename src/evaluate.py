import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    precision_recall_curve, auc, roc_auc_score,
    confusion_matrix, RocCurveDisplay
)

from src import config


def plot_precision_recall_curve(y_true, y_pred_proba, save_path=None):
    """Plot and save the Precision-Recall curve."""
    save_path = save_path or (config.OUTPUTS_DIR / "precision_recall_curve.png")
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(recall, precision, color="darkorange", lw=2,
            label=f"PR Curve (AUC = {pr_auc:.4f})")
    ax.axhline(y=y_true.mean(), color="navy", linestyle="--",
               label=f"Baseline (random) = {y_true.mean():.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Mule Account Detection")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"PR curve saved to {save_path}")
    return pr_auc


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot and save a styled confusion matrix heatmap."""
    save_path = save_path or (config.OUTPUTS_DIR / "confusion_matrix.png")
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Reds",
        xticklabels=["Normal", "Mule"],
        yticklabels=["Normal", "Mule"],
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Mule Account Detection")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_roc_curve(y_true, y_pred_proba, save_path=None):
    """Plot and save ROC curve."""
    save_path = save_path or (config.OUTPUTS_DIR / "roc_curve.png")
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    RocCurveDisplay.from_predictions(y_true, y_pred_proba, ax=ax,
                                      color="darkorange")
    ax.plot([0, 1], [0, 1], "k--", label="Random baseline")
    ax.set_title("ROC Curve — Mule Account Detection")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"ROC curve saved to {save_path}")


def print_business_impact(risk_df):
    """Print the business impact summary."""
    total = len(risk_df)
    flagged = risk_df[risk_df["risk_tier"].isin(["Red", "Amber"])]
    mules_caught = flagged[flagged["true_label"] == 1].shape[0]
    total_mules = risk_df["true_label"].sum()
    false_alarms = flagged[flagged["true_label"] == 0].shape[0]

    print("=" * 50)
    print("BUSINESS IMPACT SUMMARY")
    print("=" * 50)
    print(f"Total accounts reviewed:      {total}")
    print(f"Accounts flagged for review:  {len(flagged)} "
          f"({len(flagged)/total*100:.1f}% of all accounts)")
    print(f"Mules caught (Red + Amber):   {mules_caught}/{total_mules} "
          f"({mules_caught/total_mules*100:.1f}%)")
    print(f"False alarms:                 {false_alarms} "
          f"({false_alarms/len(flagged)*100:.1f}% of flagged)")
    print(f"Workload reduction:           "
          f"{(1 - len(flagged)/total)*100:.1f}%")
    print("=" * 50)