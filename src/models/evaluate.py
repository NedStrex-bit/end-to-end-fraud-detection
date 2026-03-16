"""Baseline evaluation helpers."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib")))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


matplotlib.use("Agg")
logging.getLogger("matplotlib").setLevel(logging.WARNING)

PLOTS_DIR = Path("artifacts/plots")


def ensure_plots_dir() -> None:
    """Create the plots directory."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def save_current_figure(filename: str) -> str:
    """Save and close the current matplotlib figure."""
    ensure_plots_dir()
    path = PLOTS_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(path)


def has_both_classes(y_true: np.ndarray) -> bool:
    """Return True if both classes are present."""
    return len(np.unique(y_true)) > 1


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, Any]:
    """Compute classification metrics with graceful degradation for single-class splits."""
    metrics: dict[str, Any] = {
        "positive_rate": float(np.mean(y_true)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "log_loss": float(log_loss(y_true, np.column_stack([1 - y_score, y_score]), labels=[0, 1])),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            zero_division=0,
            output_dict=True,
        ),
        "roc_auc": None,
        "pr_auc": None,
    }

    if has_both_classes(y_true):
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
        operating_point = compute_operating_point_metrics(y_true, y_score)
        metrics.update(operating_point)
    else:
        metrics["recall_at_precision_90"] = None
        metrics["precision_at_recall_80"] = None

    return metrics


def compute_operating_point_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float | None]:
    """Compute threshold-based metrics for practical operating points."""
    if not has_both_classes(y_true):
        return {
            "recall_at_precision_90": None,
            "precision_at_recall_80": None,
        }

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    recall_at_precision_90 = None
    precision_at_recall_80 = None

    valid_recalls = recall[precision >= 0.90]
    if len(valid_recalls) > 0:
        recall_at_precision_90 = float(np.max(valid_recalls))

    valid_precisions = precision[recall >= 0.80]
    if len(valid_precisions) > 0:
        precision_at_recall_80 = float(np.max(valid_precisions))

    return {
        "recall_at_precision_90": recall_at_precision_90,
        "precision_at_recall_80": precision_at_recall_80,
    }


def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray, filename: str) -> str | None:
    """Plot ROC curve when both classes are present."""
    if not has_both_classes(y_true):
        return None

    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    return save_current_figure(filename)


def plot_pr_curve(y_true: np.ndarray, y_score: np.ndarray, filename: str) -> str | None:
    """Plot precision-recall curve when both classes are present."""
    if not has_both_classes(y_true):
        return None

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label="PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    return save_current_figure(filename)
