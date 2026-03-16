"""Optional anomaly detection experiments for fraud detection."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib")))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.models.evaluate import evaluate_predictions


logging.getLogger("matplotlib").setLevel(logging.WARNING)

PLOTS_DIR = Path("artifacts/plots")


def build_isolation_forest() -> IsolationForest:
    """Create a simple reproducible anomaly detector."""
    return IsolationForest(
        n_estimators=200,
        contamination="auto",
        max_samples="auto",
        random_state=42,
    )


def fit_anomaly_model(train_x: pd.DataFrame, train_y: pd.Series | None = None) -> tuple[IsolationForest, list[str]]:
    """Fit anomaly model, preferring benign-only training when labels are available."""
    notes: list[str] = []
    fit_frame = train_x

    if train_y is not None and train_y.nunique(dropna=True) > 1 and (train_y == 0).any():
        fit_frame = train_x.loc[train_y == 0].copy()
        notes.append("Isolation Forest обучен на non-fraud train subset, что делает эксперимент semi-unsupervised.")
    else:
        notes.append("Isolation Forest обучен на всём train split, потому что отдельный benign-only subset недоступен.")

    model = build_isolation_forest()
    model.fit(fit_frame)
    return model, notes


def anomaly_score(model: IsolationForest, feature_frame: pd.DataFrame) -> np.ndarray:
    """Return normalized anomaly scores where higher means more anomalous."""
    raw_score = -model.score_samples(feature_frame)
    min_value = float(np.min(raw_score))
    max_value = float(np.max(raw_score))
    if np.isclose(min_value, max_value):
        return np.full_like(raw_score, 0.5, dtype="float64")
    return (raw_score - min_value) / (max_value - min_value)


def threshold_from_train_scores(train_scores: np.ndarray, quantile: float = 0.95) -> float:
    """Pick a simple anomaly threshold from train score distribution."""
    return float(np.quantile(train_scores, quantile))


def evaluate_score_model(
    y_true: np.ndarray,
    score: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Evaluate a score-based detector with a fixed threshold."""
    prediction = (score >= threshold).astype(int)
    metrics = evaluate_predictions(y_true=y_true, y_pred=prediction, y_score=score)
    metrics["decision_threshold"] = float(threshold)
    return metrics


def combine_scores(supervised_score: np.ndarray, anomaly_score_values: np.ndarray, alpha: float = 0.7) -> np.ndarray:
    """Build a lightweight ensemble score."""
    return alpha * supervised_score + (1.0 - alpha) * anomaly_score_values


def save_current_figure(filename: str) -> str:
    """Save the current matplotlib figure."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(path)


def plot_score_distribution(
    score_frame: pd.DataFrame,
    score_column: str,
    filename: str,
    title: str,
) -> str:
    """Plot score distributions by target class when possible."""
    plt.figure(figsize=(7, 4))
    negative = score_frame.loc[score_frame["target"] == 0, score_column].to_numpy()
    positive = score_frame.loc[score_frame["target"] == 1, score_column].to_numpy()

    plotted = False
    if len(negative) > 0:
        plt.hist(negative, bins=min(10, max(3, len(negative))), alpha=0.6, label="target=0")
        plotted = True
    if len(positive) > 0:
        plt.hist(positive, bins=min(10, max(3, len(positive))), alpha=0.6, label="target=1")
        plotted = True

    if plotted:
        plt.legend()
    plt.xlabel(score_column)
    plt.ylabel("Count")
    plt.title(title)
    return save_current_figure(filename)


def comparison_rows_from_metrics(metrics_bundle: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Flatten model metrics into a comparison table."""
    rows: list[dict[str, Any]] = []
    for model_name, metrics in metrics_bundle.items():
        rows.append(
            {
                "model": model_name,
                "roc_auc": metrics.get("roc_auc"),
                "pr_auc": metrics.get("pr_auc"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1"),
                "log_loss": metrics.get("log_loss"),
                "decision_threshold": metrics.get("decision_threshold"),
                "confusion_matrix": str(metrics.get("confusion_matrix")),
            }
        )
    return pd.DataFrame(rows)
