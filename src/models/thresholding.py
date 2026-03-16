"""Threshold analysis helpers for fraud decisioning."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib")))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

logging.getLogger("matplotlib").setLevel(logging.WARNING)


PLOTS_DIR = Path("artifacts/plots")


@dataclass
class ThresholdMode:
    """Named threshold regime."""

    name: str
    threshold: float
    metrics: dict[str, Any]


def save_current_figure(filename: str) -> str:
    """Save the active matplotlib figure."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(path)


def evaluate_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    cost_false_negative: float,
    cost_false_positive: float,
) -> dict[str, Any]:
    """Evaluate a single decision threshold."""
    y_pred = (y_score >= threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    business_cost = fp * cost_false_positive + fn * cost_false_negative
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "tn": int(tn),
        "predicted_positive_count": int(np.sum(y_pred)),
        "business_cost": float(business_cost),
    }


def analyze_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray,
    cost_false_negative: float,
    cost_false_positive: float,
) -> pd.DataFrame:
    """Build a threshold evaluation table."""
    rows = [
        evaluate_threshold(
            y_true=y_true,
            y_score=y_score,
            threshold=float(threshold),
            cost_false_negative=cost_false_negative,
            cost_false_positive=cost_false_positive,
        )
        for threshold in thresholds
    ]
    return pd.DataFrame(rows)


def _pick_closest_threshold(metrics_table: pd.DataFrame, target_threshold: float) -> dict[str, Any]:
    """Return row metrics closest to a target threshold."""
    ordered = metrics_table.copy()
    ordered["distance_to_target"] = (ordered["threshold"] - target_threshold).abs()
    row = ordered.sort_values(["distance_to_target", "threshold"]).iloc[0]
    return row.drop(labels=["distance_to_target"]).to_dict()


def select_threshold_modes(metrics_table: pd.DataFrame) -> tuple[ThresholdMode, ThresholdMode, ThresholdMode]:
    """Choose conservative, balanced, and aggressive operating modes."""
    balanced_row = metrics_table.sort_values(
        ["business_cost", "f1", "threshold"],
        ascending=[True, False, False],
    ).iloc[0].to_dict()

    conservative_candidates = metrics_table[metrics_table["predicted_positive_count"] > 0]
    if conservative_candidates.empty:
        conservative_row = _pick_closest_threshold(metrics_table, target_threshold=0.80)
    else:
        conservative_row = conservative_candidates.sort_values(
            ["precision", "fp", "threshold"],
            ascending=[False, True, False],
        ).iloc[0].to_dict()

    aggressive_candidates = metrics_table[metrics_table["predicted_positive_count"] > 0]
    if aggressive_candidates.empty:
        aggressive_row = _pick_closest_threshold(metrics_table, target_threshold=0.20)
    else:
        aggressive_row = aggressive_candidates.sort_values(
            ["recall", "fn", "threshold"],
            ascending=[False, True, True],
        ).iloc[0].to_dict()

    return (
        ThresholdMode(name="conservative", threshold=float(conservative_row["threshold"]), metrics=conservative_row),
        ThresholdMode(name="balanced", threshold=float(balanced_row["threshold"]), metrics=balanced_row),
        ThresholdMode(name="aggressive", threshold=float(aggressive_row["threshold"]), metrics=aggressive_row),
    )


def plot_threshold_metric(metrics_table: pd.DataFrame, metric: str, filename: str, ylabel: str) -> str:
    """Plot a metric as a function of threshold."""
    plt.figure(figsize=(7, 4))
    plt.plot(metrics_table["threshold"], metrics_table[metric], marker="o")
    plt.xlabel("Threshold")
    plt.ylabel(ylabel)
    plt.title(f"Threshold vs {ylabel}")
    return save_current_figure(filename)
