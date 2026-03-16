"""Explainability and error analysis helpers."""

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
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, log_loss


logging.getLogger("matplotlib").setLevel(logging.WARNING)

PLOTS_DIR = Path("artifacts/plots")
REPORTS_DIR = Path("artifacts/reports")


def save_current_figure(filename: str) -> str:
    """Save the active matplotlib figure."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(path)


def compute_model_feature_importance(
    model: Any,
    feature_frame: pd.DataFrame,
    target: pd.Series,
    random_state: int = 42,
) -> tuple[pd.DataFrame, list[str]]:
    """Compute feature importance with the simplest robust method available."""
    notes: list[str] = []
    importance_frame: pd.DataFrame

    if hasattr(model, "feature_importances_"):
        importance_frame = pd.DataFrame(
            {
                "feature": feature_frame.columns,
                "importance": model.feature_importances_,
                "importance_type": "native_importance",
            }
        ).sort_values("importance", ascending=False)
        notes.append("Использована встроенная feature importance модели.")
        return importance_frame.reset_index(drop=True), notes

    if target.nunique(dropna=True) > 1:
        scoring: str | Any = "f1"
    else:
        def scoring(estimator: Any, x: pd.DataFrame, y: pd.Series) -> float:
            probabilities = estimator.predict_proba(x)
            return -float(log_loss(y, probabilities, labels=[0, 1]))

    importance = permutation_importance(
        model,
        feature_frame,
        target,
        n_repeats=10,
        random_state=random_state,
        scoring=scoring,
    )
    importance_frame = pd.DataFrame(
        {
            "feature": feature_frame.columns,
            "importance": importance.importances_mean,
            "importance_std": importance.importances_std,
            "importance_type": "permutation_importance",
        }
    ).sort_values("importance", ascending=False)
    scoring_name = scoring if isinstance(scoring, str) else "custom_neg_log_loss"
    notes.append(f"SHAP недоступен; использован fallback на permutation importance со scoring=`{scoring_name}`.")
    return importance_frame.reset_index(drop=True), notes


def plot_feature_importance(importance_frame: pd.DataFrame, top_n: int = 10) -> str:
    """Plot the top-N feature importances."""
    top = importance_frame.head(top_n).iloc[::-1]
    plt.figure(figsize=(8, max(4, top_n * 0.4)))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top Feature Importances")
    return save_current_figure("feature_importance_top10.png")


def plot_feature_distribution_by_error_type(
    analysis_frame: pd.DataFrame,
    feature: str,
    filename: str,
) -> str:
    """Plot feature distributions by prediction outcome group."""
    plt.figure(figsize=(8, 4))
    groups = analysis_frame.groupby("error_type")
    labels: list[str] = []
    values: list[np.ndarray] = []

    for label, group in groups:
        if feature in group.columns:
            numeric_values = pd.to_numeric(group[feature], errors="coerce").dropna().to_numpy()
            if len(numeric_values) > 0:
                labels.append(str(label))
                values.append(numeric_values)

    if values:
        plt.boxplot(values, tick_labels=labels)
        plt.ylabel(feature)
        plt.title(f"{feature} by Error Type")
    else:
        plt.text(0.5, 0.5, "No numeric values available", ha="center", va="center")
        plt.title(f"{feature} by Error Type")
    return save_current_figure(filename)


def build_prediction_frame(
    feature_frame: pd.DataFrame,
    y_true: pd.Series,
    y_score: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """Create a row-level frame with predictions and error types."""
    frame = feature_frame.copy().reset_index(drop=True)
    frame["y_true"] = y_true.reset_index(drop=True).astype(int)
    frame["score"] = y_score
    frame["y_pred"] = (y_score >= threshold).astype(int)

    conditions = [
        (frame["y_true"] == 1) & (frame["y_pred"] == 1),
        (frame["y_true"] == 1) & (frame["y_pred"] == 0),
        (frame["y_true"] == 0) & (frame["y_pred"] == 1),
        (frame["y_true"] == 0) & (frame["y_pred"] == 0),
    ]
    choices = ["TP", "FN", "FP", "TN"]
    frame["error_type"] = np.select(conditions, choices, default="UNKNOWN")
    return frame


def summarize_error_groups(
    prediction_frame: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Build a compact group summary for TP/FP/FN/TN."""
    rows: list[dict[str, Any]] = []
    numeric_features = [column for column in feature_columns if pd.api.types.is_numeric_dtype(prediction_frame[column])]

    for error_type, group in prediction_frame.groupby("error_type", sort=False):
        row: dict[str, Any] = {
            "error_type": error_type,
            "count": int(len(group)),
            "mean_score": float(group["score"].mean()) if len(group) else None,
        }
        for feature in numeric_features[:5]:
            row[f"{feature}_mean"] = float(group[feature].mean()) if len(group) else None
        rows.append(row)
    return pd.DataFrame(rows)


def export_error_tables(prediction_frame: pd.DataFrame) -> dict[str, str]:
    """Save FP/FN/TP/TN subsets for inspection."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for error_type in ["FP", "FN", "TP", "TN"]:
        path = REPORTS_DIR / f"{error_type.lower()}_cases.csv"
        prediction_frame[prediction_frame["error_type"] == error_type].to_csv(path, index=False)
        paths[error_type] = str(path)
    summary_path = REPORTS_DIR / "error_group_summary.csv"
    summarize_error_groups(
        prediction_frame=prediction_frame,
        feature_columns=[column for column in prediction_frame.columns if column not in {"y_true", "y_pred", "score", "error_type"}],
    ).to_csv(summary_path, index=False)
    paths["summary"] = str(summary_path)
    return paths


def basic_error_findings(prediction_frame: pd.DataFrame, top_features: list[str]) -> list[str]:
    """Generate short textual findings about current errors."""
    findings: list[str] = []
    counts = prediction_frame["error_type"].value_counts().to_dict()
    findings.append(f"Распределение исходов: {counts}.")

    fp_frame = prediction_frame[prediction_frame["error_type"] == "FP"]
    fn_frame = prediction_frame[prediction_frame["error_type"] == "FN"]

    if len(fp_frame) > 0:
        findings.append(f"False positives присутствуют ({len(fp_frame)} наблюдений); модель переоценивает fraud-риск для части benign-транзакций.")
    else:
        findings.append("False positives не обнаружены на анализируемом split.")

    if len(fn_frame) > 0:
        findings.append(f"False negatives присутствуют ({len(fn_frame)} наблюдений); модель пропускает часть fraud-кейсов.")
    else:
        findings.append("False negatives не обнаружены на анализируемом split.")

    if top_features:
        findings.append(f"Для прицельного разбора ошибок полезно отслеживать признаки: {top_features[:3]}.")
    return findings


def confusion_matrix_as_list(prediction_frame: pd.DataFrame) -> list[list[int]]:
    """Return a stable binary confusion matrix."""
    return confusion_matrix(prediction_frame["y_true"], prediction_frame["y_pred"], labels=[0, 1]).tolist()
