"""EDA helpers and plot generation utilities."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib")))

import matplotlib
import numpy as np
import pandas as pd

from src.data.preprocess import (
    count_duplicate_rows,
    count_missing_values,
    detect_transaction_id_candidates,
    detect_user_id_candidates,
    get_categorical_columns,
    get_numeric_columns,
    normalize_column_name,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.getLogger("matplotlib").setLevel(logging.WARNING)


PLOTS_DIR = Path("artifacts/plots")
AMOUNT_PATTERNS = ("amount", "transaction_amount", "value")


def ensure_plots_dir() -> None:
    """Create plots directory if needed."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def save_current_figure(filename: str) -> str:
    """Save current matplotlib figure and close it."""
    ensure_plots_dir()
    path = PLOTS_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(path)


def detect_amount_column(dataframe: pd.DataFrame) -> str | None:
    """Detect a likely transaction amount column."""
    numeric_columns = get_numeric_columns(dataframe)
    for column in numeric_columns:
        normalized = normalize_column_name(column)
        if any(pattern in normalized for pattern in AMOUNT_PATTERNS):
            return column
    return None


def prepare_target_series(dataframe: pd.DataFrame, target_column: str | None) -> pd.Series | None:
    """Convert target column to a numeric binary-like series when possible."""
    if target_column is None or target_column not in dataframe.columns:
        return None

    raw_series = dataframe[target_column]
    if pd.api.types.is_bool_dtype(raw_series):
        return raw_series.astype(int)

    numeric_series = pd.to_numeric(raw_series, errors="coerce")
    non_null_values = numeric_series.dropna().unique()
    if len(non_null_values) == 0:
        return None
    if set(np.unique(non_null_values)).issubset({0, 1}):
        return numeric_series
    return None


def prepare_timestamp_series(dataframe: pd.DataFrame, timestamp_column: str | None) -> pd.Series | None:
    """Parse timestamp column when possible."""
    if timestamp_column is None or timestamp_column not in dataframe.columns:
        return None
    parsed = pd.to_datetime(dataframe[timestamp_column], errors="coerce")
    if parsed.notna().sum() == 0:
        return None
    return parsed


def analyze_target_distribution(
    dataframe: pd.DataFrame,
    target_column: str | None,
) -> tuple[dict[str, Any], list[str]]:
    """Analyze target class balance and save a bar chart."""
    notes: list[str] = []
    result: dict[str, Any] = {
        "target_column": target_column,
        "available": False,
    }
    target_series = prepare_target_series(dataframe, target_column)
    if target_series is None:
        notes.append("Target column was not detected or is not binary-like; target distribution was skipped.")
        return result, notes

    counts = target_series.value_counts(dropna=False).sort_index()
    rates = (counts / len(target_series)).round(4)

    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar")
    plt.title("Target Distribution")
    plt.xlabel("Target value")
    plt.ylabel("Count")
    plot_path = save_current_figure("target_distribution.png")

    result.update(
        {
            "available": True,
            "counts": {str(key): int(value) for key, value in counts.items()},
            "rates": {str(key): float(value) for key, value in rates.items()},
            "plot_path": plot_path,
            "fraud_rate": float(target_series.mean()),
        }
    )
    return result, notes


def analyze_numeric_features(dataframe: pd.DataFrame) -> tuple[dict[str, Any], list[str]]:
    """Summarize numeric features and save a correlation heatmap."""
    notes: list[str] = []
    numeric_columns = get_numeric_columns(dataframe)
    numeric_summary: dict[str, Any] = {
        "numeric_columns": numeric_columns,
        "descriptive_stats": {},
        "correlation_plot_path": None,
    }

    for column in numeric_columns:
        stats = dataframe[column].describe(percentiles=[0.25, 0.5, 0.75, 0.95]).to_dict()
        numeric_summary["descriptive_stats"][column] = {
            key: float(value) if pd.notna(value) else None for key, value in stats.items()
        }

    if len(numeric_columns) >= 2:
        correlation = dataframe[numeric_columns].corr(numeric_only=True)
        plt.figure(figsize=(max(6, len(numeric_columns) * 1.2), max(5, len(numeric_columns) * 1.0)))
        plt.imshow(correlation, aspect="auto")
        plt.colorbar(label="Correlation")
        plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=45, ha="right")
        plt.yticks(range(len(correlation.index)), correlation.index)
        plt.title("Numeric Feature Correlation Heatmap")
        numeric_summary["correlation_plot_path"] = save_current_figure("numeric_correlation_heatmap.png")
    else:
        notes.append("Correlation heatmap was skipped because fewer than two numeric columns were available.")

    return numeric_summary, notes


def analyze_amount_distribution(dataframe: pd.DataFrame) -> tuple[dict[str, Any], list[str]]:
    """Analyze a likely amount-like numeric column."""
    notes: list[str] = []
    amount_column = detect_amount_column(dataframe)
    result: dict[str, Any] = {
        "amount_column": amount_column,
        "available": False,
    }

    if amount_column is None:
        notes.append("No amount-like numeric column was detected; amount distribution plot was skipped.")
        return result, notes

    series = pd.to_numeric(dataframe[amount_column], errors="coerce").dropna()
    if series.empty:
        notes.append(f"Detected amount column '{amount_column}' could not be parsed to numeric values.")
        return result, notes

    plt.figure(figsize=(7, 4))
    plt.hist(series, bins=min(30, max(5, series.nunique())), edgecolor="black")
    plt.title(f"Distribution of {amount_column}")
    plt.xlabel(amount_column)
    plt.ylabel("Count")
    plot_path = save_current_figure("amount_distribution.png")

    result.update(
        {
            "available": True,
            "plot_path": plot_path,
            "summary": {
                "min": float(series.min()),
                "median": float(series.median()),
                "mean": float(series.mean()),
                "p95": float(series.quantile(0.95)),
                "max": float(series.max()),
            },
        }
    )
    return result, notes


def analyze_categorical_features(dataframe: pd.DataFrame, top_n: int = 10) -> tuple[dict[str, Any], list[str]]:
    """Summarize categorical columns."""
    notes: list[str] = []
    categorical_columns = get_categorical_columns(dataframe)
    result: dict[str, Any] = {
        "categorical_columns": categorical_columns,
        "top_values": {},
    }

    if not categorical_columns:
        notes.append("No categorical columns were detected.")
        return result, notes

    for column in categorical_columns:
        counts = dataframe[column].astype("string").fillna("<missing>").value_counts().head(top_n)
        result["top_values"][column] = {str(key): int(value) for key, value in counts.items()}

    return result, notes


def analyze_fraud_rate_by_groups(
    dataframe: pd.DataFrame,
    target_column: str | None,
    top_n: int = 10,
) -> tuple[dict[str, Any], list[str]]:
    """Analyze fraud rate across categorical groups."""
    notes: list[str] = []
    result: dict[str, Any] = {
        "available": False,
        "group_summaries": {},
        "plot_path": None,
    }
    target_series = prepare_target_series(dataframe, target_column)
    if target_series is None:
        notes.append("Fraud-rate-by-group analysis was skipped because a binary target column was not available.")
        return result, notes

    excluded_columns = set(detect_user_id_candidates(dataframe)) | set(detect_transaction_id_candidates(dataframe))
    candidate_columns = [
        column
        for column in get_categorical_columns(dataframe)
        if column != target_column
        and column not in excluded_columns
        and dataframe[column].nunique(dropna=True) <= 20
        and dataframe[column].nunique(dropna=True) / max(len(dataframe), 1) <= 0.5
    ]
    if not candidate_columns:
        notes.append("No low-cardinality non-ID categorical columns were available for fraud rate by group analysis.")
        return result, notes

    best_column = None
    best_summary = None
    best_gap = -1.0

    for column in candidate_columns:
        grouped = (
            pd.DataFrame({column: dataframe[column].astype("string").fillna("<missing>"), target_column: target_series})
            .groupby(column, dropna=False)[target_column]
            .agg(["mean", "count"])
            .sort_values(["mean", "count"], ascending=[False, False])
            .head(top_n)
        )
        fraud_rates = grouped["mean"]
        gap = float(fraud_rates.max() - fraud_rates.min()) if not fraud_rates.empty else 0.0
        result["group_summaries"][column] = {
            str(index): {
                "fraud_rate": float(row["mean"]),
                "count": int(row["count"]),
            }
            for index, row in grouped.iterrows()
        }
        if gap > best_gap:
            best_gap = gap
            best_column = column
            best_summary = grouped

    if best_column is None or best_summary is None:
        notes.append("No suitable categorical grouping was found for fraud rate analysis.")
        return result, notes

    plt.figure(figsize=(8, 4))
    plt.bar(best_summary.index.astype(str), best_summary["mean"])
    plt.title(f"Fraud Rate by {best_column}")
    plt.xlabel(best_column)
    plt.ylabel("Fraud rate")
    plt.xticks(rotation=45, ha="right")
    plot_path = save_current_figure("fraud_rate_by_group.png")

    result.update(
        {
            "available": True,
            "best_group_column": best_column,
            "plot_path": plot_path,
        }
    )
    return result, notes


def analyze_target_vs_amount(
    dataframe: pd.DataFrame,
    target_column: str | None,
) -> tuple[dict[str, Any], list[str]]:
    """Compare amount-like feature across fraud and non-fraud classes."""
    notes: list[str] = []
    result: dict[str, Any] = {
        "available": False,
        "comparison_type": "percentile_comparison",
        "plot_path": None,
    }
    target_series = prepare_target_series(dataframe, target_column)
    amount_column = detect_amount_column(dataframe)

    if target_series is None:
        notes.append("Fraud vs non-fraud numeric comparison was skipped because target is unavailable.")
        return result, notes
    if amount_column is None:
        notes.append("Fraud vs non-fraud numeric comparison was skipped because no amount-like column was found.")
        return result, notes

    numeric_series = pd.to_numeric(dataframe[amount_column], errors="coerce")
    comparison_frame = pd.DataFrame({"target": target_series, "amount": numeric_series}).dropna()
    if comparison_frame.empty or comparison_frame["target"].nunique() < 2:
        notes.append("Fraud vs non-fraud numeric comparison was skipped because both target classes were not available.")
        return result, notes

    percentiles = [0.25, 0.5, 0.75, 0.95]
    percentile_frame = comparison_frame.groupby("target")["amount"].quantile(percentiles).unstack()

    plt.figure(figsize=(8, 4))
    for target_value in percentile_frame.index:
        plt.plot(percentile_frame.columns, percentile_frame.loc[target_value], marker="o", label=f"target={int(target_value)}")
    plt.title(f"{amount_column} Percentiles by Target")
    plt.xlabel("Percentile")
    plt.ylabel(amount_column)
    plt.legend()
    plot_path = save_current_figure("amount_percentiles_by_target.png")

    result.update(
        {
            "available": True,
            "amount_column": amount_column,
            "plot_path": plot_path,
            "percentiles": {
                str(int(index)): {str(col): float(value) for col, value in row.items()}
                for index, row in percentile_frame.iterrows()
            },
        }
    )
    return result, notes


def analyze_temporal_patterns(
    dataframe: pd.DataFrame,
    timestamp_column: str | None,
    target_column: str | None,
) -> tuple[dict[str, Any], list[str]]:
    """Analyze fraud rate by hour and day of week if timestamp exists."""
    notes: list[str] = []
    result: dict[str, Any] = {
        "available": False,
        "timestamp_column": timestamp_column,
        "hour_plot_path": None,
        "day_of_week_plot_path": None,
    }
    timestamp_series = prepare_timestamp_series(dataframe, timestamp_column)
    target_series = prepare_target_series(dataframe, target_column)

    if timestamp_series is None:
        notes.append("Timestamp column was not detected or could not be parsed; temporal analysis was skipped.")
        return result, notes
    if target_series is None:
        notes.append("Temporal fraud-rate analysis was skipped because target is unavailable.")
        return result, notes

    temporal_frame = pd.DataFrame({"timestamp": timestamp_series, "target": target_series}).dropna()
    if temporal_frame.empty:
        notes.append("Temporal analysis was skipped because no valid timestamp values remained after parsing.")
        return result, notes

    temporal_frame["hour"] = temporal_frame["timestamp"].dt.hour
    temporal_frame["day_of_week"] = temporal_frame["timestamp"].dt.day_name()

    hour_rate = temporal_frame.groupby("hour")["target"].mean().reindex(range(24), fill_value=np.nan)
    plt.figure(figsize=(8, 4))
    plt.plot(hour_rate.index, hour_rate.values, marker="o")
    plt.title("Fraud Rate by Hour")
    plt.xlabel("Hour")
    plt.ylabel("Fraud rate")
    hour_plot_path = save_current_figure("fraud_rate_by_hour.png")

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_rate = temporal_frame.groupby("day_of_week")["target"].mean().reindex(day_order)
    plt.figure(figsize=(8, 4))
    plt.bar(day_rate.index.astype(str), day_rate.values)
    plt.title("Fraud Rate by Day of Week")
    plt.xlabel("Day of week")
    plt.ylabel("Fraud rate")
    plt.xticks(rotation=45, ha="right")
    day_plot_path = save_current_figure("fraud_rate_by_day_of_week.png")

    result.update(
        {
            "available": True,
            "hour_plot_path": hour_plot_path,
            "day_of_week_plot_path": day_plot_path,
            "hour_rates": {str(index): None if pd.isna(value) else float(value) for index, value in hour_rate.items()},
            "day_of_week_rates": {str(index): None if pd.isna(value) else float(value) for index, value in day_rate.items()},
        }
    )
    return result, notes


def run_full_eda(
    dataframe: pd.DataFrame,
    target_column: str | None,
    timestamp_column: str | None,
    user_id_column: str | None,
    transaction_id_column: str | None,
) -> dict[str, Any]:
    """Run all EDA blocks and return a structured summary."""
    dataset_summary = {
        "row_count": int(len(dataframe)),
        "column_count": int(dataframe.shape[1]),
        "missing_values": count_missing_values(dataframe),
        "duplicate_rows": count_duplicate_rows(dataframe),
        "target_column": target_column,
        "timestamp_column": timestamp_column,
        "user_id_column": user_id_column,
        "transaction_id_column": transaction_id_column,
    }

    target_distribution, target_notes = analyze_target_distribution(dataframe, target_column)
    numeric_analysis, numeric_notes = analyze_numeric_features(dataframe)
    amount_analysis, amount_notes = analyze_amount_distribution(dataframe)
    categorical_analysis, categorical_notes = analyze_categorical_features(dataframe)
    fraud_group_analysis, fraud_group_notes = analyze_fraud_rate_by_groups(dataframe, target_column)
    target_amount_analysis, target_amount_notes = analyze_target_vs_amount(dataframe, target_column)
    temporal_analysis, temporal_notes = analyze_temporal_patterns(dataframe, timestamp_column, target_column)

    return {
        "dataset_summary": dataset_summary,
        "target_distribution": target_distribution,
        "numeric_analysis": numeric_analysis,
        "amount_analysis": amount_analysis,
        "categorical_analysis": categorical_analysis,
        "fraud_group_analysis": fraud_group_analysis,
        "target_amount_analysis": target_amount_analysis,
        "temporal_analysis": temporal_analysis,
        "notes": (
            target_notes
            + numeric_notes
            + amount_notes
            + categorical_notes
            + fraud_group_notes
            + target_amount_notes
            + temporal_notes
        ),
    }
