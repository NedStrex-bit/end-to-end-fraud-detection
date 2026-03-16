"""Run threshold tuning and cost-sensitive analysis for the main model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.data.load_data import configure_logging
from src.models.thresholding import (
    ThresholdMode,
    analyze_thresholds,
    plot_threshold_metric,
    select_threshold_modes,
)
from src.models.train_baseline import load_processed_splits, split_xy


REPORTS_DIR = Path("artifacts/reports")
MODELS_DIR = Path("artifacts/models")
MODEL_PATH = MODELS_DIR / "main_hist_gradient_boosting.joblib"
THRESHOLD_METRICS_PATH = REPORTS_DIR / "threshold_metrics.csv"
THRESHOLD_REPORT_PATH = REPORTS_DIR / "threshold_report.md"


def load_main_model() -> dict[str, Any]:
    """Load the trained main model artifact."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Main model artifact was not found at '{MODEL_PATH}'. Run main model training first.")
    return joblib.load(MODEL_PATH)


def score_split(model_bundle: dict[str, Any], split_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Score a processed split using the saved model."""
    feature_names = model_bundle["feature_names"]
    target_column = model_bundle["target_column"]
    model = model_bundle["model"]

    features, target = split_xy(split_frame, target_column)
    features = features.reindex(columns=feature_names, fill_value=0.0)
    score = model.predict_proba(features)[:, 1]
    return target.to_numpy(), score


def mode_summary_line(mode: ThresholdMode) -> str:
    """Render one summary line for a threshold mode."""
    metrics = mode.metrics
    return (
        f"- `{mode.name}`: threshold=`{mode.threshold:.2f}`, precision=`{metrics['precision']:.4f}`, "
        f"recall=`{metrics['recall']:.4f}`, f1=`{metrics['f1']:.4f}`, "
        f"FP=`{metrics['fp']}`, FN=`{metrics['fn']}`, business_cost=`{metrics['business_cost']:.2f}`"
    )


def build_threshold_report(
    split_name: str,
    recommended_mode: ThresholdMode,
    conservative_mode: ThresholdMode,
    balanced_mode: ThresholdMode,
    aggressive_mode: ThresholdMode,
    cost_false_negative: float,
    cost_false_positive: float,
    metrics_table: pd.DataFrame,
) -> str:
    """Render the threshold tuning report."""
    data_limitations: list[str] = []
    if len(np.unique(metrics_table["fn"])) == 1 and metrics_table["fn"].iloc[0] == 0:
        data_limitations.append(
            "На выбранном split нет положительного fraud-класса, поэтому recall и FN здесь не отражают реальную способность ловить мошенничество."
        )

    lines = [
        "# Threshold Analysis Report",
        "",
        f"- Analysis split: `{split_name}`",
        f"- Cost false negative: `{cost_false_negative}`",
        f"- Cost false positive: `{cost_false_positive}`",
        f"- Recommended threshold: `{recommended_mode.threshold:.2f}`",
        "",
        "## Why Threshold Matters",
        "",
        "- Для fraud detection threshold является частью продукта: он определяет, какие транзакции уйдут в ручную проверку, будут отклонены или пропущены без действий.",
        "- False Negative означает пропущенный fraud и прямой денежный риск.",
        "- False Positive означает ложную тревогу, ухудшение клиентского опыта и нагрузку на антифрод-операции.",
        "- Поэтому один и тот же скоринг-моделью probability output нужно переводить в решение через business-aware threshold, а не фиксировать 0.5 по умолчанию.",
        "",
        "## Threshold Behavior",
        "",
        "- При росте threshold обычно растёт precision и падает recall.",
        "- При снижении threshold модель становится агрессивнее: растёт recall, но увеличиваются false positives.",
        "- Balanced режим выбирается по минимальной бизнес-стоимости ошибок, а не по accuracy.",
        "",
        "## Recommended Modes",
        "",
        mode_summary_line(conservative_mode),
        mode_summary_line(balanced_mode),
        mode_summary_line(aggressive_mode),
        "",
        "## Recommended Production Threshold",
        "",
        f"- Рекомендуемый рабочий threshold: `{recommended_mode.threshold:.2f}`",
        "- Этот threshold выбран как balanced-режим с минимальной стоимостью ошибок при заданных `cost_false_negative` и `cost_false_positive`.",
        "- Если стоимость пропуска fraud выше, threshold стоит сдвигать в aggressive сторону; если ложные алерты слишком дороги, ближе к conservative.",
        "",
        "## Data Limitations",
        "",
    ]

    if data_limitations:
        lines.extend(f"- {note}" for note in data_limitations)
    else:
        lines.append("- Analysis split contains both classes, so threshold trade-offs are directly observable.")

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Threshold table: `{THRESHOLD_METRICS_PATH}`",
            "- Plot: `artifacts/plots/threshold_vs_precision.png`",
            "- Plot: `artifacts/plots/threshold_vs_recall.png`",
            "- Plot: `artifacts/plots/threshold_vs_f1.png`",
            "- Plot: `artifacts/plots/threshold_vs_business_cost.png`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run threshold analysis for the main fraud model.")
    parser.add_argument("--split", type=str, default="valid", choices=["valid", "test"], help="Split to analyze.")
    parser.add_argument("--cost-fn", type=float, default=100.0, help="Business cost of a false negative.")
    parser.add_argument("--cost-fp", type=float, default=5.0, help="Business cost of a false positive.")
    parser.add_argument("--min-threshold", type=float, default=0.05, help="Minimum threshold to evaluate.")
    parser.add_argument("--max-threshold", type=float, default=0.95, help="Maximum threshold to evaluate.")
    parser.add_argument("--num-thresholds", type=int, default=19, help="Number of thresholds in the grid.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    configure_logging()
    args = parse_args()

    model_bundle = load_main_model()
    splits = load_processed_splits()
    split_frame = splits[args.split]
    y_true, y_score = score_split(model_bundle, split_frame)

    thresholds = np.linspace(args.min_threshold, args.max_threshold, args.num_thresholds)
    metrics_table = analyze_thresholds(
        y_true=y_true,
        y_score=y_score,
        thresholds=thresholds,
        cost_false_negative=args.cost_fn,
        cost_false_positive=args.cost_fp,
    )

    conservative_mode, balanced_mode, aggressive_mode = select_threshold_modes(metrics_table)
    recommended_mode = balanced_mode

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_table.to_csv(THRESHOLD_METRICS_PATH, index=False)

    plot_threshold_metric(metrics_table, "precision", "threshold_vs_precision.png", "Precision")
    plot_threshold_metric(metrics_table, "recall", "threshold_vs_recall.png", "Recall")
    plot_threshold_metric(metrics_table, "f1", "threshold_vs_f1.png", "F1")
    plot_threshold_metric(metrics_table, "business_cost", "threshold_vs_business_cost.png", "Business Cost")

    THRESHOLD_REPORT_PATH.write_text(
        build_threshold_report(
            split_name=args.split,
            recommended_mode=recommended_mode,
            conservative_mode=conservative_mode,
            balanced_mode=balanced_mode,
            aggressive_mode=aggressive_mode,
            cost_false_negative=args.cost_fn,
            cost_false_positive=args.cost_fp,
            metrics_table=metrics_table,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
