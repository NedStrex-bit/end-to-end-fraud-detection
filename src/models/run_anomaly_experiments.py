"""Run optional anomaly detection experiments and compare them to the supervised model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.data.load_data import configure_logging
from src.models.anomaly_detection import (
    anomaly_score,
    combine_scores,
    comparison_rows_from_metrics,
    evaluate_score_model,
    fit_anomaly_model,
    plot_score_distribution,
    threshold_from_train_scores,
)
from src.models.train_baseline import load_processed_splits, split_xy


MODELS_DIR = Path("artifacts/models")
REPORTS_DIR = Path("artifacts/reports")
MAIN_MODEL_PATH = MODELS_DIR / "main_hist_gradient_boosting.joblib"
MAIN_METRICS_PATH = REPORTS_DIR / "main_model_metrics.json"
ANOMALY_REPORT_PATH = REPORTS_DIR / "anomaly_report.md"
ANOMALY_METRICS_PATH = REPORTS_DIR / "anomaly_metrics.csv"


def load_main_model_bundle() -> dict[str, Any]:
    """Load final supervised model artifact."""
    if not MAIN_MODEL_PATH.exists():
        raise FileNotFoundError(f"Main model artifact not found at '{MAIN_MODEL_PATH}'. Run main model training first.")
    return joblib.load(MAIN_MODEL_PATH)


def score_supervised(model_bundle: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    """Score processed frame with the saved supervised model."""
    target_column = model_bundle["target_column"]
    feature_names = model_bundle["feature_names"]
    features, _ = split_xy(frame, target_column)
    features = features.reindex(columns=feature_names, fill_value=0.0)
    return model_bundle["model"].predict_proba(features)[:, 1]


def build_anomaly_report(
    fit_notes: list[str],
    holdout_metrics: dict[str, Any],
    full_metrics: dict[str, Any],
    supervised_full_metrics: dict[str, Any],
    ensemble_full_metrics: dict[str, Any] | None,
    anomaly_threshold: float,
    plots: dict[str, str],
) -> str:
    """Render markdown report for anomaly experiments."""
    qualitative_notes = [
        "Unsupervised anomaly detection может быть полезен как дополнительный сигнал для новых или редких fraud-паттернов, когда размеченных примеров мало.",
        "Supervised модель обычно сильнее там, где есть достаточная разметка и повторяющиеся fraud-шаблоны, потому что она напрямую оптимизируется под target.",
        "Isolation Forest чаще ловит статистические выбросы, но не обязательно хорошо распознаёт 'мягкий' fraud, похожий на нормальное пользовательское поведение.",
    ]
    limitations = [
        "Anomaly detection здесь рассматривается как дополнительный эксперимент, а не замена supervised fraud-модели.",
        "Combined evaluation по train+valid+test полезен как qualitative sanity check, но финальные продуктовые выводы всё равно стоит делать по отдельному holdout и business threshold.",
    ]

    lines = [
        "# Anomaly Detection Report",
        "",
        "- Experiment: `IsolationForest`",
        f"- Anomaly threshold from train score quantile: `{anomaly_threshold:.4f}`",
        "",
        "## Training Setup",
        "",
    ]
    lines.extend(f"- {note}" for note in fit_notes)

    lines.extend(
        [
            "",
            "## Holdout Comparison",
            "",
            f"- Isolation Forest ROC-AUC: `{holdout_metrics['roc_auc']}`",
            f"- Isolation Forest PR-AUC: `{holdout_metrics['pr_auc']}`",
            f"- Isolation Forest Precision: `{holdout_metrics['precision']:.4f}`",
            f"- Isolation Forest Recall: `{holdout_metrics['recall']:.4f}`",
            f"- Isolation Forest F1: `{holdout_metrics['f1']:.4f}`",
            "",
            "## Combined Evaluation (Illustrative)",
            "",
            f"- Supervised ROC-AUC: `{supervised_full_metrics['roc_auc']}`",
            f"- Isolation Forest ROC-AUC: `{full_metrics['roc_auc']}`",
            f"- Supervised PR-AUC: `{supervised_full_metrics['pr_auc']}`",
            f"- Isolation Forest PR-AUC: `{full_metrics['pr_auc']}`",
            f"- Supervised F1: `{supervised_full_metrics['f1']:.4f}`",
            f"- Isolation Forest F1: `{full_metrics['f1']:.4f}`",
        ]
    )

    if ensemble_full_metrics is not None:
        lines.extend(
            [
                "",
                "## Simple Ensemble",
                "",
                f"- Ensemble ROC-AUC: `{ensemble_full_metrics['roc_auc']}`",
                f"- Ensemble PR-AUC: `{ensemble_full_metrics['pr_auc']}`",
                f"- Ensemble F1: `{ensemble_full_metrics['f1']:.4f}`",
                "- Ensemble score uses `0.7 * supervised_score + 0.3 * anomaly_score` only as a lightweight experiment.",
            ]
        )

    lines.extend(
        [
            "",
            "## Qualitative Conclusion",
            "",
        ]
    )
    lines.extend(f"- {note}" for note in qualitative_notes)
    lines.extend(
        [
            "",
            "## Practical Recommendation",
            "",
            "- Оставлять anomaly-модуль стоит как optional experiment или дополнительный monitoring/risk signal.",
            "- В качестве основного production scorer supervised model остаётся приоритетом.",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {note}" for note in limitations)
    lines.extend(
        [
            "",
            "## Plots",
            "",
            f"- Anomaly score distribution: `{plots['anomaly_distribution']}`",
            f"- Supervised score distribution: `{plots['supervised_distribution']}`",
        ]
    )
    if "ensemble_distribution" in plots:
        lines.append(f"- Ensemble score distribution: `{plots['ensemble_distribution']}`")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Run anomaly detection experiments.")
    parser.add_argument("--ensemble-alpha", type=float, default=0.7, help="Weight of supervised score in simple ensemble.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    configure_logging()
    args = parse_args()

    model_bundle = load_main_model_bundle()
    splits = load_processed_splits()
    target_column = model_bundle["target_column"]
    feature_names = model_bundle["feature_names"]

    train_x, train_y = split_xy(splits["train"], target_column)
    train_x = train_x.reindex(columns=feature_names, fill_value=0.0)

    anomaly_model, fit_notes = fit_anomaly_model(train_x=train_x, train_y=train_y)
    train_anomaly_score = anomaly_score(anomaly_model, train_x)
    anomaly_threshold = threshold_from_train_scores(train_anomaly_score, quantile=0.95)

    holdout_frame = pd.concat([splits["valid"], splits["test"]], axis=0, ignore_index=True)
    full_frame = pd.concat([splits["train"], splits["valid"], splits["test"]], axis=0, ignore_index=True)

    holdout_x, holdout_y = split_xy(holdout_frame, target_column)
    holdout_x = holdout_x.reindex(columns=feature_names, fill_value=0.0)
    holdout_anomaly_score = anomaly_score(anomaly_model, holdout_x)
    holdout_metrics = evaluate_score_model(holdout_y.to_numpy(), holdout_anomaly_score, anomaly_threshold)

    full_x, full_y = split_xy(full_frame, target_column)
    full_x = full_x.reindex(columns=feature_names, fill_value=0.0)
    full_anomaly_score = anomaly_score(anomaly_model, full_x)
    full_metrics = evaluate_score_model(full_y.to_numpy(), full_anomaly_score, anomaly_threshold)

    supervised_full_score = score_supervised(model_bundle, full_frame)
    supervised_full_metrics = evaluate_score_model(full_y.to_numpy(), supervised_full_score, threshold=0.5)

    ensemble_score = combine_scores(supervised_full_score, full_anomaly_score, alpha=args.ensemble_alpha)
    ensemble_threshold = float(np.quantile(ensemble_score, 0.75))
    ensemble_full_metrics = evaluate_score_model(full_y.to_numpy(), ensemble_score, threshold=ensemble_threshold)

    score_frame = pd.DataFrame(
        {
            "target": full_y.to_numpy(),
            "anomaly_score": full_anomaly_score,
            "supervised_score": supervised_full_score,
            "ensemble_score": ensemble_score,
        }
    )
    plots = {
        "anomaly_distribution": plot_score_distribution(
            score_frame,
            score_column="anomaly_score",
            filename="anomaly_score_distribution.png",
            title="Isolation Forest Score Distribution",
        ),
        "supervised_distribution": plot_score_distribution(
            score_frame,
            score_column="supervised_score",
            filename="supervised_score_distribution.png",
            title="Supervised Score Distribution",
        ),
        "ensemble_distribution": plot_score_distribution(
            score_frame,
            score_column="ensemble_score",
            filename="ensemble_score_distribution.png",
            title="Ensemble Score Distribution",
        ),
    }

    metrics_table = comparison_rows_from_metrics(
        {
            "supervised_full": supervised_full_metrics,
            "isolation_forest_holdout": holdout_metrics,
            "isolation_forest_full": full_metrics,
            "ensemble_full": ensemble_full_metrics,
        }
    )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_table.to_csv(ANOMALY_METRICS_PATH, index=False)
    ANOMALY_REPORT_PATH.write_text(
        build_anomaly_report(
            fit_notes=fit_notes,
            holdout_metrics=holdout_metrics,
            full_metrics=full_metrics,
            supervised_full_metrics=supervised_full_metrics,
            ensemble_full_metrics=ensemble_full_metrics,
            anomaly_threshold=anomaly_threshold,
            plots=plots,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
