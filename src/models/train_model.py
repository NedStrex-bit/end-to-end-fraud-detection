"""Train, tune, and compare the main fraud detection model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

from src.data.load_data import configure_logging
from src.models.evaluate import evaluate_predictions, plot_pr_curve, plot_roc_curve
from src.models.train_baseline import load_feature_metadata, load_processed_splits, split_xy


REPORTS_DIR = Path("artifacts/reports")
MODELS_DIR = Path("artifacts/models")
MAIN_MODEL_PATH = MODELS_DIR / "main_hist_gradient_boosting.joblib"
MAIN_METRICS_PATH = REPORTS_DIR / "main_model_metrics.json"
COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
MODEL_REPORT_PATH = REPORTS_DIR / "model_report.md"
BASELINE_METRICS_PATH = REPORTS_DIR / "baseline_metrics.json"


def _json_default(value: Any) -> Any:
    """Convert numpy scalars to Python primitives."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def build_model(params: dict[str, Any]) -> HistGradientBoostingClassifier:
    """Create the main gradient boosting model."""
    return HistGradientBoostingClassifier(
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        max_iter=params["max_iter"],
        min_samples_leaf=params["min_samples_leaf"],
        l2_regularization=params["l2_regularization"],
        early_stopping=False,
        random_state=42,
    )


def evaluate_split(model: HistGradientBoostingClassifier, frame: pd.DataFrame, target_column: str, split_name: str) -> dict[str, Any]:
    """Evaluate a model on a split and save ranking curves when available."""
    features, target = split_xy(frame, target_column)
    score = model.predict_proba(features)[:, 1]
    prediction = (score >= 0.5).astype(int)
    metrics = evaluate_predictions(target.to_numpy(), prediction, score)
    metrics["roc_curve_plot"] = plot_roc_curve(target.to_numpy(), score, f"{split_name}_main_roc_curve.png")
    metrics["pr_curve_plot"] = plot_pr_curve(target.to_numpy(), score, f"{split_name}_main_pr_curve.png")
    return metrics


def load_baseline_metrics() -> dict[str, Any]:
    """Load previously saved baseline metrics if available."""
    if not BASELINE_METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Baseline metrics were not found at '{BASELINE_METRICS_PATH}'. Run baseline training first."
        )
    return json.loads(BASELINE_METRICS_PATH.read_text(encoding="utf-8"))


def choose_selection_metric(metrics: dict[str, Any]) -> tuple[str, float, str]:
    """Choose a robust validation metric for tuning."""
    if metrics["pr_auc"] is not None:
        return "pr_auc", float(metrics["pr_auc"]), "max"
    if metrics["roc_auc"] is not None:
        return "roc_auc", float(metrics["roc_auc"]), "max"
    return "log_loss", float(metrics["log_loss"]), "min"


def tune_model(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    target_column: str,
) -> tuple[HistGradientBoostingClassifier, dict[str, Any], list[dict[str, Any]]]:
    """Run a small reproducible hyperparameter search on the validation split."""
    train_x, train_y = split_xy(train_frame, target_column)
    sample_weight = compute_sample_weight(class_weight="balanced", y=train_y)

    grid = [
        {"learning_rate": 0.03, "max_depth": 3, "max_iter": 200, "min_samples_leaf": 20, "l2_regularization": 0.0},
        {"learning_rate": 0.05, "max_depth": 3, "max_iter": 300, "min_samples_leaf": 20, "l2_regularization": 0.0},
        {"learning_rate": 0.05, "max_depth": 4, "max_iter": 300, "min_samples_leaf": 10, "l2_regularization": 0.0},
        {"learning_rate": 0.08, "max_depth": 4, "max_iter": 200, "min_samples_leaf": 10, "l2_regularization": 0.1},
    ]

    best_model: HistGradientBoostingClassifier | None = None
    best_params: dict[str, Any] | None = None
    tuning_results: list[dict[str, Any]] = []
    best_score: float | None = None
    best_direction = "max"
    best_metric_name = "pr_auc"

    for params in grid:
        candidate = build_model(params)
        candidate.fit(train_x, train_y, sample_weight=sample_weight)
        valid_metrics = evaluate_split(candidate, valid_frame, target_column, "valid_candidate")
        metric_name, metric_value, direction = choose_selection_metric(valid_metrics)
        row = {
            "params": params,
            "selection_metric": metric_name,
            "selection_value": metric_value,
            "direction": direction,
            "valid_metrics": valid_metrics,
        }
        tuning_results.append(row)

        is_better = False
        if best_score is None:
            is_better = True
        elif direction == "max" and metric_value > best_score:
            is_better = True
        elif direction == "min" and metric_value < best_score:
            is_better = True

        if is_better:
            best_model = candidate
            best_params = params
            best_score = metric_value
            best_direction = direction
            best_metric_name = metric_name

    if best_model is None or best_params is None or best_score is None:
        raise RuntimeError("Model tuning did not produce a valid candidate.")

    selection_summary = {
        "best_params": best_params,
        "selection_metric": best_metric_name,
        "selection_direction": best_direction,
        "selection_value": best_score,
    }
    return best_model, selection_summary, tuning_results


def retrain_final_model(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    target_column: str,
    best_params: dict[str, Any],
) -> HistGradientBoostingClassifier:
    """Retrain the selected model on train+valid."""
    combined = pd.concat([train_frame, valid_frame], axis=0, ignore_index=True)
    train_x, train_y = split_xy(combined, target_column)
    sample_weight = compute_sample_weight(class_weight="balanced", y=train_y)
    model = build_model(best_params)
    model.fit(train_x, train_y, sample_weight=sample_weight)
    return model


def save_model_comparison(baseline_metrics: dict[str, Any], main_metrics: dict[str, Any]) -> None:
    """Save a flat comparison table for baseline vs main model."""
    rows = []
    for model_name, metrics in [("baseline", baseline_metrics["test"]), ("main_model", main_metrics["test"])]:
        rows.append(
            {
                "model": model_name,
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "recall_at_precision_90": metrics.get("recall_at_precision_90"),
                "precision_at_recall_80": metrics.get("precision_at_recall_80"),
                "log_loss": metrics.get("log_loss"),
                "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
            }
        )

    COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with COMPARISON_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_model_report(
    feature_metadata: dict[str, Any],
    baseline_metrics: dict[str, Any],
    main_metrics: dict[str, Any],
    selection_summary: dict[str, Any],
) -> str:
    """Render the main model report."""
    baseline_test = baseline_metrics["test"]
    main_test = main_metrics["test"]
    selection_metric = selection_summary["selection_metric"]
    selection_value = selection_summary["selection_value"]
    split_metadata = feature_metadata["split_metadata"]

    model_choice_notes = [
        "В качестве основной модели выбран `HistGradientBoostingClassifier`, потому что текущий feature pipeline уже отдаёт числовую матрицу признаков, и бустинг по деревьям обычно сильнее линейного baseline на табличных fraud-задачах.",
        "Модель устойчиво работает в существующем sklearn-стеке, не требует отдельного сложного рантайма и поддерживает умеренный воспроизводимый tuning.",
        "Accuracy не является ключевой метрикой: при сильном class imbalance она может вводить в заблуждение и скрывать слабое качество детекции fraud.",
        "PR-AUC, Recall и Precision важнее, потому что задача сводится к качественному отбору редкого positive-класса и контролю ложных тревог.",
    ]

    tradeoff_notes = [
        f"Выбор гиперпараметров делался по validation `{selection_metric}` со значением `{selection_value}`.",
        "Recall показывает, сколько fraud-кейсов модель не пропускает.",
        "Precision показывает, насколько дорогими будут алерты для ручной проверки или downstream anti-fraud rules.",
        "Даже если бустинг улучшает ranking, рабочий threshold всё равно нужно будет подбирать под бизнес-стоимость false positives vs false negatives.",
    ]
    data_limitations = []
    if baseline_test["pr_auc"] is None or main_test["pr_auc"] is None:
        data_limitations.append(
            "На текущем sample valid/test не содержат fraud-класс, поэтому ROC-AUC, PR-AUC и curve-based сравнение недоступны."
        )
        data_limitations.append(
            "Выбор основной модели сейчас опирается на устойчивость алгоритма и пригодность для табличных данных, а не на статистически сильное превосходство по holdout-метрикам."
        )

    lines = [
        "# Main Model Report",
        "",
        "- Final model: `HistGradientBoostingClassifier`",
        f"- Selection metric: `{selection_metric}`",
        f"- Best params: `{selection_summary['best_params']}`",
        f"- Train rows: `{split_metadata['train_rows']}`",
        f"- Valid rows: `{split_metadata['valid_rows']}`",
        f"- Test rows: `{split_metadata['test_rows']}`",
        "",
        "## Why This Model Was Chosen",
        "",
    ]
    lines.extend(f"- {note}" for note in model_choice_notes)

    lines.extend(
        [
            "",
            "## Baseline vs Main Model",
            "",
            f"- Baseline test ROC-AUC: `{baseline_test['roc_auc']}`",
            f"- Main model test ROC-AUC: `{main_test['roc_auc']}`",
            f"- Baseline test PR-AUC: `{baseline_test['pr_auc']}`",
            f"- Main model test PR-AUC: `{main_test['pr_auc']}`",
            f"- Baseline test Precision: `{baseline_test['precision']:.4f}`",
            f"- Main model test Precision: `{main_test['precision']:.4f}`",
            f"- Baseline test Recall: `{baseline_test['recall']:.4f}`",
            f"- Main model test Recall: `{main_test['recall']:.4f}`",
            f"- Baseline test F1: `{baseline_test['f1']:.4f}`",
            f"- Main model test F1: `{main_test['f1']:.4f}`",
            f"- Baseline recall@precision>=0.90: `{baseline_test.get('recall_at_precision_90')}`",
            f"- Main recall@precision>=0.90: `{main_test.get('recall_at_precision_90')}`",
            f"- Baseline precision@recall>=0.80: `{baseline_test.get('precision_at_recall_80')}`",
            f"- Main precision@recall>=0.80: `{main_test.get('precision_at_recall_80')}`",
            "",
            "## Data Limitations",
            "",
        ]
    )
    if data_limitations:
        lines.extend(f"- {note}" for note in data_limitations)
    else:
        lines.append("- Holdout splits contain both classes, so ranking metrics are directly comparable.")

    lines.extend(
        [
            "",
            "## Precision / Recall Trade-Off",
            "",
        ]
    )
    lines.extend(f"- {note}" for note in tradeoff_notes)

    lines.extend(
        [
            "",
            "## Error Analysis",
            "",
            f"- Baseline confusion matrix: `{baseline_test['confusion_matrix']}`",
            f"- Main model confusion matrix: `{main_test['confusion_matrix']}`",
            "- Если ranking-метрики на реальном датасете вырастут, это будет сигналом, что бустинг лучше ловит нелинейные fraud-паттерны и взаимодействия признаков.",
            "",
            "## Artifacts",
            "",
            f"- Final model: `{MAIN_MODEL_PATH}`",
            f"- Main metrics: `{MAIN_METRICS_PATH}`",
            f"- Model comparison: `{COMPARISON_PATH}`",
            f"- Validation ROC curve: `{main_metrics['valid']['roc_curve_plot']}`",
            f"- Validation PR curve: `{main_metrics['valid']['pr_curve_plot']}`",
            f"- Test ROC curve: `{main_metrics['test']['roc_curve_plot']}`",
            f"- Test PR curve: `{main_metrics['test']['pr_curve_plot']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train and compare the main fraud model.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    configure_logging()
    parse_args()

    feature_metadata = load_feature_metadata()
    target_column = feature_metadata["target_column"]
    if target_column is None:
        raise ValueError("Target column is missing in feature metadata. Model training requires a target.")

    splits = load_processed_splits()
    baseline_metrics = load_baseline_metrics()

    _, selection_summary, tuning_results = tune_model(
        train_frame=splits["train"],
        valid_frame=splits["valid"],
        target_column=target_column,
    )
    final_model = retrain_final_model(
        train_frame=splits["train"],
        valid_frame=splits["valid"],
        target_column=target_column,
        best_params=selection_summary["best_params"],
    )

    valid_metrics = evaluate_split(final_model, splits["valid"], target_column, "valid_main")
    test_metrics = evaluate_split(final_model, splits["test"], target_column, "test_main")
    main_metrics = {
        "model": "hist_gradient_boosting",
        "target_column": target_column,
        "selection_summary": selection_summary,
        "tuning_results": tuning_results,
        "valid": valid_metrics,
        "test": test_metrics,
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": final_model,
            "target_column": target_column,
            "feature_names": feature_metadata["final_features"],
            "selection_summary": selection_summary,
        },
        MAIN_MODEL_PATH,
    )
    MAIN_METRICS_PATH.write_text(
        json.dumps(main_metrics, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    save_model_comparison(baseline_metrics, main_metrics)
    MODEL_REPORT_PATH.write_text(
        build_model_report(feature_metadata, baseline_metrics, main_metrics, selection_summary),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
