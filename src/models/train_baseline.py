"""Train and evaluate a baseline logistic regression model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.load_data import configure_logging
from src.models.evaluate import evaluate_predictions, plot_pr_curve, plot_roc_curve


PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("artifacts/reports")
MODELS_DIR = Path("artifacts/models")
FEATURE_LIST_PATH = REPORTS_DIR / "feature_list.json"
METRICS_PATH = REPORTS_DIR / "baseline_metrics.json"
REPORT_PATH = REPORTS_DIR / "baseline_report.md"
MODEL_PATH = MODELS_DIR / "baseline_logistic_regression.joblib"


def _json_default(value: Any) -> Any:
    """Convert numpy scalars to Python primitives."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def load_feature_metadata() -> dict[str, Any]:
    """Load metadata produced by the feature pipeline."""
    if not FEATURE_LIST_PATH.exists():
        raise FileNotFoundError(
            f"Feature metadata was not found at '{FEATURE_LIST_PATH}'. Run the feature pipeline first."
        )
    return json.loads(FEATURE_LIST_PATH.read_text(encoding="utf-8"))


def load_split(path: Path) -> pd.DataFrame:
    """Load a processed split."""
    if not path.exists():
        raise FileNotFoundError(f"Processed split was not found: '{path}'. Run the feature pipeline first.")
    return pd.read_parquet(path)


def load_processed_splits() -> dict[str, pd.DataFrame]:
    """Load train/valid/test splits."""
    return {
        "train": load_split(PROCESSED_DIR / "train.parquet"),
        "valid": load_split(PROCESSED_DIR / "valid.parquet"),
        "test": load_split(PROCESSED_DIR / "test.parquet"),
    }


def build_model() -> Pipeline:
    """Create the baseline logistic regression pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )


def split_xy(frame: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """Split a processed frame into X and y."""
    if target_column not in frame.columns:
        raise ValueError(f"Target column '{target_column}' was not found in processed data.")
    return frame.drop(columns=[target_column]), frame[target_column].astype(int)


def evaluate_split(model: Pipeline, frame: pd.DataFrame, target_column: str, split_name: str) -> dict[str, Any]:
    """Evaluate model on a split and generate plots when possible."""
    features, target = split_xy(frame, target_column)
    score = model.predict_proba(features)[:, 1]
    prediction = (score >= 0.5).astype(int)
    metrics = evaluate_predictions(target.to_numpy(), prediction, score)
    metrics["roc_curve_plot"] = plot_roc_curve(target.to_numpy(), score, f"{split_name}_roc_curve.png")
    metrics["pr_curve_plot"] = plot_pr_curve(target.to_numpy(), score, f"{split_name}_pr_curve.png")
    return metrics


def build_markdown_report(
    feature_metadata: dict[str, Any],
    metrics: dict[str, Any],
) -> str:
    """Create the baseline markdown report."""
    target_column = feature_metadata["target_column"]
    final_features = feature_metadata["final_features"]
    valid_metrics = metrics["valid"]
    test_metrics = metrics["test"]
    split_metadata = feature_metadata["split_metadata"]

    error_analysis: list[str] = []
    valid_cm = valid_metrics["confusion_matrix"]
    test_cm = test_metrics["confusion_matrix"]
    test_positive_support = test_metrics["classification_report"].get("1", {}).get("support", 0.0)

    if test_positive_support == 0:
        error_analysis.append("Test split не содержит fraud-примеров, поэтому оценка false negatives и ranking-метрик ограничена.")
    elif len(test_cm) == 2 and len(test_cm[0]) == 2:
        tn, fp = test_cm[0]
        fn, tp = test_cm[1]
        if fp > fn:
            error_analysis.append("На test доминируют false positives: baseline переоценивает fraud-риск.")
        elif fn > fp:
            error_analysis.append("На test доминируют false negatives: baseline пропускает fraud-кейсы.")
        else:
            error_analysis.append("На test false positives и false negatives находятся на сопоставимом уровне.")
    else:
        error_analysis.append("Test split содержит один класс или слишком мал, поэтому error analysis ограничен.")

    suitability_notes = [
        "Baseline пригоден как sanity check для всего pipeline: признаки, split и сохранение артефактов работают end-to-end.",
        "Для fraud detection accuracy не является ключевой метрикой, потому что при дисбалансе классов она может быть высокой даже при бесполезной модели.",
        "PR-AUC важнее ROC-AUC, когда fraud-класс редкий: она лучше показывает качество отбора подозрительных транзакций.",
        "Recall критичен, если пропуск fraud дорогой; Precision важен, если слишком много ложных тревог перегружает антифрод-процесс.",
        "Линейная Logistic Regression ограничена для сложных нелинейных взаимодействий, поэтому следующим шагом нужен более сильный tabular model.",
    ]

    lines = [
        "# Baseline Model Report",
        "",
        "- Model: `LogisticRegression(class_weight='balanced')`",
        f"- Target column: `{target_column}`",
        f"- Number of model features: `{len(final_features)}`",
        f"- Train rows: `{split_metadata['train_rows']}`",
        f"- Valid rows: `{split_metadata['valid_rows']}`",
        f"- Test rows: `{split_metadata['test_rows']}`",
        "",
        "## Why Accuracy Is Not The Focus",
        "",
    ]
    lines.extend(f"- {note}" for note in suitability_notes)

    lines.extend(
        [
            "",
            "## Validation Metrics",
            "",
            f"- ROC-AUC: `{valid_metrics['roc_auc']}`",
            f"- PR-AUC: `{valid_metrics['pr_auc']}`",
            f"- Precision: `{valid_metrics['precision']:.4f}`",
            f"- Recall: `{valid_metrics['recall']:.4f}`",
            f"- F1: `{valid_metrics['f1']:.4f}`",
            f"- Confusion matrix: `{valid_metrics['confusion_matrix']}`",
            "",
            "## Test Metrics",
            "",
            f"- ROC-AUC: `{test_metrics['roc_auc']}`",
            f"- PR-AUC: `{test_metrics['pr_auc']}`",
            f"- Precision: `{test_metrics['precision']:.4f}`",
            f"- Recall: `{test_metrics['recall']:.4f}`",
            f"- F1: `{test_metrics['f1']:.4f}`",
            f"- Confusion matrix: `{test_metrics['confusion_matrix']}`",
            "",
            "## Error Analysis",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in error_analysis)

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Model: `{MODEL_PATH}`",
            f"- Metrics JSON: `{METRICS_PATH}`",
            f"- Validation ROC curve: `{valid_metrics['roc_curve_plot']}`",
            f"- Validation PR curve: `{valid_metrics['pr_curve_plot']}`",
            f"- Test ROC curve: `{test_metrics['roc_curve_plot']}`",
            f"- Test PR curve: `{test_metrics['pr_curve_plot']}`",
            "",
            "## Next Step",
            "",
            "- Следующим этапом нужен более сильный tabular model, например tree boosting, чтобы поймать нелинейные взаимодействия и редкие fraud-паттерны.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train and evaluate the baseline fraud model.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    configure_logging()
    parse_args()

    feature_metadata = load_feature_metadata()
    target_column = feature_metadata["target_column"]
    if target_column is None:
        raise ValueError("Target column is missing in feature metadata. Baseline training requires a target.")

    splits = load_processed_splits()
    train_x, train_y = split_xy(splits["train"], target_column)

    model = build_model()
    model.fit(train_x, train_y)

    metrics = {
        "model": "baseline_logistic_regression",
        "target_column": target_column,
        "feature_count": len(train_x.columns),
        "valid": evaluate_split(model, splits["valid"], target_column, "valid"),
        "test": evaluate_split(model, splits["test"], target_column, "test"),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_names": train_x.columns.tolist(),
            "target_column": target_column,
        },
        MODEL_PATH,
    )
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    REPORT_PATH.write_text(build_markdown_report(feature_metadata, metrics), encoding="utf-8")


if __name__ == "__main__":
    main()
