"""Run explainability and error analysis for the final model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.load_data import configure_logging
from src.models.error_analysis import build_error_analysis_report, build_explainability_report
from src.models.explainability import (
    basic_error_findings,
    build_prediction_frame,
    compute_model_feature_importance,
    confusion_matrix_as_list,
    export_error_tables,
    plot_feature_distribution_by_error_type,
    plot_feature_importance,
)
from src.models.run_threshold_analysis import score_split
from src.models.train_baseline import load_processed_splits


REPORTS_DIR = Path("artifacts/reports")
MODELS_DIR = Path("artifacts/models")
THRESHOLD_METRICS_PATH = REPORTS_DIR / "threshold_metrics.csv"
THRESHOLD_REPORT_PATH = REPORTS_DIR / "threshold_report.md"
EXPLAINABILITY_REPORT_PATH = REPORTS_DIR / "explainability_report.md"
ERROR_REPORT_PATH = REPORTS_DIR / "error_analysis_report.md"
MAIN_MODEL_PATH = MODELS_DIR / "main_hist_gradient_boosting.joblib"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run explainability and error analysis for the final model.")
    parser.add_argument("--split", type=str, default="valid", choices=["train", "valid", "test"], help="Split to analyze.")
    parser.add_argument("--threshold", type=float, default=None, help="Override threshold. If omitted, use balanced threshold from threshold_metrics.csv.")
    return parser.parse_args()


def load_threshold_from_artifacts() -> float:
    """Load the recommended balanced threshold from threshold analysis output."""
    if THRESHOLD_METRICS_PATH.exists():
        metrics = pd.read_csv(THRESHOLD_METRICS_PATH)
        best = metrics.sort_values(["business_cost", "f1", "threshold"], ascending=[True, False, False]).iloc[0]
        return float(best["threshold"])
    return 0.5


def load_main_model_bundle() -> dict:
    """Load saved main model artifact."""
    if not MAIN_MODEL_PATH.exists():
        raise FileNotFoundError(f"Main model artifact not found at '{MAIN_MODEL_PATH}'. Run main model training first.")
    return joblib.load(MAIN_MODEL_PATH)


def choose_importance_dataset(
    splits: dict[str, pd.DataFrame],
    target_column: str,
    requested_split: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Choose a reasonable dataset for feature importance computation."""
    requested = splits[requested_split]
    requested_target = requested[target_column]
    notes: list[str] = []

    if len(requested) >= 10 and requested_target.nunique(dropna=True) > 1:
        return requested, notes

    combined = pd.concat([splits["train"], splits["valid"], splits["test"]], axis=0, ignore_index=True)
    notes.append(
        "Для feature importance использован объединённый train+valid+test, потому что выбранный split слишком мал или одноклассовый."
    )
    return combined, notes


def main() -> None:
    """CLI entry point."""
    configure_logging()
    args = parse_args()

    model_bundle = load_main_model_bundle()
    splits = load_processed_splits()
    split_frame = splits[args.split]
    threshold = args.threshold if args.threshold is not None else load_threshold_from_artifacts()

    feature_names = model_bundle["feature_names"]
    target_column = model_bundle["target_column"]
    model = model_bundle["model"]

    features = split_frame.drop(columns=[target_column]).reindex(columns=feature_names, fill_value=0.0)
    target = split_frame[target_column].astype(int)
    score = model.predict_proba(features)[:, 1]

    importance_frame_source, importance_notes = choose_importance_dataset(
        splits=splits,
        target_column=target_column,
        requested_split=args.split,
    )
    importance_features = importance_frame_source.drop(columns=[target_column]).reindex(columns=feature_names, fill_value=0.0)
    importance_target = importance_frame_source[target_column].astype(int)

    importance_frame, explainability_notes = compute_model_feature_importance(
        model=model,
        feature_frame=importance_features,
        target=importance_target,
    )
    explainability_notes = importance_notes + explainability_notes
    top_features = importance_frame.head(5).to_dict(orient="records")
    importance_plot_path = plot_feature_importance(importance_frame, top_n=min(10, len(importance_frame)))

    prediction_frame = build_prediction_frame(
        feature_frame=features,
        y_true=target,
        y_score=score,
        threshold=threshold,
    )
    exported_case_paths = export_error_tables(prediction_frame)

    plotted_features: list[str] = []
    for feature in importance_frame["feature"].head(3).tolist():
        if feature in prediction_frame.columns:
            plot_feature_distribution_by_error_type(
                analysis_frame=prediction_frame,
                feature=feature,
                filename=f"{feature}_by_error_type.png",
            )
            plotted_features.append(feature)

    findings = basic_error_findings(prediction_frame, [row["feature"] for row in top_features])

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    EXPLAINABILITY_REPORT_PATH.write_text(
        build_explainability_report(
            top_features=top_features,
            explainability_notes=explainability_notes,
            importance_plot_path=importance_plot_path,
            threshold=threshold,
        ),
        encoding="utf-8",
    )
    ERROR_REPORT_PATH.write_text(
        build_error_analysis_report(
            findings=findings,
            confusion_matrix=confusion_matrix_as_list(prediction_frame),
            summary_table_path=exported_case_paths["summary"],
            exported_case_paths=exported_case_paths,
            plotted_features=plotted_features,
            threshold=threshold,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
