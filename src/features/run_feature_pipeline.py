"""Run the feature engineering pipeline and save train-ready datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.load_data import DEFAULT_RAW_DATA_DIR, configure_logging, load_dataset, resolve_data_path
from src.data.preprocess import get_first_candidate
from src.features.build_features import (
    build_feature_frame,
    save_processed_splits,
    select_model_columns,
    split_feature_frame,
)
from src.features.encoders import build_feature_encoder, transform_frame


REPORTS_DIR = Path("artifacts/reports")
FEATURE_LIST_PATH = REPORTS_DIR / "feature_list.json"
FEATURE_REPORT_PATH = REPORTS_DIR / "feature_report.md"


def _json_default(value: Any) -> Any:
    """Convert numpy scalar values to Python primitives."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _can_stratify(target_series: pd.Series | None) -> bool:
    """Check if target supports stable stratified split."""
    if target_series is None:
        return False
    counts = target_series.value_counts(dropna=True)
    return counts.shape[0] > 1 and int(counts.min()) >= 3


def build_processed_splits(
    split_frames: dict[str, pd.DataFrame],
    target_column: str | None,
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Fit encoder on train and transform all splits into train-ready frames."""
    train_frame = split_frames["train"].drop(columns=["_target"], errors="ignore")
    encoder = build_feature_encoder(train_frame, numeric_features, categorical_features)

    processed_splits: dict[str, pd.DataFrame] = {}
    for split_name, split_frame in split_frames.items():
        raw_features = split_frame.drop(columns=["_target"], errors="ignore")
        transformed = transform_frame(encoder, raw_features).reset_index(drop=True)
        if target_column is not None and "_target" in split_frame.columns:
            transformed[target_column] = split_frame["_target"].reset_index(drop=True).to_numpy()
        processed_splits[split_name] = transformed
    return processed_splits, encoder.output_features


def build_feature_report(
    dataset_path: Path,
    feature_metadata: dict[str, Any],
) -> str:
    """Render the feature engineering report."""
    required_timestamp_features = [
        "event_hour",
        "event_day_of_week",
        "is_weekend",
    ]
    required_user_time_features = [
        "user_prior_transaction_count",
        "seconds_since_prev_transaction",
        "user_amount_mean_prior",
        "user_amount_std_prior",
        "amount_vs_user_mean",
        "user_txn_count_last_1h",
        "user_txn_count_last_24h",
    ]

    built_features = feature_metadata["built_features"]
    dropped_features = feature_metadata["dropped_features"]
    unavailable_features = feature_metadata["unavailable_features"]
    split_metadata = feature_metadata["split_metadata"]
    final_features = feature_metadata["final_features"]

    lines = [
        "# Feature Engineering Report",
        "",
        f"- Dataset: `{dataset_path}`",
        f"- Split method: `{split_metadata['split_method']}`",
        f"- Train rows: `{split_metadata['train_rows']}`",
        f"- Valid rows: `{split_metadata['valid_rows']}`",
        f"- Test rows: `{split_metadata['test_rows']}`",
        "",
        "## Built Features",
        "",
        f"- Engineered features: `{built_features if built_features else 'none'}`",
        f"- Final model-ready feature count: `{len(final_features)}`",
        "",
        "## Dropped Features",
        "",
        f"- Dropped from model inputs: `{dropped_features if dropped_features else 'none'}`",
        "",
        "## Conditional Features",
        "",
        f"- Require timestamp: `{required_timestamp_features}`",
        f"- Require user_id and timestamp: `{required_user_time_features}`",
        "",
        "## Leakage Prevention",
        "",
        "- Raw target is never used to build features.",
        "- Calendar and behavioral features are computed from transaction metadata only.",
        "- User behavioral features use only prior events via chronological ordering, `shift`, `diff`, and left-closed rolling windows.",
        "- Categorical encoding and numeric imputation are fitted on the train split only, then reused for valid/test.",
        "- Transaction identifiers and raw timestamps are excluded from direct model inputs.",
        "",
        "## Split Notes",
        "",
    ]

    if split_metadata.get("limitations"):
        lines.extend(f"- {item}" for item in split_metadata["limitations"])
    else:
        lines.append("- No extra split limitations recorded.")

    lines.extend(
        [
            "",
            "## Unavailable Or Skipped Features",
            "",
        ]
    )
    if unavailable_features:
        lines.extend(f"- {item}" for item in unavailable_features)
    else:
        lines.append("- All configured feature blocks were available.")

    lines.extend(
        [
            "",
            "## Final Encoded Features",
            "",
            f"- `{final_features}`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run the feature engineering pipeline.")
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Dataset filename inside data/raw. If omitted, the first supported file is used.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    configure_logging()
    args = parse_args()
    dataset_path = resolve_data_path(filename=args.file, data_dir=DEFAULT_RAW_DATA_DIR)
    dataframe = load_dataset(dataset_path)

    target_column = get_first_candidate(dataframe, "target")
    timestamp_column = get_first_candidate(dataframe, "timestamp")
    user_id_column = get_first_candidate(dataframe, "user_id")
    transaction_id_column = get_first_candidate(dataframe, "transaction_id")

    feature_result = build_feature_frame(
        dataframe=dataframe,
        target_column=target_column,
        timestamp_column=timestamp_column,
        user_id_column=user_id_column,
        transaction_id_column=transaction_id_column,
    )

    split_frames, split_metadata, split_notes = split_feature_frame(
        feature_frame=feature_result.feature_frame,
        target_series=feature_result.target_series,
        parsed_timestamp=feature_result.parsed_timestamp,
        target_column=target_column,
    )

    numeric_features, categorical_features = select_model_columns(
        feature_frame=split_frames["train"].drop(columns=["_target"], errors="ignore"),
        target_column=target_column,
        user_id_column=user_id_column,
    )

    processed_splits, final_features = build_processed_splits(
        split_frames=split_frames,
        target_column=target_column,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    processed_paths = save_processed_splits(processed_splits)

    feature_metadata = {
        "raw_dataset": str(dataset_path),
        "target_column": target_column,
        "timestamp_column": timestamp_column,
        "user_id_column": user_id_column,
        "transaction_id_column": transaction_id_column,
        "amount_column": feature_result.amount_column,
        "built_features": feature_result.built_features,
        "dropped_features": feature_result.dropped_features + ([user_id_column] if user_id_column else []),
        "unavailable_features": feature_result.unavailable_features + feature_result.notes + split_notes,
        "input_numeric_features": numeric_features,
        "input_categorical_features": categorical_features,
        "final_features": final_features,
        "processed_paths": processed_paths,
        "split_metadata": split_metadata,
        "stratification_enabled": _can_stratify(feature_result.target_series),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_LIST_PATH.write_text(
        json.dumps(feature_metadata, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    FEATURE_REPORT_PATH.write_text(build_feature_report(dataset_path, feature_metadata), encoding="utf-8")


if __name__ == "__main__":
    main()
