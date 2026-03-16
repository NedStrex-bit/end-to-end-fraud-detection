"""Feature engineering pipeline helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.preprocess import get_categorical_columns, get_numeric_columns, normalize_column_name
from src.features.time_features import add_time_features


DEFAULT_PROCESSED_DIR = Path("data/processed")
AMOUNT_PATTERNS = ("amount", "transaction_amount", "value", "amt")
DROP_EXACT_COLUMNS = {"unnamed:_0", "unnamed_0"}
DROP_PATTERN_COLUMNS = ("first", "last", "street", "job", "dob", "trans_num", "unix_time")
MAX_CATEGORICAL_CARDINALITY = 50
MAX_CATEGORICAL_CARDINALITY_RATIO = 0.01


@dataclass
class FeatureBuildResult:
    """Container for engineered features and metadata."""

    feature_frame: pd.DataFrame
    target_series: pd.Series | None
    parsed_timestamp: pd.Series | None
    built_features: list[str]
    dropped_features: list[str]
    unavailable_features: list[str]
    notes: list[str]
    amount_column: str | None
    target_column: str | None
    timestamp_column: str | None
    user_id_column: str | None
    transaction_id_column: str | None


def detect_amount_column(dataframe: pd.DataFrame) -> str | None:
    """Detect an amount-like numeric column."""
    for column in get_numeric_columns(dataframe):
        normalized = normalize_column_name(column)
        if any(pattern in normalized for pattern in AMOUNT_PATTERNS):
            return column
    return None


def prepare_binary_target(dataframe: pd.DataFrame, target_column: str | None) -> pd.Series | None:
    """Prepare a binary target series when available."""
    if target_column is None or target_column not in dataframe.columns:
        return None

    target = pd.to_numeric(dataframe[target_column], errors="coerce")
    valid_values = target.dropna().unique().tolist()
    if not valid_values:
        return None
    if set(valid_values).issubset({0, 1}):
        return target.astype("Int64")
    return None


def add_amount_features(dataframe: pd.DataFrame, amount_column: str | None) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Add amount-oriented features when a suitable base column exists."""
    if amount_column is None or amount_column not in dataframe.columns:
        return dataframe.copy(), [], ["Amount-like column was not detected; amount-based features were skipped."]

    result = dataframe.copy()
    amount_values = pd.to_numeric(result[amount_column], errors="coerce")
    result["amount_log1p"] = np.log1p(amount_values.clip(lower=0))
    return result, ["amount_log1p"], []


def _compute_window_counts(group: pd.DataFrame, timestamp_column: str, window: str) -> pd.Series:
    """Count prior user transactions in a rolling time window."""
    timestamps = pd.to_datetime(group[timestamp_column], errors="coerce").astype("int64").to_numpy()
    window_ns = pd.Timedelta(window).value
    counts = np.zeros(len(group), dtype="float64")

    for index, current_timestamp in enumerate(timestamps):
        left_bound = np.searchsorted(timestamps, current_timestamp - window_ns, side="left")
        counts[index] = float(index - left_bound)

    return pd.Series(counts, index=group.index, dtype="float64")


def _assign_window_counts(result: pd.DataFrame, user_id_column: str, window: str) -> pd.Series:
    """Assign prior transaction counts per user for a rolling time window."""
    counts = pd.Series(np.zeros(len(result), dtype="float64"), index=result.index)
    for _, group in result.groupby(user_id_column, sort=False):
        counts.loc[group.index] = _compute_window_counts(group, timestamp_column="_parsed_timestamp", window=window)
    return counts


def add_behavioral_features(
    dataframe: pd.DataFrame,
    amount_column: str | None,
    timestamp_column: str | None,
    user_id_column: str | None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Add user history features using only past events."""
    if user_id_column is None or user_id_column not in dataframe.columns:
        return dataframe.copy(), [], ["User ID column was not detected; behavioral features were skipped."]
    if timestamp_column is None or timestamp_column not in dataframe.columns:
        return dataframe.copy(), [], ["Timestamp column was not detected; behavioral features were skipped."]

    result = dataframe.copy()
    parsed_timestamp = pd.to_datetime(result[timestamp_column], errors="coerce")
    if parsed_timestamp.notna().sum() == 0:
        return result, [], [f"Timestamp column '{timestamp_column}' could not be parsed; behavioral features were skipped."]

    result["_parsed_timestamp"] = parsed_timestamp
    result["_original_order"] = np.arange(len(result))
    sort_columns = ["_parsed_timestamp", "_original_order"]
    result = result.sort_values(sort_columns).reset_index(drop=True)

    group = result.groupby(user_id_column, sort=False)
    created_features = [
        "user_prior_transaction_count",
        "seconds_since_prev_transaction",
        "user_amount_mean_prior",
        "user_amount_std_prior",
        "amount_vs_user_mean",
        "user_txn_count_last_1h",
        "user_txn_count_last_24h",
    ]

    # These features are causal: each row uses only prior rows for the same user.
    result["user_prior_transaction_count"] = group.cumcount()
    result["seconds_since_prev_transaction"] = (
        group["_parsed_timestamp"].diff().dt.total_seconds()
    )

    if amount_column is not None and amount_column in result.columns:
        numeric_amount = pd.to_numeric(result[amount_column], errors="coerce")
        result["_numeric_amount"] = numeric_amount
        shifted_amount = group["_numeric_amount"].shift(1)
        result["user_amount_mean_prior"] = shifted_amount.groupby(result[user_id_column]).expanding().mean().reset_index(level=0, drop=True)
        result["user_amount_std_prior"] = shifted_amount.groupby(result[user_id_column]).expanding().std().reset_index(level=0, drop=True)
        result["amount_vs_user_mean"] = result["_numeric_amount"] - result["user_amount_mean_prior"]
    else:
        created_features = [
            feature
            for feature in created_features
            if feature not in {"user_amount_mean_prior", "user_amount_std_prior", "amount_vs_user_mean"}
        ]

    result["user_txn_count_last_1h"] = _assign_window_counts(result, user_id_column=user_id_column, window="1h")
    result["user_txn_count_last_24h"] = _assign_window_counts(result, user_id_column=user_id_column, window="24h")

    result = result.sort_values("_original_order").drop(columns=["_original_order"])
    if "_numeric_amount" in result.columns:
        result = result.drop(columns=["_numeric_amount"])
    return result, created_features, []


def build_feature_frame(
    dataframe: pd.DataFrame,
    target_column: str | None,
    timestamp_column: str | None,
    user_id_column: str | None,
    transaction_id_column: str | None,
) -> FeatureBuildResult:
    """Build a feature-ready frame and track metadata."""
    notes: list[str] = []
    unavailable_features: list[str] = []
    built_features: list[str] = []

    target_series = prepare_binary_target(dataframe, target_column)
    if target_column and target_series is None:
        notes.append(f"Target column '{target_column}' is not binary-like; stratification and target-aware reports may be limited.")
    elif target_column is None:
        notes.append("Target column was not detected; split will run without stratification.")

    result = dataframe.copy()

    result, time_features, time_notes, parsed_timestamp = add_time_features(result, timestamp_column)
    built_features.extend(time_features)
    unavailable_features.extend(time_notes)

    amount_column = detect_amount_column(result)
    result, amount_features, amount_notes = add_amount_features(result, amount_column)
    built_features.extend(amount_features)
    unavailable_features.extend(amount_notes)

    result, behavioral_features, behavioral_notes = add_behavioral_features(
        result,
        amount_column=amount_column,
        timestamp_column=timestamp_column,
        user_id_column=user_id_column,
    )
    built_features.extend(behavioral_features)
    unavailable_features.extend(behavioral_notes)

    dropped_features = [
        column
        for column in [target_column, timestamp_column, transaction_id_column]
        if column is not None and column in result.columns
    ]
    feature_frame = result.drop(columns=dropped_features, errors="ignore")

    # Drop obvious raw identifiers/PII and extremely high-cardinality categoricals before split/encoding.
    memory_heavy_columns: list[str] = []
    row_count = max(len(feature_frame), 1)
    for column in feature_frame.columns:
        normalized = normalize_column_name(column).replace(" ", "_")
        if normalized in DROP_EXACT_COLUMNS:
            memory_heavy_columns.append(column)
            continue
        if any(pattern == normalized or pattern in normalized for pattern in DROP_PATTERN_COLUMNS):
            memory_heavy_columns.append(column)
            continue
        if column == user_id_column:
            continue
        if pd.api.types.is_object_dtype(feature_frame[column]) or pd.api.types.is_string_dtype(feature_frame[column]):
            cardinality = feature_frame[column].nunique(dropna=True)
            if cardinality > MAX_CATEGORICAL_CARDINALITY or (cardinality / row_count) > MAX_CATEGORICAL_CARDINALITY_RATIO:
                memory_heavy_columns.append(column)

    if memory_heavy_columns:
        unique_columns = list(dict.fromkeys(memory_heavy_columns))
        feature_frame = feature_frame.drop(columns=unique_columns, errors="ignore")
        dropped_features.extend(unique_columns)
        notes.append(
            f"High-cardinality or PII-like raw columns were dropped for memory-safe training: {unique_columns}."
        )

    if parsed_timestamp is not None:
        feature_frame["_parsed_timestamp"] = pd.to_datetime(dataframe[timestamp_column], errors="coerce")
    elif "_parsed_timestamp" in feature_frame.columns:
        parsed_timestamp = pd.to_datetime(feature_frame["_parsed_timestamp"], errors="coerce")
    else:
        parsed_timestamp = None

    return FeatureBuildResult(
        feature_frame=feature_frame,
        target_series=target_series,
        parsed_timestamp=parsed_timestamp,
        built_features=built_features,
        dropped_features=dropped_features,
        unavailable_features=unavailable_features,
        notes=notes,
        amount_column=amount_column,
        target_column=target_column,
        timestamp_column=timestamp_column,
        user_id_column=user_id_column,
        transaction_id_column=transaction_id_column,
    )


def _resolve_split_sizes(row_count: int, train_ratio: float, valid_ratio: float) -> tuple[int, int, int]:
    """Resolve split sizes with at least one row per split when possible."""
    if row_count < 3:
        raise ValueError("At least 3 rows are required to create train/valid/test splits.")

    train_end = max(1, int(row_count * train_ratio))
    valid_end = max(train_end + 1, int(row_count * (train_ratio + valid_ratio)))
    valid_end = min(valid_end, row_count - 1)
    train_end = min(train_end, valid_end - 1)
    return train_end, valid_end, row_count - valid_end


def split_feature_frame(
    feature_frame: pd.DataFrame,
    target_series: pd.Series | None,
    parsed_timestamp: pd.Series | None,
    target_column: str | None,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    random_state: int = 42,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any], list[str]]:
    """Split data into train/valid/test."""
    notes: list[str] = []
    split_metadata: dict[str, Any] = {}

    if parsed_timestamp is not None and parsed_timestamp.notna().sum() >= len(feature_frame) * 0.8:
        sortable = feature_frame.copy()
        sortable["_parsed_timestamp"] = parsed_timestamp
        sortable["_target"] = target_series
        sortable = sortable.sort_values(["_parsed_timestamp"]).reset_index(drop=True)

        train_end, valid_end, test_size = _resolve_split_sizes(len(sortable), train_ratio, valid_ratio)
        train_frame = sortable.iloc[:train_end].copy()
        valid_frame = sortable.iloc[train_end:valid_end].copy()
        test_frame = sortable.iloc[valid_end:].copy()

        notes.append("Time-based split was used because a timestamp column is available.")
        split_metadata["split_method"] = "time_based"
        split_metadata["limitations"] = []
    else:
        split_source = feature_frame.copy()
        split_source["_target"] = target_series
        stratify_target = target_series if target_series is not None and target_series.nunique(dropna=True) > 1 else None
        if parsed_timestamp is None:
            notes.append("Timestamp column is unavailable; fallback to stratified/random split.")
        else:
            notes.append("Timestamp coverage is insufficient for robust time-based split; fallback to stratified/random split.")

        train_frame, temp_frame = train_test_split(
            split_source,
            test_size=1 - train_ratio,
            random_state=random_state,
            stratify=stratify_target,
        )

        temp_target = temp_frame["_target"] if "_target" in temp_frame.columns else None
        temp_stratify = temp_target if temp_target is not None and temp_target.nunique(dropna=True) > 1 else None
        valid_share = valid_ratio / (1 - train_ratio)
        valid_frame, test_frame = train_test_split(
            temp_frame,
            test_size=1 - valid_share,
            random_state=random_state,
            stratify=temp_stratify,
        )

        train_frame = train_frame.reset_index(drop=True)
        valid_frame = valid_frame.reset_index(drop=True)
        test_frame = test_frame.reset_index(drop=True)
        split_metadata["split_method"] = "stratified" if stratify_target is not None else "random"
        split_metadata["limitations"] = [
            "Without a reliable timestamp, the split does not fully simulate future production data."
        ]

    split_frames = {
        "train": train_frame,
        "valid": valid_frame,
        "test": test_frame,
    }
    for name, frame in split_frames.items():
        split_metadata[f"{name}_rows"] = int(len(frame))
    split_metadata["target_column"] = target_column
    return split_frames, split_metadata, notes


def select_model_columns(
    feature_frame: pd.DataFrame,
    target_column: str | None,
    user_id_column: str | None,
) -> tuple[list[str], list[str]]:
    """Select numeric and categorical columns safe for model ingestion."""
    excluded = {"_parsed_timestamp"}
    if user_id_column:
        excluded.add(user_id_column)

    numeric_features = [
        column for column in get_numeric_columns(feature_frame) if column not in excluded and column != target_column
    ]
    categorical_features = [
        column
        for column in get_categorical_columns(feature_frame)
        if column not in excluded
        and column != target_column
        and feature_frame[column].nunique(dropna=True) <= MAX_CATEGORICAL_CARDINALITY
    ]
    return numeric_features, categorical_features


def save_processed_splits(processed_splits: dict[str, pd.DataFrame], output_dir: Path = DEFAULT_PROCESSED_DIR) -> dict[str, str]:
    """Save processed splits to parquet files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for split_name, frame in processed_splits.items():
        path = output_dir / f"{split_name}.parquet"
        frame.to_parquet(path, index=False)
        paths[split_name] = str(path)
    return paths
