"""Initial dataset inspection helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd


TARGET_PATTERNS = ("is_fraud", "fraud", "target", "label", "class")
TIME_PATTERNS = (
    "timestamp",
    "datetime",
    "event_time",
    "transaction_time",
    "trans_date_trans_time",
    "unix_time",
    "date",
    "time",
)
USER_ID_PATTERNS = ("user_id", "customer_id", "account_id", "client_id", "cc_num")
TRANSACTION_ID_PATTERNS = ("transaction_id", "tx_id", "trans_num", "id")


def count_missing_values(dataframe: pd.DataFrame) -> dict[str, int]:
    """Return missing value counts for all columns."""
    return {column: int(count) for column, count in dataframe.isna().sum().items()}


def count_duplicate_rows(dataframe: pd.DataFrame) -> int:
    """Return the number of duplicated rows."""
    return int(dataframe.duplicated().sum())


def get_numeric_columns(dataframe: pd.DataFrame) -> list[str]:
    """Return numeric columns."""
    return dataframe.select_dtypes(include=["number"]).columns.tolist()


def get_categorical_columns(dataframe: pd.DataFrame) -> list[str]:
    """Return categorical-like columns."""
    return dataframe.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


def normalize_column_name(column_name: str) -> str:
    """Normalize column name for heuristic matching."""
    return column_name.strip().lower().replace(" ", "_")


def find_columns_by_patterns(columns: list[str], patterns: tuple[str, ...]) -> list[str]:
    """Find columns whose normalized names contain one of the known patterns."""
    matches: list[str] = []
    for column in columns:
        normalized = normalize_column_name(column)
        if any(pattern in normalized for pattern in patterns):
            matches.append(column)
    return matches


def detect_target_candidates(dataframe: pd.DataFrame) -> list[str]:
    """Detect likely target columns."""
    return find_columns_by_patterns(dataframe.columns.tolist(), TARGET_PATTERNS)


def detect_timestamp_candidates(dataframe: pd.DataFrame) -> list[str]:
    """Detect likely timestamp columns."""
    return find_columns_by_patterns(dataframe.columns.tolist(), TIME_PATTERNS)


def detect_user_id_candidates(dataframe: pd.DataFrame) -> list[str]:
    """Detect likely user identifier columns."""
    return find_columns_by_patterns(dataframe.columns.tolist(), USER_ID_PATTERNS)


def detect_transaction_id_candidates(dataframe: pd.DataFrame) -> list[str]:
    """Detect likely transaction identifier columns."""
    columns = dataframe.columns.tolist()
    matches: list[str] = []
    user_id_candidates = set(detect_user_id_candidates(dataframe))

    for column in columns:
        normalized = normalize_column_name(column)
        is_direct_transaction_match = any(
            pattern in normalized for pattern in TRANSACTION_ID_PATTERNS if pattern != "id"
        )
        is_generic_id_match = normalized == "id"

        if is_direct_transaction_match or is_generic_id_match:
            matches.append(column)
            continue

        if normalized.endswith("_id") and column not in user_id_candidates:
            matches.append(column)

    return matches


def get_first_candidate(dataframe: pd.DataFrame, role: str) -> str | None:
    """Return the first detected column candidate for a given semantic role."""
    detectors = {
        "target": detect_target_candidates,
        "timestamp": detect_timestamp_candidates,
        "user_id": detect_user_id_candidates,
        "transaction_id": detect_transaction_id_candidates,
    }
    detector = detectors.get(role)
    if detector is None:
        raise ValueError(f"Unsupported role: '{role}'.")

    candidates = detector(dataframe)
    return candidates[0] if candidates else None


def build_data_quality_summary(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Create a compact summary for the initial data audit."""
    return {
        "row_count": int(len(dataframe)),
        "column_count": int(dataframe.shape[1]),
        "columns": dataframe.columns.tolist(),
        "dtypes": {column: str(dtype) for column, dtype in dataframe.dtypes.items()},
        "missing_values": count_missing_values(dataframe),
        "duplicate_rows": count_duplicate_rows(dataframe),
        "numeric_columns": get_numeric_columns(dataframe),
        "categorical_columns": get_categorical_columns(dataframe),
        "candidate_columns": {
            "target": detect_target_candidates(dataframe),
            "timestamp": detect_timestamp_candidates(dataframe),
            "user_id": detect_user_id_candidates(dataframe),
            "transaction_id": detect_transaction_id_candidates(dataframe),
        },
    }
