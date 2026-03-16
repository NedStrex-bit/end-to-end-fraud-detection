"""Time-based feature helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd


def add_time_features(dataframe: pd.DataFrame, timestamp_column: str | None) -> tuple[pd.DataFrame, list[str], list[str], pd.Series | None]:
    """Add simple calendar features derived from a timestamp column."""
    if timestamp_column is None or timestamp_column not in dataframe.columns:
        return dataframe.copy(), [], ["Timestamp column was not detected; calendar features were skipped."], None

    parsed_timestamp = pd.to_datetime(dataframe[timestamp_column], errors="coerce")
    if parsed_timestamp.notna().sum() == 0:
        return (
            dataframe.copy(),
            [],
            [f"Timestamp column '{timestamp_column}' could not be parsed; calendar features were skipped."],
            None,
        )

    result = dataframe.copy()
    created_features = ["event_hour", "event_day_of_week", "is_weekend"]
    result["event_hour"] = parsed_timestamp.dt.hour
    result["event_day_of_week"] = parsed_timestamp.dt.dayofweek
    result["is_weekend"] = parsed_timestamp.dt.dayofweek.isin([5, 6]).astype("Int64")
    return result, created_features, [], parsed_timestamp
