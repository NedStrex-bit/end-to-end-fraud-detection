"""Reusable fraud prediction service."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.features.build_features import build_feature_frame


MODELS_DIR = Path("artifacts/models")
REPORTS_DIR = Path("artifacts/reports")
MODEL_PATH = MODELS_DIR / "main_hist_gradient_boosting.joblib"
FEATURE_LIST_PATH = REPORTS_DIR / "feature_list.json"
THRESHOLD_METRICS_PATH = REPORTS_DIR / "threshold_metrics.csv"


class FraudPredictor:
    """Thin inference wrapper around the trained model and feature metadata."""

    def __init__(self) -> None:
        self.model_bundle = self._load_model_bundle()
        self.feature_metadata = self._load_feature_metadata()
        self.threshold = self._load_threshold()
        self.model = self.model_bundle["model"]
        self.feature_names: list[str] = self.model_bundle["feature_names"]
        self.target_column: str = self.model_bundle["target_column"]
        self.timestamp_column: str | None = self.feature_metadata.get("timestamp_column")
        self.user_id_column: str | None = self.feature_metadata.get("user_id_column")
        self.transaction_id_column: str | None = self.feature_metadata.get("transaction_id_column")
        self.raw_categorical_features: list[str] = self.feature_metadata.get("input_categorical_features", [])
        self.raw_numeric_features: list[str] = self.feature_metadata.get("input_numeric_features", [])

    def _load_model_bundle(self) -> dict[str, Any]:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model artifact not found at '{MODEL_PATH}'. Train the main model first.")
        return joblib.load(MODEL_PATH)

    def _load_feature_metadata(self) -> dict[str, Any]:
        if not FEATURE_LIST_PATH.exists():
            raise FileNotFoundError(f"Feature metadata not found at '{FEATURE_LIST_PATH}'. Run feature pipeline first.")
        return json.loads(FEATURE_LIST_PATH.read_text(encoding="utf-8"))

    def _load_threshold(self) -> float:
        if THRESHOLD_METRICS_PATH.exists():
            metrics = pd.read_csv(THRESHOLD_METRICS_PATH)
            best = metrics.sort_values(["business_cost", "f1", "threshold"], ascending=[True, False, False]).iloc[0]
            return float(best["threshold"])
        return 0.5

    def _prepare_raw_frame(self, payload: dict[str, Any]) -> pd.DataFrame:
        row = payload.copy()
        if self.target_column not in row:
            row[self.target_column] = None
        return pd.DataFrame([row])

    def _engineer_features(self, payload: dict[str, Any]) -> pd.DataFrame:
        raw_frame = self._prepare_raw_frame(payload)
        feature_result = build_feature_frame(
            dataframe=raw_frame,
            target_column=self.target_column,
            timestamp_column=self.timestamp_column,
            user_id_column=self.user_id_column,
            transaction_id_column=self.transaction_id_column,
        )
        engineered = feature_result.feature_frame.drop(columns=["_parsed_timestamp"], errors="ignore").copy()
        return engineered

    def _build_encoded_vector(self, engineered_frame: pd.DataFrame, payload: dict[str, Any]) -> pd.DataFrame:
        row = engineered_frame.iloc[0].to_dict()
        encoded: dict[str, float] = {}

        for feature_name in self.feature_names:
            if feature_name in row:
                encoded[feature_name] = self._coerce_numeric(row.get(feature_name))
                continue

            matched = False
            for categorical_feature in self.raw_categorical_features:
                prefix = f"{categorical_feature}_"
                if feature_name.startswith(prefix):
                    expected_category = feature_name[len(prefix):]
                    raw_value = row.get(categorical_feature, payload.get(categorical_feature))
                    encoded[feature_name] = 1.0 if str(raw_value) == expected_category else 0.0
                    matched = True
                    break

            if not matched:
                encoded[feature_name] = 0.0

        return pd.DataFrame([encoded], columns=self.feature_names)

    @staticmethod
    def _coerce_numeric(value: Any) -> float:
        if value is None:
            return 0.0
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(numeric):
            return 0.0
        return float(numeric)

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        engineered = self._engineer_features(payload)
        encoded = self._build_encoded_vector(engineered, payload)
        probability = float(self.model.predict_proba(encoded)[0, 1])
        label = int(probability >= self.threshold)
        return {
            "fraud_probability": probability,
            "fraud_label": label,
            "threshold": float(self.threshold),
        }


@lru_cache(maxsize=1)
def get_predictor() -> FraudPredictor:
    """Return a singleton predictor instance for API use."""
    return FraudPredictor()
