"""Pydantic schemas for inference API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TransactionRequest(BaseModel):
    """Flexible request schema for a single transaction."""

    model_config = ConfigDict(extra="allow")

    amount: float | None = None
    transaction_time: str | None = None
    timestamp: str | None = None
    datetime: str | None = None
    event_time: str | None = None
    user_id: str | None = None
    customer_id: str | None = None
    account_id: str | None = None
    client_id: str | None = None
    transaction_id: str | None = None
    tx_id: str | None = None
    currency: str | None = None
    merchant: str | None = None

    def to_feature_dict(self) -> dict[str, Any]:
        """Return all explicit and extra fields as a plain dict."""
        payload = self.model_dump(exclude_none=True)
        extras = self.model_extra or {}
        return {**extras, **payload}


class PredictionResponse(BaseModel):
    """Prediction response schema."""

    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    fraud_label: int = Field(..., ge=0, le=1)
    threshold: float = Field(..., ge=0.0, le=1.0)
