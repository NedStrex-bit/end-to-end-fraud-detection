"""FastAPI application for fraud prediction."""

from __future__ import annotations

from fastapi import FastAPI

from src.inference.predictor import get_predictor
from src.inference.schemas import PredictionResponse, TransactionRequest


app = FastAPI(title="Fraud Detection System API", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: TransactionRequest) -> PredictionResponse:
    """Score a single transaction and return probability plus business label."""
    predictor = get_predictor()
    prediction = predictor.predict(request.to_feature_dict())
    return PredictionResponse(**prediction)
