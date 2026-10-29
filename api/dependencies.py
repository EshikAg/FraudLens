"""FastAPI dependencies for the shared, application-scoped services."""

from __future__ import annotations

from fastapi import HTTPException, Request, status

from api.services.prediction_service import PredictionService


def get_prediction_service(request: Request) -> PredictionService:
    """Return the one service initialized for the application lifespan."""
    service = getattr(request.app.state, "prediction_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Fraud scoring model is not configured or available",
        )
    return service
