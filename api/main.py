"""FastAPI endpoints for online Adaptive-GAAP transaction scoring."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Mapping

from fastapi import Depends, FastAPI, HTTPException, status

from api.dependencies import get_prediction_service
from api.schemas import (
    HealthResponse,
    LabelUpdateRequest,
    PredictionResponse,
    TransactionRequest,
)
from api.services.prediction_service import PredictionExecutionError, PredictionService
from api.services.transaction_service import (
    InvalidTransactionError,
    TransactionConflictError,
)
from models.banking_gaap import PROJECT_ROOT


LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "api_config.yaml"


def _load_api_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return PredictionService.load_config(path)


def create_app(
    config: Mapping[str, Any] | None = None,
    *,
    prediction_service: PredictionService | None = None,
) -> FastAPI:
    """Create an app; the saved model is loaded once during lifespan startup."""
    settings = dict(config) if config is not None else _load_api_config()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        service = prediction_service
        if service is None and settings.get("model_directory"):
            service = PredictionService.from_config(settings)
        application.state.prediction_service = service
        if service is None:
            LOGGER.warning(
                "fraud_scoring_model_not_configured",
                extra={"event": "fraud_scoring_model_not_configured"},
            )
        yield

    application = FastAPI(
        title="Adaptive-GAAP Fraud Scoring API",
        version="1.0.0",
        description="Online transaction scoring backed by the existing GAAP model.",
        lifespan=lifespan,
    )
    application.state.prediction_service = prediction_service

    @application.get("/health", response_model=HealthResponse)
    def health() -> dict[str, Any]:
        service = getattr(application.state, "prediction_service", None)
        if service is None:
            return {"status": "degraded", "model_loaded": False, "model_version": None}
        information = service.model_information()
        return {
            "status": "healthy",
            "model_loaded": True,
            "model_version": information["model_version"],
        }

    @application.post("/predict", response_model=PredictionResponse)
    def predict(
        transaction: TransactionRequest,
        service: PredictionService = Depends(get_prediction_service),
    ) -> dict[str, Any]:
        try:
            return service.predict(transaction.model_dump(mode="python"))
        except TransactionConflictError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=str(exc)
            ) from exc
        except InvalidTransactionError as exc:
            raise HTTPException(
                status_code=422, detail=str(exc)
            ) from exc
        except PredictionExecutionError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
            ) from exc

    @application.post("/labels/{transaction_id}")
    def update_label(
        transaction_id: str,
        label_update: LabelUpdateRequest,
        service: PredictionService = Depends(get_prediction_service),
    ) -> dict[str, Any]:
        return service.update_label(
            transaction_id,
            label_update.label,
            timestamp=label_update.timestamp,
        )

    @application.get("/model")
    def model_information() -> dict[str, Any]:
        service = getattr(application.state, "prediction_service", None)
        if service is None:
            return {"model_loaded": False, "model_version": None}
        return service.model_information()

    @application.get("/drift")
    def drift_status(
        service: PredictionService = Depends(get_prediction_service),
    ) -> dict[str, Any]:
        return service.drift_status()

    @application.get("/metrics")
    def metrics(
        service: PredictionService = Depends(get_prediction_service),
    ) -> dict[str, Any]:
        return service.metrics()

    return application


app = create_app()
