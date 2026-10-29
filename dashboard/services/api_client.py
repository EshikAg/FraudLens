"""Small HTTP client for the existing FraudLens API."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd
import requests


class DashboardAPIError(RuntimeError):
    """The configured scoring API could not provide a usable response."""


class FraudLensAPIClient:
    """Read scores and monitoring data from the running API."""

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 5.0,
        prediction_timeout_seconds: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.prediction_timeout_seconds = float(prediction_timeout_seconds)
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("API base URL must use http:// or https://")
        if self.timeout_seconds <= 0 or self.prediction_timeout_seconds <= 0:
            raise ValueError("API timeouts must be positive")

    def health(self) -> dict[str, Any]:
        return self._get("/health")

    def model(self) -> dict[str, Any]:
        return self._get("/model")

    def drift(self) -> dict[str, Any]:
        return self._get("/drift")

    def predict(self, transaction: Mapping[str, Any]) -> dict[str, Any]:
        """Submit one transaction without sending an offline label."""
        payload = {
            key: self._json_value(value)
            for key, value in transaction.items()
            if key != "label"
        }
        try:
            response = requests.post(
                f"{self.base_url}/predict",
                json=payload,
                timeout=self.prediction_timeout_seconds,
            )
        except requests.RequestException as exc:
            raise DashboardAPIError(f"Prediction API is unavailable: {exc}") from exc
        if not response.ok:
            raise DashboardAPIError(
                f"Prediction API returned HTTP {response.status_code}; "
                "check transaction fields and model availability."
            )
        try:
            result = response.json()
        except requests.JSONDecodeError as exc:
            raise DashboardAPIError("Prediction API returned invalid JSON") from exc
        if not isinstance(result, dict):
            raise DashboardAPIError("Prediction API response must be a JSON object")
        return result

    def _get(self, path: str) -> dict[str, Any]:
        try:
            response = requests.get(
                f"{self.base_url}{path}", timeout=self.timeout_seconds
            )
        except requests.RequestException as exc:
            raise DashboardAPIError(f"FraudLens API is unavailable: {exc}") from exc
        if not response.ok:
            raise DashboardAPIError(
                f"FraudLens API returned HTTP {response.status_code}."
            )
        try:
            result = response.json()
        except requests.JSONDecodeError as exc:
            raise DashboardAPIError(f"{path} returned invalid JSON") from exc
        if not isinstance(result, dict):
            raise DashboardAPIError(f"{path} response must be a JSON object")
        return result

    @staticmethod
    def _json_value(value: Any) -> Any:
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        if hasattr(value, "isoformat"):
            return value.isoformat()
        if hasattr(value, "item"):
            return value.item()
        return value
