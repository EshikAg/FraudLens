"""Validated request and response models for online transaction scoring."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TransactionRequest(BaseModel):
    """Canonical online transaction input; extra source fields are accepted."""

    model_config = ConfigDict(extra="allow", allow_inf_nan=False)

    transaction_id: str = Field(min_length=1, max_length=256)
    timestamp: datetime
    sender_account_id: str = Field(min_length=1, max_length=256)
    receiver_account_id: str = Field(min_length=1, max_length=256)
    amount: float = Field(gt=0)
    currency: str = Field(min_length=3, max_length=8)
    transaction_type: str = Field(min_length=1, max_length=64)
    channel: str = Field(min_length=1, max_length=64)
    merchant_id: str | None = Field(default=None, max_length=256)
    device_id: str | None = Field(default=None, max_length=256)
    ip_id: str | None = Field(default=None, max_length=128)
    location: str | None = Field(default=None, max_length=256)
    sender_balance_before: float | None = Field(default=None, ge=0)
    sender_balance_after: float | None = Field(default=None, ge=0)
    receiver_balance_before: float | None = Field(default=None, ge=0)
    receiver_balance_after: float | None = Field(default=None, ge=0)

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must include a timezone")
        return value

    @field_validator(
        "transaction_id",
        "sender_account_id",
        "receiver_account_id",
        "currency",
        "transaction_type",
        "channel",
    )
    @classmethod
    def required_strings_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be blank")
        return value.strip()


class PredictionResponse(BaseModel):
    """Analyst-facing online score and supporting evidence."""

    transaction_id: str
    fraud_probability: float = Field(ge=0, le=1)
    risk_level: str
    decision: str
    drift_detected: bool
    model_version: str
    reasons: list[dict[str, Any]]
    related_transactions: list[dict[str, Any]]
    timestamp: datetime


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None


class LabelUpdateRequest(BaseModel):
    """Delayed binary label used only by online concept-drift monitoring."""

    label: int = Field(ge=0, le=1)
    timestamp: datetime | None = None

    @field_validator("timestamp")
    @classmethod
    def optional_timestamp_must_be_timezone_aware(
        cls, value: datetime | None
    ) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("timestamp must include a timezone")
        return value
