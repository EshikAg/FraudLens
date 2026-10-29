"""Persistent metadata registry for immutable GAAP model versions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if value is pd.NA:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


class ModelRegistry:
    """Persist model lineage, checkpoints, promotion, and rollback state."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.directory / "registry.json"
        if self.metadata_path.exists():
            data = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not isinstance(data.get("versions"), list):
                raise ValueError(f"Invalid model registry file: {self.metadata_path}")
            self._versions = data["versions"]
            self._current_version = data.get("current_version")
            self._promotion_history = list(data.get("promotion_history", []))
            self._validate_loaded_state()
        else:
            self._versions: list[dict[str, Any]] = []
            self._current_version: str | None = None
            self._promotion_history: list[str] = []

    @property
    def current_version(self) -> str | None:
        return self._current_version

    @property
    def current(self) -> dict[str, Any] | None:
        return self.get(self._current_version) if self._current_version else None

    def next_version(self) -> str:
        """Return the next monotonic vN name."""
        numbers = [
            int(record["version"][1:])
            for record in self._versions
            if str(record.get("version", "")).startswith("v")
            and str(record["version"])[1:].isdigit()
        ]
        return f"v{max(numbers, default=0) + 1}"

    def register(
        self,
        *,
        version: str,
        checkpoint_path: str | Path,
        training_data_window: Mapping[str, Any],
        drift_event: Mapping[str, Any] | None,
        metrics: Mapping[str, Any],
        feature_schema_version: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Register a checkpoint and its lineage without promoting it."""
        if not version or self.get(version) is not None:
            raise ValueError(f"Model version is empty or already registered: {version!r}")
        checkpoint = Path(checkpoint_path).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Model checkpoint does not exist: {checkpoint}")
        if not feature_schema_version:
            raise ValueError("feature_schema_version must not be empty")
        record = {
            "version": version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "training_data_window": _json_value(training_data_window),
            "drift_event": _json_value(drift_event),
            "metrics": _json_value(metrics),
            "checkpoint_path": str(checkpoint),
            "feature_schema_version": feature_schema_version,
            "status": "candidate",
            "metadata": _json_value(metadata or {}),
        }
        self._versions.append(record)
        self._persist()
        return dict(record)

    def get(self, version: str | None) -> dict[str, Any] | None:
        """Return a copy of the requested version record."""
        if version is None:
            return None
        for record in self._versions:
            if record["version"] == version:
                return dict(record)
        return None

    def list_versions(self) -> list[dict[str, Any]]:
        """Return all registered versions in creation order."""
        return [dict(record) for record in self._versions]

    def promote(self, version: str) -> dict[str, Any]:
        """Set current version only after the caller's promotion checks pass."""
        record = self.get(version)
        if record is None:
            raise KeyError(f"Unknown model version: {version}")
        if not Path(record["checkpoint_path"]).is_file():
            raise FileNotFoundError(
                f"Cannot promote missing checkpoint: {record['checkpoint_path']}"
            )
        if self._current_version is not None:
            self._set_status(self._current_version, "previous")
        self._current_version = version
        self._promotion_history.append(version)
        self._set_status(version, "promoted")
        self._persist()
        return dict(self.get(version) or {})

    def reject(self, version: str, reason: str) -> None:
        """Mark an evaluated candidate as rejected without changing current."""
        if self.get(version) is None:
            raise KeyError(f"Unknown model version: {version}")
        if not reason:
            raise ValueError("A rejection reason must be recorded")
        self._set_status(version, "rejected")
        record = self._record(version)
        record["rejection_reason"] = reason
        self._persist()

    def rollback(self) -> dict[str, Any]:
        """Restore the prior promoted version, preserving all version records."""
        if len(self._promotion_history) < 2:
            raise RuntimeError("No previous promoted model is available for rollback")
        failed = self._promotion_history.pop()
        previous = self._promotion_history[-1]
        previous_record = self.get(previous)
        if previous_record is None or not Path(previous_record["checkpoint_path"]).is_file():
            self._promotion_history.append(failed)
            raise FileNotFoundError(
                f"Previous model checkpoint is unavailable for rollback: {previous}"
            )
        self._set_status(failed, "rolled_back")
        self._set_status(previous, "promoted")
        self._current_version = previous
        self._persist()
        return dict(self.get(previous) or {})

    def checkpoint_path(self, version: str | None = None) -> Path:
        """Return an existing checkpoint for the requested or current version."""
        record = self.get(version or self._current_version)
        if record is None:
            raise RuntimeError("No registered model version is available")
        path = Path(record["checkpoint_path"])
        if not path.is_file():
            raise FileNotFoundError(f"Registered model checkpoint is missing: {path}")
        return path

    def _record(self, version: str) -> dict[str, Any]:
        for record in self._versions:
            if record["version"] == version:
                return record
        raise KeyError(f"Unknown model version: {version}")

    def _set_status(self, version: str, status: str) -> None:
        self._record(version)["status"] = status

    def _validate_loaded_state(self) -> None:
        versions = [record.get("version") for record in self._versions]
        if len(versions) != len(set(versions)):
            raise ValueError("Model registry contains duplicate version names")
        if self._current_version is not None and self._current_version not in versions:
            raise ValueError("Model registry current_version is not registered")
        if any(version not in versions for version in self._promotion_history):
            raise ValueError("Model registry promotion history references unknown versions")

    def _persist(self) -> None:
        payload = {
            "format_version": 1,
            "current_version": self._current_version,
            "promotion_history": self._promotion_history,
            "versions": self._versions,
        }
        temporary = self.metadata_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(_json_value(payload), indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.metadata_path)
