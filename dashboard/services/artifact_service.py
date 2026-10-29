"""Read-only access to model registry and experiment reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ArtifactService:
    """Load optional local metrics without inventing unavailable values."""

    def __init__(self, paths: Mapping[str, str | Path]) -> None:
        self.paths = {
            name: self._resolve(path) for name, path in paths.items()
        }

    @staticmethod
    def _resolve(path: str | Path) -> Path:
        resolved = Path(path).expanduser()
        return resolved if resolved.is_absolute() else (PROJECT_ROOT / resolved).resolve()

    def read_json(self, name: str) -> dict[str, Any] | None:
        """Return a JSON object, None when an optional file does not exist."""
        path = self.paths.get(name)
        if path is None or not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Unable to read dashboard artifact {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Dashboard artifact must contain a JSON object: {path}")
        return value

    def model_versions(self) -> tuple[list[dict[str, Any]], str | None]:
        registry = self.read_json("model_registry")
        if registry is None:
            return [], None
        versions = registry.get("versions")
        if not isinstance(versions, list) or not all(
            isinstance(record, dict) for record in versions
        ):
            raise ValueError("Model registry artifact must contain a versions list")
        return versions, registry.get("current_version")
