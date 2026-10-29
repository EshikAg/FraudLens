"""Adapt existing analyst explanations to a freshly built inference graph."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

from explainability.explanation_service import ExplanationService as GAAPExplanationService
from models.banking_gaap import BankingGAAP, _resolve_path


class ExplanationService:
    """Reuse the established feature and graph explanation implementation."""

    def __init__(self, config: Mapping[str, Any] | str | Path | None = None) -> None:
        if isinstance(config, (str, Path)):
            self.service = GAAPExplanationService(_resolve_path(config))
        else:
            self.service = GAAPExplanationService(config)

    def explain(self, model: BankingGAAP, inference_graph: Any, transaction_id: str) -> dict[str, Any]:
        """Explain the scored node without changing the loaded model state."""
        graph_view = copy.copy(model)
        graph_view.graph = inference_graph
        return self.service.explain(graph_view, transaction_id)

