"""Replay-based incremental adaptation around the existing GAAP model."""

from typing import Any

__all__ = ["AdaptationManager", "ReplayBuffer"]


def __getattr__(name: str) -> Any:
    if name == "AdaptationManager":
        from adaptation.adaptation_manager import AdaptationManager

        return AdaptationManager
    if name == "ReplayBuffer":
        from adaptation.replay_buffer import ReplayBuffer

        return ReplayBuffer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
