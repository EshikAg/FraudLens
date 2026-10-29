"""Model adapters for Adaptive-GAAP."""

from models.banking_gaap import BankingGAAP
from models.gaap_adapter import GAAPAdapter, TemporalTransactionDataModule

__all__ = [
    "AdaptiveGAAP",
    "BankingGAAP",
    "GAAPAdapter",
    "TemporalTransactionDataModule",
]


def __getattr__(name: str):
    if name == "AdaptiveGAAP":
        from models.adaptive_gaap import AdaptiveGAAP

        return AdaptiveGAAP
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
