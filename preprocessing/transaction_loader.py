"""CSV ingestion and canonical transaction loading."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from preprocessing.transaction_schema import TransactionSchema


def load_transactions(
    path: str | Path,
    schema: TransactionSchema | None = None,
    availability_report_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load a CSV, normalize its columns, and optionally persist availability."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Transaction CSV does not exist: {csv_path}")
    source = pd.read_csv(csv_path)
    canonical = (schema or TransactionSchema.default()).normalize(source)
    if availability_report_path is not None:
        TransactionSchema.write_availability_report(
            canonical, availability_report_path
        )
    return canonical
