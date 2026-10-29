"""Deterministic synthetic banking data for pipeline tests and examples."""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


_TRANSACTION_TYPES = (
    "UPI",
    "IMPS",
    "NEFT",
    "RTGS",
    "CARD_PAYMENT",
    "CARD_TRANSFER",
    "BANK_TRANSFER",
    "WALLET",
    "MERCHANT_PAYMENT",
    "BILL_PAYMENT",
    "CASH_WITHDRAWAL",
)


def generate_synthetic_transactions(
    row_count: int = 120,
    seed: int = 42,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Create reproducible legitimate and suspicious transaction examples."""
    if row_count < 1:
        raise ValueError("row_count must be positive")
    rng = random.Random(seed)
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows: list[dict[str, object]] = []
    balances = {f"account_{index}": 10000.0 for index in range(8)}
    for index in range(row_count):
        suspicious = index % 11 in (0, 1) and index >= 11
        sender = f"account_{6 if suspicious else rng.randrange(6)}"
        receiver = (
            "shared_beneficiary"
            if suspicious or index % 5 == 0
            else f"account_{rng.randrange(6)}"
        )
        amount = (
            1800.0 + (index % 7) * 175.0
            if suspicious
            else round(rng.uniform(5.0, 240.0) + index * 0.15, 2)
        )
        before = balances.setdefault(sender, 10000.0)
        balances[sender] = before - amount
        timestamp = start + timedelta(minutes=index * 7)
        rows.append(
            {
                "transaction_id": f"synthetic_{index:05d}",
                "timestamp": timestamp.isoformat(),
                "sender_account_id": sender,
                "receiver_account_id": receiver,
                "amount": amount,
                "currency": "INR",
                "transaction_type": _TRANSACTION_TYPES[index % len(_TRANSACTION_TYPES)],
                "channel": ("mobile", "web", "pos")[index % 3],
                "merchant_id": f"merchant_{index % 9}",
                "device_id": f"device_{index % 4}",
                "ip_id": f"ip_{index % 5}",
                "location": f"city_{index % 3}",
                "sender_balance_before": before,
                "sender_balance_after": balances[sender],
                "receiver_balance_before": pd.NA,
                "receiver_balance_after": pd.NA,
                "label": int(suspicious),
            }
        )
    transactions = pd.DataFrame(rows)
    if output_path is not None:
        csv_path = Path(output_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        transactions.to_csv(csv_path, index=False)
    return transactions
