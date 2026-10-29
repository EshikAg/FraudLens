"""Configurable tabular baselines and adapters for externally scored graph models."""

from __future__ import annotations

import importlib
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from evaluation.latency import measure_prediction_latency
from evaluation.metrics import evaluate_binary_predictions


def _estimator_factories(seed: int) -> dict[str, Callable[[], Any]]:
    factories: dict[str, Callable[[], Any]] = {
        "logistic_regression": lambda: LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=seed
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,
        ),
    }
    try:
        xgboost = importlib.import_module("xgboost")
    except ImportError:
        pass
    else:
        factories["xgboost"] = lambda: xgboost.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
        )
    return factories


def run_benchmark(
    training: pd.DataFrame,
    test: pd.DataFrame,
    *,
    features: Sequence[str],
    model_names: Sequence[str] = ("logistic_regression", "random_forest"),
    label_column: str = "label",
    threshold: float = 0.5,
    seed: int = 42,
    precision_recall_k: Sequence[int] = (10, 50, 100),
    external_predictions: Mapping[str, pd.DataFrame] | None = None,
    latency_repeats: int = 10,
) -> dict[str, Any]:
    """Train requested tabular baselines and include provided graph-model scores.

    External predictions must contain transaction_id and fraud_probability;
    labels are aligned by transaction ID against the supplied test set.
    """
    if not features or label_column not in training or label_column not in test:
        raise ValueError("features and labeled training/test frames are required")
    missing = set(features).difference(training.columns) | set(features).difference(test.columns)
    if missing:
        raise ValueError(f"Benchmark features are missing from data: {sorted(missing)}")
    if training.empty or test.empty:
        raise ValueError("Training and test frames must be non-empty")
    if training[label_column].nunique() != 2:
        raise ValueError("Training set must contain both label classes")
    if "transaction_id" not in test:
        raise ValueError("Test frame must contain transaction_id for model alignment")

    numeric = [column for column in features if pd.api.types.is_numeric_dtype(training[column])]
    categorical = [column for column in features if column not in numeric]
    transforms: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transforms.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transforms.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )
    preprocessor = ColumnTransformer(transforms)
    factories = _estimator_factories(seed)
    unknown_models = set(model_names).difference(factories)
    if unknown_models:
        if "xgboost" in unknown_models:
            raise ImportError("xgboost was requested but is not installed")
        raise ValueError(f"Unsupported benchmark models: {sorted(unknown_models)}")

    results: list[dict[str, Any]] = []
    for name in model_names:
        estimator = Pipeline([("features", preprocessor), ("model", factories[name]())])
        started = time.perf_counter()
        estimator.fit(training[list(features)], training[label_column].astype(int))
        training_seconds = time.perf_counter() - started
        class_positions = np.flatnonzero(estimator.classes_ == 1)
        if len(class_positions) != 1:
            raise RuntimeError(f"{name} estimator did not expose the fraud class")
        probabilities = estimator.predict_proba(test[list(features)])[:, class_positions[0]]
        metrics = evaluate_binary_predictions(
            test[label_column].astype(int),
            probabilities,
            threshold=threshold,
            precision_recall_k=precision_recall_k,
        )
        latency = measure_prediction_latency(
            lambda: estimator.predict_proba(test[list(features)]),
            repeats=latency_repeats,
            warmup=1,
        )
        results.append(
            {
                "model": name,
                "model_version": f"{name}-seed-{seed}",
                "seed": int(seed),
                "training_seconds": training_seconds,
                "metrics": metrics,
                "latency": latency,
                "prediction_records": _prediction_records(
                    test, test[label_column], probabilities, f"{name}-seed-{seed}"
                ),
            }
        )

    for name, prediction_frame in (external_predictions or {}).items():
        required = {"transaction_id", "fraud_probability"}
        if not required.issubset(prediction_frame.columns):
            raise ValueError(
                f"External predictions for {name!r} must contain {sorted(required)}"
            )
        if prediction_frame["transaction_id"].duplicated().any():
            raise ValueError(f"External predictions for {name!r} have duplicate IDs")
        test_columns = ["transaction_id", label_column]
        if "timestamp" in test:
            test_columns.append("timestamp")
        aligned = test[test_columns].merge(
            prediction_frame[["transaction_id", "fraud_probability", "model_version"]]
            if "model_version" in prediction_frame
            else prediction_frame[["transaction_id", "fraud_probability"]],
            on="transaction_id",
            how="inner",
            validate="one_to_one",
        )
        if len(aligned) != len(test):
            raise ValueError(f"External predictions for {name!r} do not cover test IDs")
        results.append(
            {
                "model": name,
                "model_version": (
                    str(aligned["model_version"].iloc[0])
                    if "model_version" in aligned and aligned["model_version"].nunique() == 1
                    else name
                ),
                "seed": int(seed),
                "training_seconds": None,
                "metrics": evaluate_binary_predictions(
                    aligned[label_column],
                    aligned["fraud_probability"],
                    threshold=threshold,
                    precision_recall_k=precision_recall_k,
                ),
                "latency": None,
                "prediction_records": _prediction_records(
                    aligned,
                    aligned[label_column],
                    aligned["fraud_probability"].to_numpy(),
                    name,
                    model_version_column=(
                        "model_version" if "model_version" in aligned else None
                    ),
                ),
            }
        )
    return {"seed": int(seed), "models": results}


def _prediction_records(
    test: pd.DataFrame,
    labels: pd.Series,
    probabilities: np.ndarray,
    model_version: str,
    *,
    model_version_column: str | None = None,
) -> list[dict[str, Any]]:
    """Keep scored IDs and timestamps attached to the actual evaluation outputs."""
    records = []
    for position, (_, row) in enumerate(test.iterrows()):
        version = (
            str(row[model_version_column])
            if model_version_column and pd.notna(row[model_version_column])
            else model_version
        )
        record = {
            "transaction_id": str(row["transaction_id"]),
            "true_label": int(labels.iloc[position]),
            "fraud_probability": float(probabilities[position]),
            "model_version": version,
        }
        if "timestamp" in row.index:
            record["timestamp"] = pd.to_datetime(row["timestamp"], utc=True).isoformat()
        records.append(record)
    return records
