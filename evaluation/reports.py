"""Persist machine-readable research results and plots from observed outputs."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is pd.NA:
        return None
    return value


def generate_reports(
    results: Mapping[str, Any],
    output_dir: str | Path,
    *,
    metadata: Mapping[str, Any],
    limitations: list[str],
) -> dict[str, str]:
    """Write timestamped JSON/CSV/Markdown and figures for available evidence.

    Missing inputs are recorded as unavailable; no metrics are synthesized.
    """
    if not limitations:
        raise ValueError("Limitations must be explicitly recorded")
    root = Path(output_dir)
    figures_dir, tables_dir, results_dir = (
        root / "figures",
        root / "tables",
        root / "results",
    )
    for directory in (figures_dir, tables_dir, results_dir):
        directory.mkdir(parents=True, exist_ok=True)

    created_at = datetime.now(timezone.utc).isoformat()
    result_snapshot = _json_safe(dict(results))
    for key in ("models", "variants"):
        entries = result_snapshot.get(key)
        if isinstance(entries, list):
            for entry in entries:
                if isinstance(entry, dict):
                    entry["result_timestamp"] = created_at
    payload = _json_safe(
        {
            "created_at": created_at,
            "metadata": dict(metadata),
            "results": result_snapshot,
            "limitations": list(limitations),
        }
    )
    result_path = results_dir / "evaluation_results.json"
    result_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    records = _result_records(results)
    for record in records:
        record["result_timestamp"] = created_at
    csv_path = tables_dir / "model_metrics.csv"
    pd.DataFrame(records).to_csv(csv_path, index=False)
    markdown_path = root / "final_results.md"
    markdown_path.write_text(
        _markdown_report(payload, records),
        encoding="utf-8",
    )
    figure_paths = _generate_figures(results, records, figures_dir)
    payload["figures"] = figure_paths
    result_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "json": str(result_path),
        "csv": str(csv_path),
        "markdown": str(markdown_path),
        **{f"figure_{index}": path for index, path in enumerate(figure_paths)},
    }


def _result_records(results: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates = results.get("models", results.get("variants", []))
    if isinstance(candidates, Mapping):
        candidates = [
            {"model": name, **(value if isinstance(value, dict) else {})}
            for name, value in candidates.items()
        ]
    records: list[dict[str, Any]] = []
    if not isinstance(candidates, list):
        return records
    for result in candidates:
        if not isinstance(result, Mapping):
            continue
        metrics = result.get("metrics", {})
        row = {
            "model": result.get("model", "unknown"),
            "model_version": result.get("model_version"),
            "seed": result.get("seed", results.get("seed")),
        }
        if isinstance(metrics, Mapping):
            row.update(metrics)
        if isinstance(result.get("latency"), Mapping):
            row.update(
                {
                    f"latency_{key}": value
                    for key, value in result["latency"].items()
                }
            )
        records.append(_json_safe(row))
    return records


def _markdown_report(payload: Mapping[str, Any], records: list[dict[str, Any]]) -> str:
    metadata = payload["metadata"]
    columns = (
        "model",
        "model_version",
        "average_precision",
        "recall",
        "precision",
        "f1",
        "roc_auc",
        "false_positive_rate",
    )
    lines = [
        "# Adaptive-GAAP evaluation results",
        "",
        f"- Generated: {payload['created_at']}",
        f"- Dataset: {metadata.get('dataset', 'not recorded')}",
        f"- Seed: {metadata.get('seed', 'not recorded')}",
        f"- Experiment: {metadata.get('experiment', 'not recorded')}",
        "",
        "## Experiment configuration",
        "",
        "```json",
        json.dumps(metadata.get("experiment_config", {}), indent=2, sort_keys=True),
        "```",
        "",
        "## Model metrics",
        "",
    ]
    if records:
        lines.extend(
            [
                "| " + " | ".join(columns) + " |",
                "| " + " | ".join("---" for _ in columns) + " |",
            ]
        )
        for row in records:
            lines.append(
                "| "
                + " | ".join(
                    "unavailable" if row.get(key) is None else str(row.get(key))
                    for key in columns
                )
                + " |"
            )
    else:
        lines.append("No model metric records were supplied; no comparison is available.")

    lines.extend(["", "## Observations", ""])
    if records:
        valid = [
            row for row in records if isinstance(row.get("average_precision"), (int, float))
        ]
        if len(valid) >= 2:
            best = max(valid, key=lambda row: row["average_precision"])
            lines.append(
                f"Highest observed PR-AUC: {best['model']} ({best['average_precision']:.4f}). "
                "This is descriptive and does not establish statistical significance."
            )
        else:
            lines.append("Insufficient comparable PR-AUC results for a ranking.")
        gaap = next((row for row in records if row.get("model") == "GAAP"), None)
        adaptive = next(
            (row for row in records if row.get("model") == "Adaptive-GAAP"), None
        )
        if gaap and adaptive:
            differences = []
            for metric in ("average_precision", "recall", "precision", "f1"):
                left, right = gaap.get(metric), adaptive.get(metric)
                if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                    differences.append(f"{metric} {right - left:+.4f}")
            if differences:
                lines.append(
                    "Adaptive-GAAP minus fixed GAAP on the same post-drift validation "
                    "transactions: " + ", ".join(differences) + "."
                )
    else:
        lines.append("No quantitative model comparison was provided.")
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in payload["limitations"])
    lines.append("")
    return "\n".join(lines)


def _save_bar(records: list[dict[str, Any]], metric: str, title: str, path: Path) -> bool:
    selected = [row for row in records if isinstance(row.get(metric), (int, float))]
    if not selected:
        return False
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar([str(row["model"]) for row in selected], [row[metric] for row in selected])
    axis.set_title(title)
    axis.set_ylabel(metric.replace("_", " ").title())
    axis.tick_params(axis="x", labelrotation=25)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return True


def _generate_figures(
    results: Mapping[str, Any], records: list[dict[str, Any]], directory: Path
) -> list[str]:
    paths: list[str] = []
    for metric, name, title in (
        ("average_precision", "pr_auc_comparison.png", "PR-AUC comparison"),
        ("recall", "recall_comparison.png", "Recall comparison"),
        ("f1", "f1_comparison.png", "F1 comparison"),
        ("latency_mean_latency_ms", "prediction_latency.png", "Prediction latency"),
    ):
        destination = directory / name
        if _save_bar(records, metric, title, destination):
            paths.append(str(destination))

    streaming = results.get("streaming_windows", [])
    if isinstance(streaming, list) and streaming:
        windows = [
            item for item in streaming
            if isinstance(item, Mapping) and isinstance(item.get("metrics"), Mapping)
        ]
        for metric, name, title in (
            ("average_precision", "fraud_performance_over_time.png", "Fraud performance over time"),
            ("recall", "recall_over_time.png", "Recall over time"),
        ):
            grouped: dict[str, list[tuple[int, float]]] = {}
            for item in windows:
                value = item["metrics"].get(metric)
                if value is not None:
                    grouped.setdefault(
                        str(item.get("model", "model")), []
                    ).append((int(item.get("window_index", len(grouped))), float(value)))
            if grouped:
                figure, axis = plt.subplots(figsize=(8, 4.5))
                for model, points in grouped.items():
                    axis.plot(
                        [point[0] for point in points],
                        [point[1] for point in points],
                        marker="o",
                        label=model,
                    )
                axis.set_title(title)
                axis.set_xlabel("Chronological window")
                axis.set_ylabel(metric.replace("_", " ").title())
                axis.set_ylim(0, 1)
                if len(grouped) > 1:
                    axis.legend()
                figure.tight_layout()
                destination = directory / name
                figure.savefig(destination, dpi=140)
                plt.close(figure)
                paths.append(str(destination))

    drift = results.get("drift_experiment", {})
    drift_points = drift.get("drift_scores", []) if isinstance(drift, Mapping) else []
    if drift_points:
        values = [
            float(item["score"])
            for item in drift_points
            if isinstance(item, Mapping) and item.get("score") is not None
        ]
        if not values and all(isinstance(item, (int, float)) for item in drift_points):
            values = [float(item) for item in drift_points]
    else:
        values = []
    if values:
        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.plot(range(len(values)), values, marker="o")
        axis.set_title("Drift score over time")
        axis.set_xlabel("Chronological window")
        axis.set_ylabel("Drift score")
        figure.tight_layout()
        destination = directory / "drift_score_over_time.png"
        figure.savefig(destination, dpi=140)
        plt.close(figure)
        paths.append(str(destination))

    before_after = drift.get("period_metrics", {}) if isinstance(drift, Mapping) else {}
    if len(before_after) >= 2:
        _plot_period_metrics(before_after, directory / "performance_before_after_drift.png")
        paths.append(str(directory / "performance_before_after_drift.png"))

    recovery = results.get("adaptation_comparison", [])
    if isinstance(recovery, list) and recovery:
        recovery_records = [
            {
                "model": item.get("model", "unknown"),
                **(
                    item.get("metrics", {})
                    if isinstance(item.get("metrics"), Mapping)
                    else {}
                ),
            }
            for item in recovery
        ]
        destination = directory / "adaptive_recovery_comparison.png"
        if _save_bar(
            recovery_records,
            "average_precision",
            "Post-drift GAAP and Adaptive-GAAP comparison",
            destination,
        ):
            paths.append(str(destination))

    ablation = results.get("ablation", {})
    variants = ablation.get("variants", []) if isinstance(ablation, Mapping) else []
    if isinstance(variants, list) and variants:
        destination = directory / "ablation_comparison.png"
        if _save_bar(
            variants,
            "average_precision",
            "Ablation comparison",
            destination,
        ):
            paths.append(str(destination))

    distributions = results.get("fraud_probability_distributions", {})
    if isinstance(distributions, Mapping) and distributions:
        figure, axis = plt.subplots(figsize=(8, 4.5))
        plotted = False
        for label, values in distributions.items():
            values = list(values)
            if values:
                axis.hist(values, bins=20, alpha=0.45, label=str(label))
                plotted = True
        if plotted:
            axis.set_title("Fraud probability distribution")
            axis.set_xlabel("Fraud probability")
            axis.set_ylabel("Transactions")
            axis.legend()
            figure.tight_layout()
            destination = directory / "fraud_probability_distribution.png"
            figure.savefig(destination, dpi=140)
            plt.close(figure)
            paths.append(str(destination))
        else:
            plt.close(figure)

    graph = results.get("example_graph")
    if isinstance(graph, Mapping) and graph.get("nodes"):
        destination = directory / "example_transaction_graph.png"
        if _plot_graph(graph, destination):
            paths.append(str(destination))
    return paths


def _plot_period_metrics(periods: Mapping[str, Any], destination: Path) -> None:
    names = list(periods)
    metric_names = ("average_precision", "recall", "precision", "f1")
    x = np.arange(len(names))
    width = 0.18
    figure, axis = plt.subplots(figsize=(9, 4.5))
    plotted = False
    for index, metric in enumerate(metric_names):
        values = [
            periods[name].get(metric) if isinstance(periods[name], Mapping) else None
            for name in names
        ]
        if any(value is not None for value in values):
            axis.bar(
                x + (index - 1.5) * width,
                [0 if value is None else value for value in values],
                width,
                label=metric,
            )
            plotted = True
    if plotted:
        axis.set_xticks(x, names)
        axis.set_ylim(0, 1)
        axis.set_title("Model performance before and after drift")
        axis.legend()
        figure.tight_layout()
        figure.savefig(destination, dpi=140)
    plt.close(figure)


def _plot_graph(graph: Mapping[str, Any], destination: Path) -> bool:
    nodes = graph["nodes"]
    edges = graph.get("edges", [])
    if len(nodes) > 100:
        raise ValueError("Example graph visualizations are limited to 100 nodes")
    snapshot = pd.to_datetime(graph["snapshot_timestamp"], utc=True)
    timestamps = {
        str(node["transaction_id"]): pd.to_datetime(node["timestamp"], utc=True)
        for node in nodes
    }
    visible = {
        key for key, timestamp in timestamps.items() if timestamp <= snapshot
    }
    if len(visible) != len(timestamps):
        nodes = [node for node in nodes if str(node["transaction_id"]) in visible]
    filtered_edges = []
    for edge in edges:
        source, target = str(edge["source"]), str(edge["destination"])
        edge_time = pd.to_datetime(edge["timestamp"], utc=True)
        if source in visible and target in visible and edge_time <= snapshot:
            filtered_edges.append(edge)
    identifiers = [str(node["transaction_id"]) for node in nodes]
    positions = {identifier: (index, 0.0) for index, identifier in enumerate(identifiers)}
    figure, axis = plt.subplots(figsize=(max(6, len(nodes) * 0.6), 4))
    for edge in filtered_edges:
        source, target = str(edge["source"]), str(edge["destination"])
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        axis.plot([x1, x2], [y1, y2], color="gray", alpha=0.6, zorder=1)
    fraud = {
        str(node["transaction_id"]): bool(node.get("label") == 1) for node in nodes
    }
    axis.scatter(
        [positions[key][0] for key in identifiers],
        [0.0] * len(identifiers),
        c=["crimson" if fraud[key] else "steelblue" for key in identifiers],
        zorder=2,
    )
    axis.set_xticks(range(len(identifiers)), identifiers, rotation=45, ha="right")
    axis.set_yticks([])
    axis.set_title("Historical transaction graph (as of snapshot)")
    figure.tight_layout()
    figure.savefig(destination, dpi=140)
    plt.close(figure)
    return True
