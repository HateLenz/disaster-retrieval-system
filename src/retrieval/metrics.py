from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np


def compute_recall_at_ks(
    query_ids: Iterable[str],
    retrieved_ids: np.ndarray,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    query_ids = np.asarray(list(query_ids))
    metrics: dict[str, float] = {}
    for k in ks:
        hits = [
            int(query_id in row[: min(k, row.shape[0])])
            for query_id, row in zip(query_ids, retrieved_ids, strict=True)
        ]
        metrics[f"recall@{k}"] = float(np.mean(hits)) if hits else 0.0
    return metrics


def compute_mrr(
    query_ids: Iterable[str],
    retrieved_ids: np.ndarray,
) -> float:
    reciprocal_ranks: list[float] = []
    for query_id, row in zip(query_ids, retrieved_ids, strict=True):
        matches = np.where(row == query_id)[0]
        reciprocal_ranks.append(0.0 if matches.size == 0 else 1.0 / float(matches[0] + 1))
    return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0


def compute_metrics(
    query_ids: Iterable[str],
    retrieved_ids: np.ndarray,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    metrics = compute_recall_at_ks(query_ids=query_ids, retrieved_ids=retrieved_ids, ks=ks)
    metrics["mrr"] = compute_mrr(query_ids=query_ids, retrieved_ids=retrieved_ids)
    return metrics


def compute_grouped_metrics(
    query_ids: Iterable[str],
    retrieved_ids: np.ndarray,
    groups: Iterable[str],
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, dict[str, float]]:
    grouped_rows: dict[str, list[tuple[str, np.ndarray]]] = defaultdict(list)
    for query_id, row, group in zip(query_ids, retrieved_ids, groups, strict=True):
        grouped_rows[str(group)].append((str(query_id), row))

    grouped_metrics: dict[str, dict[str, float]] = {}
    for group in sorted(grouped_rows):
        items = grouped_rows[group]
        group_query_ids = [query_id for query_id, _ in items]
        group_retrieved = np.stack([row for _, row in items], axis=0)
        grouped_metrics[group] = compute_metrics(group_query_ids, group_retrieved, ks=ks)
    return grouped_metrics


def compute_macro_metrics(
    grouped_metrics: dict[str, dict[str, float]],
) -> dict[str, float]:
    if not grouped_metrics:
        return {}

    metric_names = sorted({metric_name for metrics in grouped_metrics.values() for metric_name in metrics})
    return {
        metric_name: float(np.mean([metrics[metric_name] for metrics in grouped_metrics.values()]))
        for metric_name in metric_names
    }


__all__ = [
    "compute_grouped_metrics",
    "compute_macro_metrics",
    "compute_metrics",
    "compute_mrr",
    "compute_recall_at_ks",
]
