from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

from .models import Query
from .pipeline import DocumentRankPipeline


def evaluate(pipeline: DocumentRankPipeline, queries: Iterable[Query], cutoffs: tuple[int, ...] = (1, 5, 10)) -> dict[str, object]:
    per_query = []
    for query in queries:
        max_cutoff = max(cutoffs)
        ranked_ids = [item.chunk.chunk_id for item in pipeline.search(query.text, max_cutoff)]
        metrics = _query_metrics(ranked_ids, query.relevant_chunk_ids, cutoffs)
        per_query.append({"query_id": query.query_id, "category": query.category, **metrics})
    return _summarize(per_query, cutoffs)


def _query_metrics(ranked_ids: list[str], relevant_ids: frozenset[str], cutoffs: tuple[int, ...]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for cutoff in cutoffs:
        retrieved = ranked_ids[:cutoff]
        metrics[f"recall@{cutoff}"] = len(set(retrieved) & relevant_ids) / len(relevant_ids) if relevant_ids else 0.0
        metrics[f"ndcg@{cutoff}"] = _ndcg(retrieved, relevant_ids)
    reciprocal_rank = 0.0
    for position, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in relevant_ids:
            reciprocal_rank = 1.0 / position
            break
    metrics["mrr"] = reciprocal_rank
    return metrics


def _ndcg(ranked_ids: list[str], relevant_ids: frozenset[str]) -> float:
    dcg = sum(1.0 / math.log2(position + 1) for position, chunk_id in enumerate(ranked_ids, start=1) if chunk_id in relevant_ids)
    ideal_count = min(len(relevant_ids), len(ranked_ids))
    ideal = sum(1.0 / math.log2(position + 1) for position in range(1, ideal_count + 1))
    return dcg / ideal if ideal else 0.0


def _summarize(per_query: list[dict[str, object]], cutoffs: tuple[int, ...]) -> dict[str, object]:
    metric_names = [*(f"recall@{cutoff}" for cutoff in cutoffs), *(f"ndcg@{cutoff}" for cutoff in cutoffs), "mrr"]
    overall = _mean_metrics(per_query, metric_names)
    groups: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for result in per_query:
        groups[str(result["category"])].append(result)
    return {
        "query_count": len(per_query),
        "overall": overall,
        "by_category": {category: _mean_metrics(results, metric_names) for category, results in sorted(groups.items())},
        "per_query": per_query,
    }


def _mean_metrics(results: list[dict[str, object]], metric_names: list[str]) -> dict[str, float]:
    if not results:
        return {metric: 0.0 for metric in metric_names}
    return {
        metric: sum(float(result[metric]) for result in results) / len(results)
        for metric in metric_names
    }
