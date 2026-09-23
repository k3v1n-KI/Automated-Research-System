"""Reproducible Study 2 ranking evaluation for the Pathways smoke fixture."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pathways_domain import PathwaysStore, seed_resources
from pathways_find import FindSearch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = Path(__file__).with_name("ranking_cases.json")


def average_precision(retrieved: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for index, resource_id in enumerate(retrieved, 1):
        if resource_id in relevant:
            hits += 1
            total += hits / index
    return total / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for index, resource_id in enumerate(retrieved, 1):
        if resource_id in relevant:
            return 1.0 / index
    return 0.0


def graded_ndcg(retrieved: list[str], relevance: dict[str, int], k: int) -> float:
    def gain(score: int) -> float:
        return (2**score) - 1

    top = retrieved[:k]
    dcg = sum(gain(relevance.get(resource_id, 0)) / math.log2(index + 2) for index, resource_id in enumerate(top))
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(gain(score) / math.log2(index + 2) for index, score in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def fit_only_order(search: FindSearch, query: str, now: datetime, limit: int = 20) -> list[str]:
    results = search.search(query, now=now, limit=limit)
    return [result["id"] for result in sorted(results, key=lambda result: (-result["fit_score"], result["name"].casefold(), result["id"]))]


def evaluate(cases_path: Path = DEFAULT_CASES) -> dict[str, Any]:
    fixture = json.loads(cases_path.read_text(encoding="utf-8"))
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []

    for case in fixture["cases"]:
        store = PathwaysStore(seed_resources())
        for event in case.get("verification_events", []):
            store.verify(
                event["resource_id"],
                event["field"],
                actor="evaluation",
                created_at=now - timedelta(days=event["days_ago"]),
            )
        search = FindSearch(store)
        ranked = search.search(case["query"], now=now, limit=20)
        pathways_ids = [result["id"] for result in ranked]
        fit_ids = fit_only_order(search, case["query"], now)
        relevant = set(case["relevant_ids"])
        graded = {key: int(value) for key, value in case.get("graded_relevance", {}).items()}
        rows.append(
            {
                "id": case["id"],
                "query": case["query"],
                "pathways_top_ids": pathways_ids[:10],
                "fit_only_top_ids": fit_ids[:10],
                "pathways_precision_at_5": sum(resource_id in relevant for resource_id in pathways_ids[:5]) / min(5, len(pathways_ids)) if pathways_ids else 0.0,
                "fit_only_precision_at_5": sum(resource_id in relevant for resource_id in fit_ids[:5]) / min(5, len(fit_ids)) if fit_ids else 0.0,
                "pathways_ndcg_at_5": graded_ndcg(pathways_ids, graded, 5),
                "fit_only_ndcg_at_5": graded_ndcg(fit_ids, graded, 5),
                "pathways_mrr": reciprocal_rank(pathways_ids, relevant),
                "fit_only_mrr": reciprocal_rank(fit_ids, relevant),
                "pathways_map": average_precision(pathways_ids, relevant),
                "fit_only_map": average_precision(fit_ids, relevant),
            }
        )

    metric_names = [
        "precision_at_5",
        "ndcg_at_5",
        "mrr",
        "map",
    ]
    summary = {}
    for metric in metric_names:
        summary[f"pathways_{metric}"] = sum(row[f"pathways_{metric}"] for row in rows) / len(rows)
        summary[f"fit_only_{metric}"] = sum(row[f"fit_only_{metric}"] for row in rows) / len(rows)

    return {
        "schema_version": 1,
        "fixture": str(cases_path.relative_to(ROOT)),
        "generated_at": now.isoformat(),
        "evaluation_type": "development_smoke_fixture",
        "case_count": len(rows),
        "summary": summary,
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate(args.cases)
    payload = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
