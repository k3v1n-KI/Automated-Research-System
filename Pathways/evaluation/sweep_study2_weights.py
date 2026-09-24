"""Sweep Study 2 ranking weights on the controlled development corpus."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

try:
    from .evaluate_study2 import DEFAULT_CORPUS, metrics, resource_from_json
except ImportError:
    from evaluate_study2 import DEFAULT_CORPUS, metrics, resource_from_json
from pathways_domain import PathwaysStore
from pathways_find import FindSearch


def evaluate_weights(corpus_path: Path, freshness_weights: list[float]) -> dict[str, Any]:
    fixture = json.loads(corpus_path.read_text(encoding="utf-8"))
    now = datetime.fromisoformat(fixture["now"])
    resources = [resource_from_json(record) for record in fixture["resources"]]
    reports = []
    for freshness_weight in freshness_weights:
        fit_weight = 1.0 - freshness_weight
        case_metrics = []
        for case in fixture["cases"]:
            store = PathwaysStore(resources)
            for event in fixture.get("verification_events", []):
                store.verify(
                    event["resource_id"],
                    event["field"],
                    actor="evaluation",
                    created_at=now - timedelta(days=event["days_ago"]),
                )
            search = FindSearch(store)
            results = search.search(case["query"], now=now, limit=100)
            results.sort(
                key=lambda result: (
                    -(fit_weight * result["fit_score"] + freshness_weight * result["trust_score"] - 0.4 * result["flag_penalty"]),
                    result["name"].casefold(),
                    result["id"],
                )
            )
            order = [result["id"] for result in results]
            case_metrics.append(metrics(order, {key: int(value) for key, value in case["relevance"].items()}))
        summary = {
            metric: sum(row[metric] for row in case_metrics) / len(case_metrics)
            for metric in ("precision_at_5", "ndcg_at_5", "mrr", "map")
        }
        reports.append({"fit_weight": fit_weight, "freshness_weight": freshness_weight, "summary": summary})
    best = max(reports, key=lambda report: (report["summary"]["ndcg_at_5"], report["summary"]["map"]))
    return {
        "schema_version": 1,
        "evaluation_type": "in_sample_weight_sensitivity",
        "fixture": str(corpus_path),
        "case_count": len(fixture["cases"]),
        "reports": reports,
        "best_by_ndcg_at_5": best,
        "caveat": "The best weight is selected in-sample and must be validated on held-out or independently adjudicated queries.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_weights(args.corpus, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["best_by_ndcg_at_5"], indent=2))


if __name__ == "__main__":
    main()
