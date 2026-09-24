"""Evaluate the expanded Study 2 ranking corpus against BM25 and fit-only baselines."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pathways_domain import PathwaysStore, Resource
from pathways_find import FindSearch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = Path(__file__).with_name("study2_corpus.json")
STOP_WORDS = {"a", "an", "and", "for", "in", "me", "near", "of", "the", "to", "with"}


def tokens(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", text.casefold()) if token not in STOP_WORDS and len(token) > 1]


def graded_ndcg(retrieved: list[str], relevance: dict[str, int], k: int) -> float:
    def gain(score: int) -> float:
        return (2**score) - 1

    dcg = sum(gain(relevance.get(resource_id, 0)) / math.log2(index + 2) for index, resource_id in enumerate(retrieved[:k]))
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(gain(score) / math.log2(index + 2) for index, score in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


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


class BM25:
    def __init__(self, resources: list[Resource], k1: float = 1.5, b: float = 0.75) -> None:
        self.resources = resources
        self.k1 = k1
        self.b = b
        self.documents = [tokens(self.document(resource)) for resource in resources]
        self.lengths = [len(document) for document in self.documents]
        self.average_length = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        document_frequency = Counter(term for document in self.documents for term in set(document))
        self.idf = {
            term: math.log(1 + (len(self.documents) - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    @staticmethod
    def document(resource: Resource) -> str:
        return " ".join(
            [resource.name, resource.category, resource.city, resource.address, *resource.languages, *resource.tags]
        )

    def order(self, query: str) -> list[str]:
        query_terms = tokens(query)
        scored = []
        for resource, document, length in zip(self.resources, self.documents, self.lengths):
            counts = Counter(document)
            score = 0.0
            for term in query_terms:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (1 - self.b + self.b * length / self.average_length)
                score += self.idf.get(term, 0.0) * frequency * (self.k1 + 1) / denominator
            scored.append((score, resource.name.casefold(), resource.id))
        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        return [resource_id for _, _, resource_id in scored]


def resource_from_json(record: dict[str, Any]) -> Resource:
    return Resource(
        id=record["id"],
        name=record["name"],
        category=record["category"],
        city=record["city"],
        address=record.get("address", ""),
        languages=tuple(record.get("languages", [])),
        tags=tuple(record.get("tags", [])),
    )


def path_order(search: FindSearch, query: str, now: datetime) -> list[str]:
    return [result["id"] for result in search.search(query, now=now, limit=100)]


def fit_order(search: FindSearch, query: str, now: datetime) -> list[str]:
    results = search.search(query, now=now, limit=100)
    return [result["id"] for result in sorted(results, key=lambda result: (-result["fit_score"], result["name"].casefold(), result["id"]))]


def metrics(order: list[str], relevance: dict[str, int]) -> dict[str, float]:
    relevant = set(relevance)
    top = order[:5]
    return {
        "precision_at_5": sum(resource_id in relevant for resource_id in top) / len(top) if top else 0.0,
        "ndcg_at_5": graded_ndcg(order, relevance, 5),
        "mrr": reciprocal_rank(order, relevant),
        "map": average_precision(order, relevant),
    }


def load_adjudicated_relevance(path: Path) -> tuple[dict[str, dict[str, int]], int]:
    relevance: dict[str, dict[str, int]] = {}
    judgment_count = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label = row.get("label", "").strip()
            if label == "":
                continue
            try:
                parsed_label = int(label)
            except ValueError as error:
                raise ValueError(f"Invalid adjudication label: {label!r}") from error
            if not 0 <= parsed_label <= 3:
                raise ValueError(f"Adjudication labels must be between 0 and 3: {parsed_label}")
            case_id = row.get("case_id", "").strip()
            candidate_id = row.get("candidate_id", "").strip()
            if not case_id or not candidate_id:
                raise ValueError("Adjudication rows require case_id and candidate_id")
            relevance.setdefault(case_id, {})[candidate_id] = parsed_label
            judgment_count += 1
    return relevance, judgment_count


def evaluate(
    corpus_path: Path = DEFAULT_CORPUS,
    adjudication_path: Path | None = None,
) -> dict[str, Any]:
    corpus_path = Path(corpus_path).resolve()
    if adjudication_path is not None:
        adjudication_path = Path(adjudication_path).resolve()
    fixture = json.loads(corpus_path.read_text(encoding="utf-8"))
    now = datetime.fromisoformat(fixture["now"])
    resources = [resource_from_json(record) for record in fixture["resources"]]
    adjudicated_relevance = {}
    judgment_count = None
    if adjudication_path is not None:
        adjudicated_relevance, judgment_count = load_adjudicated_relevance(adjudication_path)
    bm25 = BM25(resources)
    rows = []
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
        orders = {
            "pathways": path_order(search, case["query"], now),
            "fit_only": fit_order(search, case["query"], now),
            "bm25": bm25.order(case["query"]),
        }
        relevance = (
            adjudicated_relevance[case["id"]]
            if adjudication_path is not None
            else {key: int(value) for key, value in case["relevance"].items()}
        )
        row = {"id": case["id"], "query": case["query"], "relevance": relevance}
        for baseline, order in orders.items():
            row[f"{baseline}_top_ids"] = order[:10]
            row[baseline] = metrics(order, relevance)
        rows.append(row)

    summary = {}
    for baseline in ("pathways", "fit_only", "bm25"):
        for metric in ("precision_at_5", "ndcg_at_5", "mrr", "map"):
            summary[f"{baseline}_{metric}"] = sum(row[baseline][metric] for row in rows) / len(rows)
    summary["pathways_minus_bm25_ndcg_at_5"] = summary["pathways_ndcg_at_5"] - summary["bm25_ndcg_at_5"]
    summary["pathways_minus_fit_only_ndcg_at_5"] = summary["pathways_ndcg_at_5"] - summary["fit_only_ndcg_at_5"]

    report = {
        "schema_version": 1,
        "fixture": str(corpus_path.relative_to(ROOT)),
        "generated_at": now.isoformat(),
        "evaluation_type": "independent_adjudication" if adjudication_path else "controlled_development_corpus",
        "corpus_size": len(resources),
        "case_count": len(rows),
        "baselines": ["pathways", "fit_only", "bm25"],
        "summary": summary,
        "cases": rows,
    }
    if adjudication_path is not None:
        report["adjudication"] = str(adjudication_path.relative_to(ROOT))
        report["judgment_count"] = judgment_count
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--adjudication", type=Path, help="CSV file containing independent 0-3 relevance labels.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate(args.corpus, adjudication_path=args.adjudication)
    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
