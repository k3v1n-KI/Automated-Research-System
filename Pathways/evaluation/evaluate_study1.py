"""Evaluate natural-language filter extraction for Study 1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pathways_domain import PathwaysStore, seed_resources
from pathways_find import FindSearch, SearchFilters

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = Path(__file__).with_name("study1_extraction_cases.json")
DEFAULT_RESOURCES = ROOT / "data" / "processed" / "normalized_resources.jsonl"
FILTER_FIELDS = ("need", "modality", "population", "coverage", "location", "language", "urgency")
SET_FIELDS = {"need", "modality", "population", "coverage", "language"}


def _field_values(filters: SearchFilters, field: str) -> set[str]:
    value = getattr(filters, field)
    if field in SET_FIELDS:
        return set(value)
    return {value} if value else set()


def _gold_values(gold: dict[str, Any], field: str) -> set[str]:
    value = gold.get(field, [] if field in SET_FIELDS else "")
    if field in SET_FIELDS:
        return set(value)
    return {value} if value else set()


def evaluate(cases_path: Path = DEFAULT_CASES, resources_path: Path | None = None) -> dict[str, Any]:
    cases_path = Path(cases_path).resolve()
    fixture = json.loads(cases_path.read_text(encoding="utf-8"))
    if resources_path is None:
        search = FindSearch(PathwaysStore(seed_resources()))
    else:
        search = FindSearch.from_jsonl(Path(resources_path).resolve())
    rows = []
    true_positive = false_positive = false_negative = 0
    field_correct = {field: 0 for field in FILTER_FIELDS}

    for case in fixture["cases"]:
        predicted = search.extract_filters(case["query"])
        mismatches = []
        case_exact = True
        for field in FILTER_FIELDS:
            actual = _field_values(predicted, field)
            expected = _gold_values(case["gold"], field)
            true_positive += len(actual & expected)
            false_positive += len(actual - expected)
            false_negative += len(expected - actual)
            if actual == expected:
                field_correct[field] += 1
            else:
                case_exact = False
                mismatches.append(field)
        rows.append({
            "id": case["id"],
            "query": case["query"],
            "gold": case["gold"],
            "predicted": {field: getattr(predicted, field) for field in FILTER_FIELDS},
            "exact_match": case_exact,
            "mismatched_fields": mismatches,
        })

    denominator = true_positive + false_positive
    precision = true_positive / denominator if denominator else 1.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    case_count = len(rows)
    return {
        "schema_version": 1,
        "evaluation_type": "study1_filter_extraction",
        "fixture": str(cases_path.relative_to(ROOT)),
        "resources": str(Path(resources_path).resolve().relative_to(ROOT)) if resources_path else "seed_resources",
        "case_count": case_count,
        "fields": list(FILTER_FIELDS),
        "summary": {
            "exact_case_accuracy": sum(row["exact_match"] for row in rows) / case_count if case_count else 0.0,
            "micro_precision": precision,
            "micro_recall": recall,
            "micro_f1": f1,
            "field_exact_accuracy": {
                field: field_correct[field] / case_count if case_count else 0.0
                for field in FILTER_FIELDS
            },
        },
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--resources", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(evaluate(args.cases, resources_path=args.resources), indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
