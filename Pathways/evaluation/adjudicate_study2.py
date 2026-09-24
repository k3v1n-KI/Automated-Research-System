"""Build an independent adjudication bundle for Study 2 ranking judgments."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = Path(__file__).with_name("study2_corpus.json")


def _normalize_label(value: Any) -> int | None:
    try:
        label = int(value)
    except (TypeError, ValueError):
        return None
    return label if 0 <= label <= 3 else None


def generate_adjudication_bundle(corpus_path: str | Path) -> dict[str, Any]:
    """Create a reviewer-facing bundle containing every case-resource pair.

    Each item is left unlabeled so a human reviewer can score it independently
    on a 0–3 relevance scale without exposing the development labels from the
    controlled fixture.
    """
    path = Path(corpus_path)
    fixture = json.loads(path.read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc)

    resources = fixture["resources"]
    items: list[dict[str, Any]] = []
    for case in fixture["cases"]:
        for resource in resources:
            items.append(
                {
                    "case_id": case["id"],
                    "query": case["query"],
                    "candidate_id": resource["id"],
                    "name": resource["name"],
                    "category": resource["category"],
                    "city": resource["city"],
                    "address": resource.get("address", ""),
                    "languages": resource.get("languages", []),
                    "tags": resource.get("tags", []),
                    "label": None,
                    "label_name": None,
                    "notes": "",
                }
            )

    return {
        "schema_version": 1,
        "evaluation_type": "independent_adjudication_bundle",
        "generated_at": now.isoformat(),
        "fixture": str(path.relative_to(ROOT)) if path.is_absolute() else str(path),
        "resource_count": len(resources),
        "case_count": len(fixture["cases"]),
        "instructions": {
            "scale": {
                "0": "not relevant",
                "1": "weakly relevant / partial fit",
                "2": "relevant / strong fit",
                "3": "highly relevant / exact fit",
            },
            "task": "For each query, judge whether the candidate resource is a good answer for the requester, independent of any existing system ranking or development labels.",
        },
        "items": items,
    }


def score_adjudication_labels(bundle: dict[str, Any]) -> dict[str, Any]:
    """Summarize an adjudication bundle after labels have been assigned."""
    items = bundle.get("items", [])
    case_ids = sorted({item["case_id"] for item in items})
    labeled_items = [item for item in items if _normalize_label(item.get("label")) is not None]

    case_summaries = []
    for case_id in case_ids:
        case_labels = [_normalize_label(item.get("label")) for item in items if item.get("case_id") == case_id]
        case_labels = [label for label in case_labels if label is not None]
        if case_labels:
            case_summaries.append(
                {
                    "case_id": case_id,
                    "items_labeled": len(case_labels),
                    "mean_label": sum(case_labels) / len(case_labels),
                    "max_label": max(case_labels),
                }
            )

    mean_case_label = (
        sum(summary["mean_label"] for summary in case_summaries) / len(case_summaries)
        if case_summaries
        else 0.0
    )

    return {
        "schema_version": 1,
        "evaluation_type": "study2_adjudication_summary",
        "case_count": len(case_ids),
        "resource_count": bundle.get("resource_count", 0),
        "items_labeled": len(labeled_items),
        "mean_case_label": mean_case_label,
        "case_summaries": case_summaries,
    }


def export_bundle_csv(bundle: dict[str, Any], output_path: str | Path) -> Path:
    """Export the adjudication bundle to a simple CSV suitable for human review."""
    path = Path(output_path)
    fieldnames = [
        "case_id",
        "query",
        "candidate_id",
        "name",
        "category",
        "city",
        "address",
        "languages",
        "tags",
        "label",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in bundle.get("items", []):
            writer.writerow(
                {
                    "case_id": item.get("case_id", ""),
                    "query": item.get("query", ""),
                    "candidate_id": item.get("candidate_id", ""),
                    "name": item.get("name", ""),
                    "category": item.get("category", ""),
                    "city": item.get("city", ""),
                    "address": item.get("address", ""),
                    "languages": "; ".join(item.get("languages", [])),
                    "tags": "; ".join(item.get("tags", [])),
                    "label": item.get("label", ""),
                    "notes": item.get("notes", ""),
                }
            )
    return path


def bundle_from_csv(path: str | Path) -> dict[str, Any]:
    """Rehydrate an adjudication bundle from the corresponding CSV export."""
    csv_path = Path(path)
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    items = []
    for row in rows:
        line = {
            "case_id": row.get("case_id", ""),
            "query": row.get("query", ""),
            "candidate_id": row.get("candidate_id", ""),
            "name": row.get("name", ""),
            "category": row.get("category", ""),
            "city": row.get("city", ""),
            "address": row.get("address", ""),
            "languages": [value.strip() for value in row.get("languages", "").split(";") if value.strip()],
            "tags": [value.strip() for value in row.get("tags", "").split(";") if value.strip()],
            "label": _normalize_label(row.get("label", "")),
            "notes": row.get("notes", ""),
        }
        items.append(line)

    resource_count = len({item["candidate_id"] for item in items}) if items else 0
    case_count = len({item["case_id"] for item in items}) if items else 0
    return {
        "schema_version": 1,
        "evaluation_type": "independent_adjudication_bundle",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "resource_count": resource_count,
        "case_count": case_count,
        "items": items,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path, help="Optional CSV export for reviewer scoring.")
    parser.add_argument("--score", type=Path, help="Optional CSV file to score and summarize after reviewer labeling.")
    args = parser.parse_args()

    bundle = generate_adjudication_bundle(args.corpus)
    if args.csv:
        export_bundle_csv(bundle, args.csv)
        print(f"Exported reviewer CSV to {args.csv}")
    if args.score:
        scored_bundle = bundle_from_csv(args.score)
        print(json.dumps(score_adjudication_labels(scored_bundle), indent=2))
        raise SystemExit(0)

    args.output.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote adjudication bundle for {bundle['case_count']} cases across {bundle['resource_count']} resources to {args.output}")


if __name__ == "__main__":
    main()
