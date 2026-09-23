"""Automated missing-data benchmark with sealed expected values.

The public fixture never contains the removed field values. The gold fixture is
written separately and must not be passed to ARS during a run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from random import Random
from typing import Any

from evaluation.entity_matching import entity_match_score

FIELD_MAP = {
    "name": "Name",
    "address": "Address",
    "city": "City",
    "phone": "Phone Number",
    "website": "Website",
    "postal_code": "Zip Code",
}
DATASET_FILES = ("Addiction_Dataset.csv", "Hospital_Dataset.csv", "Pharmacy_Dataset.csv")
SCORABLE_FIELDS = tuple(FIELD_MAP)


def clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def norm_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", clean(value).casefold())


def norm_phone(value: Any) -> str:
    digits = re.sub(r"\D", "", clean(value))
    return digits[1:] if len(digits) == 11 and digits.startswith("1") else digits


def norm_url(value: Any) -> str:
    return clean(value).casefold().rstrip("/")


def norm_field(field: str, value: Any) -> str:
    if field == "phone":
        return norm_phone(value)
    if field == "website":
        return norm_url(value)
    return norm_text(value)


def benchmark_id(source_file: str, source_row_number: int) -> str:
    return hashlib.sha256(f"{source_file}:{source_row_number}".encode()).hexdigest()[:16]


def read_rows(input_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for filename in DATASET_FILES:
        with (input_dir / filename).open(newline="", encoding="utf-8-sig") as handle:
            for row_number, row in enumerate(csv.DictReader(handle), start=2):
                normalized = {
                    field: clean(row.get(source_column, ""))
                    for field, source_column in FIELD_MAP.items()
                }
                normalized.update({"source_file": filename, "source_row_number": row_number})
                rows.append(normalized)
    return rows


def generate_benchmark(input_dir: Path, output_dir: Path, sample_size: int = 500, seed: int = 20260922) -> dict[str, Any]:
    rows = [row for row in read_rows(input_dir) if row["name"] and row["city"]]
    rng = Random(seed)
    selected = rng.sample(rows, min(sample_size, len(rows)))
    public_rows = []
    gold_rows = []
    for index, row in enumerate(selected, 1):
        available = [field for field in SCORABLE_FIELDS if row[field] and field not in {"name", "city"}]
        if not available:
            continue
        missing_field = rng.choice(available)
        benchmark_key = benchmark_id(row["source_file"], row["source_row_number"])
        public = {field: row[field] for field in SCORABLE_FIELDS if field != missing_field}
        public.update({"benchmark_id": benchmark_key, "source_file": row["source_file"], "source_row_number": row["source_row_number"], "missing_fields": [missing_field]})
        gold = {"benchmark_id": benchmark_key, "missing_fields": [missing_field], "expected": {missing_field: row[missing_field]}}
        public_rows.append(public)
        gold_rows.append(gold)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "missing_data_public.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in public_rows), encoding="utf-8")
    (output_dir / "missing_data_gold.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in gold_rows), encoding="utf-8")
    metadata = {"schema_version": 1, "seed": seed, "requested_rows": sample_size, "benchmark_rows": len(public_rows), "gold_file": "missing_data_gold.jsonl"}
    (output_dir / "missing_data_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def entity_supported(prediction: dict[str, Any], benchmark_row: dict[str, Any]) -> bool:
    return entity_match_score(prediction, benchmark_row)["supported"]


def score_predictions(public_path: Path, gold_path: Path, predictions_path: Path) -> dict[str, Any]:
    public = {row["benchmark_id"]: row for row in map(json.loads, public_path.read_text().splitlines())}
    gold = {row["benchmark_id"]: row for row in map(json.loads, gold_path.read_text().splitlines())}
    predictions = {row["benchmark_id"]: row for row in map(json.loads, predictions_path.read_text().splitlines())}
    totals = {"requested": 0, "filled": 0, "exact": 0, "entity_supported": 0, "wrong_entity": 0, "unresolved": 0, "unexpected_fills": 0}
    field_results = []
    for benchmark_id, gold_row in gold.items():
        target = public[benchmark_id]
        prediction = predictions.get(benchmark_id, {})
        allowed = set(gold_row["missing_fields"])
        for field, value in prediction.items():
            if field in {"name", "address", "city", "postal_code", "website", "source", "source_url", "evidence", "place_match"}:
                continue
            if field in SCORABLE_FIELDS and field not in allowed and clean(value):
                totals["unexpected_fills"] += 1
        for field in gold_row["missing_fields"]:
            totals["requested"] += 1
            candidate = clean(prediction.get(field))
            if not candidate:
                totals["unresolved"] += 1
                field_results.append({"benchmark_id": benchmark_id, "field": field, "status": "unresolved"})
                continue
            match = entity_match_score(prediction, target)
            supported = match["supported"]
            totals["filled"] += 1
            if not supported:
                totals["wrong_entity"] += 1
                field_results.append({"benchmark_id": benchmark_id, "field": field, "status": "wrong_entity", "value": candidate, "entity_match": match})
                continue
            totals["entity_supported"] += 1
            exact = norm_field(field, candidate) == norm_field(field, gold_row["expected"].get(field))
            if exact:
                totals["exact"] += 1
            field_results.append({"benchmark_id": benchmark_id, "field": field, "status": "exact" if exact else "mismatch", "value": candidate, "entity_match": match})
    requested = totals["requested"] or 1
    return {"schema_version": 1, "evaluation_type": "automated_sealed_benchmark", "metrics": {**totals, "fill_rate": totals["filled"] / requested, "validated_fill_rate": totals["entity_supported"] / requested, "exact_rate": totals["exact"] / requested, "wrong_entity_rate": totals["wrong_entity"] / requested, "unexpected_fill_rate": totals["unexpected_fills"] / requested}, "fields": field_results}


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--input-dir", type=Path, default=Path(__file__).parent.parent)
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.add_argument("--sample-size", type=int, default=500)
    generate.add_argument("--seed", type=int, default=20260922)
    score = subparsers.add_parser("score")
    score.add_argument("--public", type=Path, required=True)
    score.add_argument("--gold", type=Path, required=True)
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "generate":
        print(json.dumps(generate_benchmark(args.input_dir, args.output_dir, args.sample_size, args.seed), indent=2))
    else:
        args.output.write_text(json.dumps(score_predictions(args.public, args.gold, args.predictions), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
