"""Classify entity-supported hybrid benchmark mismatches for review."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz


def clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def phone_base(value: Any) -> str:
    value = re.split(r"\b(?:ext|extension|x)\b", clean(value).casefold())[0]
    digits = re.sub(r"\D", "", value)
    return digits[1:] if len(digits) == 11 and digits.startswith("1") else digits


def domain(value: Any) -> str:
    value = clean(value).casefold().rstrip("/")
    value = re.sub(r"^https?://", "", value)
    return value.split("/", 1)[0].removeprefix("www.")


def address_tokens(value: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", clean(value).casefold()))


def house_number(value: Any) -> str:
    match = re.search(r"\b\d+[a-z]?\b", clean(value).casefold())
    return match.group(0) if match else ""


def classify(field: str, predicted: str, expected: str) -> tuple[str, str]:
    if field == "website":
        if domain(predicted) == domain(expected):
            return "same_domain_different_path", "high_confidence_representation"
        return "different_domain", "needs_manual_verification"

    if field == "phone":
        predicted_base = phone_base(predicted)
        expected_base = phone_base(expected)
        if predicted_base == expected_base:
            return "same_base_number_extension_or_format", "high_confidence_representation"
        if predicted_base[:3] == expected_base[:3]:
            return "same_area_code_different_number", "likely_valid_alternate"
        return "different_number_or_area_code", "needs_manual_verification"

    if field == "postal_code":
        return "different_postal_code", "needs_manual_verification"

    score = fuzz.token_set_ratio(predicted, expected) / 100
    expected_tokens = address_tokens(expected)
    overlap = len(address_tokens(predicted) & expected_tokens) / max(1, len(expected_tokens))
    same_house = house_number(predicted) == house_number(expected) and bool(house_number(expected))
    if score >= 0.90:
        return "near_equivalent_address_wording", "high_confidence_representation"
    if score >= 0.75 and same_house:
        return "same_street_number_expanded_or_abbreviated", "high_confidence_representation"
    if same_house and overlap >= 0.30:
        return "same_street_number_partial_overlap", "likely_valid_alternate"
    return "materially_different_address_text", "needs_manual_verification"


def build_report(report_path: Path, gold_path: Path, output_path: Path) -> dict[str, Any]:
    evaluation = json.loads(report_path.read_text(encoding="utf-8"))
    gold = {row["benchmark_id"]: row for row in map(json.loads, gold_path.read_text(encoding="utf-8").splitlines())}
    rows = []
    for result in evaluation["fields"]:
        if result["status"] != "mismatch":
            continue
        benchmark_id = result["benchmark_id"]
        field = result["field"]
        predicted = clean(result["value"])
        expected = clean(gold[benchmark_id]["expected"].get(field))
        category, tier = classify(field, predicted, expected)
        rows.append(
            {
                "benchmark_id": benchmark_id,
                "field": field,
                "category": category,
                "tier": tier,
                "entity_match": result["entity_match"],
                "predicted": predicted,
                "expected": expected,
            }
        )

    report = {
        "schema_version": 1,
        "source_report": str(report_path),
        "evaluation_type": "entity_supported_mismatch_audit",
        "mismatch_count": len(rows),
        "categories": dict(Counter(row["category"] for row in rows)),
        "tiers": dict(Counter(row["tier"] for row in rows)),
        "rows": rows,
    }
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.report, args.gold, args.output)
    print(json.dumps({key: report[key] for key in ("mismatch_count", "categories", "tiers")}, indent=2))


if __name__ == "__main__":
    main()