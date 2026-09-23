"""Compare automated missing-data benchmark conditions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--fuzzy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from missing_data_benchmark import score_predictions

    public_rows = {row["benchmark_id"]: row for row in map(json.loads, args.public.read_text().splitlines())}

    def leakage_count(path: Path) -> int:
        count = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            prediction = json.loads(line)
            target = public_rows.get(prediction.get("benchmark_id"), {})
            identity_fields = ("name", "address", "city")
            if any(prediction.get(field) and prediction.get(field) == target.get(field) for field in identity_fields):
                count += 1
        return count

    reports = {
        "first_result_places": score_predictions(args.public, args.gold, args.baseline),
        "fuzzy_places_abstention": score_predictions(args.public, args.gold, args.fuzzy),
    }
    comparison = {
        "schema_version": 1,
        "evaluation_type": "missing_data_condition_comparison",
        "conditions": {
            name: report["metrics"] for name, report in reports.items()
        },
        "interpretation": {
            "first_result_places": "Invalid for entity accuracy if identity leakage is detected; original runner copied target identity fields.",
            "fuzzy_places_abstention": "Branch-aware fuzzy matching with abstention when identity evidence is weak.",
        },
        "identity_leakage_rows": {
            "first_result_places": leakage_count(args.baseline),
            "fuzzy_places_abstention": leakage_count(args.fuzzy),
        },
    }
    args.output.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
