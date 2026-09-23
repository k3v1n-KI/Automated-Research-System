"""Run automated missing-field predictions with Google Places evidence.

This is a bounded Study 4 runner. It writes checkpointed predictions and never
loads the sealed gold file during prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import sleep

from google_places import search_places


def load_public(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def predict_row(row: dict) -> dict:
    query = " ".join(str(row.get(field, "")) for field in ("name", "address", "city") if row.get(field))
    places = search_places(query, max_results=5)
    prediction = {"benchmark_id": row["benchmark_id"], "source": "google_places"}
    candidates = []
    for place in places:
        candidate = dict(place)
        from evaluation.entity_matching import entity_match_score
        candidate["match"] = entity_match_score(candidate, row)
        candidates.append(candidate)
    candidates.sort(key=lambda candidate: (-candidate["match"]["score"], candidate.get("name", "").casefold()))
    best = candidates[0] if candidates and candidates[0]["match"]["supported"] else None
    if best:
        prediction.update({field: best.get(field, "") for field in ("name", "address", "city", "postal_code", "website")})
        prediction["place_match"] = best["match"]
        prediction["source_url"] = best.get("source_url", "")
        prediction["evidence"] = best.get("evidence", "")
    else:
        return prediction
    for field in row["missing_fields"]:
        value = best.get(field, "")
        if value:
            prediction[field] = value
    return prediction


def run(public_path: Path, output_path: Path, limit: int | None = None, pause_seconds: float = 0.0, condition: str = "google_places_fuzzy") -> dict:
    rows = load_public(public_path)
    if limit is not None:
        rows = rows[:limit]
    existing = {}
    if output_path.exists():
        existing = {row["benchmark_id"]: row for row in load_public(output_path)}
    completed = dict(existing)
    for index, row in enumerate(rows, 1):
        if row["benchmark_id"] in completed:
            continue
        completed[row["benchmark_id"]] = predict_row(row)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in completed.values()), encoding="utf-8")
        print(f"completed={index}/{len(rows)} benchmark_id={row['benchmark_id']}", flush=True)
        if pause_seconds:
            sleep(pause_seconds)
    manifest = {
        "schema_version": 1,
        "condition": condition,
        "public_input": str(public_path),
        "public_sha256": hashlib.sha256(public_path.read_bytes()).hexdigest(),
        "rows_requested": len(rows),
        "rows_completed": len(completed),
        "output": str(output_path),
        "matcher": "rapidfuzz name/address/city/postal/website with abstention",
        "provider": "Google Places Text Search",
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--pause-seconds", type=float, default=0.0)
    parser.add_argument("--condition", default="google_places_fuzzy")
    args = parser.parse_args()
    print(json.dumps(run(args.public, args.output, args.limit, args.pause_seconds, args.condition), indent=2))


if __name__ == "__main__":
    main()
