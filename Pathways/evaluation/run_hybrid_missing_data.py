"""Run the original ARS fallback only for unresolved Study 4 rows."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import sleep


WORKER_SCRIPT = """
import json
import sys
from pathways_ars_worker import default_executor
job = json.loads(sys.stdin.read())
print(json.dumps(default_executor(job)))
"""


def load_rows(path: Path) -> dict[str, dict]:
    return {json.loads(line)["benchmark_id"]: json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def unresolved_ids(public: dict[str, dict], places: dict[str, dict]) -> list[str]:
    return [benchmark_id for benchmark_id, row in public.items() if not places.get(benchmark_id, {}).get(row["missing_fields"][0])]


def run_executor_isolated(job: dict) -> list[dict]:
    completed = subprocess.run(
        [sys.executable, "-u", "-c", WORKER_SCRIPT],
        input=json.dumps(job),
        text=True,
        capture_output=True,
        timeout=300,
        check=True,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1]) if lines else []


def run(public_path: Path, places_path: Path, output_path: Path, limit: int | None = None, pause_seconds: float = 0.0) -> dict:
    public = load_rows(public_path)
    places = load_rows(places_path)
    unresolved = unresolved_ids(public, places)
    if limit is not None:
        unresolved = unresolved[:limit]
    merged = dict(places)
    for index, benchmark_id in enumerate(unresolved, 1):
        row = public[benchmark_id]
        field = row["missing_fields"][0]
        job = {
            "id": f"study4-hybrid-{benchmark_id}",
            "trigger_type": "missing_field",
            "resource_id": None,
            "field": field,
            "prompt": (
                f"Find the current {field} for {row.get('name', '')} at {row.get('address', '')}, "
                f"{row.get('city', '')}. Use official websites, government registries, and regional directories. "
                "Return only evidence-backed values with source URL and excerpt."
            ),
            "resource_name": row.get("name", ""),
            "resource_address": row.get("address", ""),
            "resource_city": row.get("city", ""),
            "resource_website": row.get("website", ""),
        }
        try:
            records = run_executor_isolated(job)
        except (subprocess.SubprocessError, json.JSONDecodeError) as error:
            print(f"failed={index}/{len(unresolved)} benchmark_id={benchmark_id} error={error}", flush=True)
            records = []
        prediction = {
            "benchmark_id": benchmark_id,
            "name": row.get("name", ""),
            "city": row.get("city", ""),
            "source": "ars_fallback",
        }
        for record in records:
            value = record.get(field) or record.get("value") or record.get("hours")
            if value:
                prediction[field] = value
                prediction["source_url"] = record.get("source_url") or record.get("url", "")
                prediction["evidence"] = record.get("evidence") or record.get("snippet", "")
                break
        merged[benchmark_id] = prediction
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in merged.values()), encoding="utf-8")
        print(f"completed={index}/{len(unresolved)} benchmark_id={benchmark_id}", flush=True)
        if pause_seconds:
            sleep(pause_seconds)
    return {"places_rows": len(places), "unresolved_input": len(unresolved), "merged_rows": len(merged), "output": str(output_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--places", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--pause-seconds", type=float, default=0.0)
    args = parser.parse_args()
    print(json.dumps(run(args.public, args.places, args.output, args.limit, args.pause_seconds), indent=2))


if __name__ == "__main__":
    main()
