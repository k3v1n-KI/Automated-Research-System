"""Evaluate natural-language filter extraction for Study 1."""

from __future__ import annotations

import argparse
import json
import os
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


def _category(case_id: str) -> str:
    return "canonical" if case_id.startswith("c") else "paraphrased"


def evaluate(
    cases_path: Path = DEFAULT_CASES,
    resources_path: Path | None = None,
    extractor: str = "baseline",
    llm_mode: str = "off",
    semantic_mode: str = "on",
    alias_mode: str | None = None,
    batch_size: int = 1,
) -> dict[str, Any]:
    cases_path = Path(cases_path).resolve()
    fixture = json.loads(cases_path.read_text(encoding="utf-8"))
    if resources_path is None:
        search = FindSearch(PathwaysStore(seed_resources()))
    else:
        search = FindSearch.from_jsonl(Path(resources_path).resolve())
    if extractor not in {"baseline", "hybrid"}:
        raise ValueError(f"Unknown extractor: {extractor}")
    if llm_mode not in {"off", "mock", "openai", "gemini"}:
        raise ValueError(f"Unknown llm_mode: {llm_mode}")
    if semantic_mode not in {"off", "on"}:
        raise ValueError(f"Unknown semantic_mode: {semantic_mode}")
    if alias_mode is None:
        alias_mode = "on" if extractor == "hybrid" else "off"
    if alias_mode not in {"off", "on"}:
        raise ValueError(f"Unknown alias_mode: {alias_mode}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    rows = []
    true_positive = false_positive = false_negative = 0
    field_correct = {field: 0 for field in FILTER_FIELDS}

    def mock_llm(prompt: str) -> dict[str, Any]:
        request = json.loads(prompt)
        query = request["query"].casefold()
        # Test double for the constrained proposal contract, not production logic.
        if ("teenager" in query or "teen" in query) and ("addiction" in query or "walk-in" in query):
            return {"population": ["adolescent"]}
        if "teenager" in query or "teen" in query:
            return {"population": ["youth"]}
        return {}

    def openai_llm(prompt: str) -> dict[str, Any]:
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "Return only a JSON object containing conservative filter suggestions. Abstain with {} when unsupported.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return json.loads(response.choices[0].message.content or "{}")

    def gemini_llm(prompt: str) -> dict[str, Any]:
        from openai import OpenAI
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[2] / "searxng" / ".env")

        api_key = os.environ["GEMINI_API_KEY"]
        client = OpenAI(
            api_key=api_key,
            base_url=os.getenv(
                "GEMINI_OPENAI_BASE_URL",
                "https://generativelanguage.googleapis.com/v1beta/openai/",
            ),
        )
        response = client.chat.completions.create(
            model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "Return only a JSON object containing conservative filter suggestions. Abstain with {} when unsupported.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return json.loads(response.choices[0].message.content or "{}")

    def gemini_batch(prompts: list[tuple[str, str]]) -> dict[str, dict[str, Any]]:
        from openai import OpenAI
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[2] / "searxng" / ".env")
        client = OpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url=os.getenv(
                "GEMINI_OPENAI_BASE_URL",
                "https://generativelanguage.googleapis.com/v1beta/openai/",
            ),
        )
        request = {
            "requests": [
                {"id": case_id, "prompt": json.loads(prompt)}
                for case_id, prompt in prompts
            ]
        }
        response = client.chat.completions.create(
            model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "For every request, return a conservative ontology-constrained suggestion. Return only JSON with a results array; each item must contain the supplied id and a suggestion object, using {} when unsupported.",
                },
                {"role": "user", "content": json.dumps(request)},
            ],
        )
        payload = json.loads(response.choices[0].message.content or "{}")
        results = payload.get("results", []) if isinstance(payload, dict) else []
        return {
            item["id"]: item.get("suggestion", {})
            for item in results
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        }

    batched_suggestions: dict[str, dict[str, Any]] = {}
    if llm_mode == "gemini" and batch_size > 1:
        prompts = [
            (
                case["id"],
                search._llm_filter_prompt(
                    case["query"],
                    [
                        token
                        for token in case["query"].casefold().split()
                        if len(token) > 2
                    ],
                ),
            )
            for case in fixture["cases"]
        ]
        for start in range(0, len(prompts), batch_size):
            batched_suggestions.update(gemini_batch(prompts[start:start + batch_size]))

    for case in fixture["cases"]:
        use_trace = extractor == "hybrid" or alias_mode == "off" or llm_mode != "off" or semantic_mode == "on"
        trace = (
            search.extract_filters_hybrid_trace(
                case["query"],
                llm_proposer=(
                    mock_llm if llm_mode == "mock"
                    else openai_llm if llm_mode == "openai"
                    else (
                        (lambda prompt, case_id=case["id"]: batched_suggestions.get(case_id, {}))
                        if llm_mode == "gemini" and batch_size > 1
                        else gemini_llm
                    ) if llm_mode == "gemini"
                    else None
                ),
                semantic_enabled=semantic_mode == "on",
                aliases_enabled=alias_mode == "on",
            )
            if use_trace
            else {
                "filters": search.extract_filters(case["query"]),
                "matched_aliases": [],
                "unresolved_terms": [],
                "extraction_route": "baseline_rules",
            }
        )
        predicted = trace["filters"]
        mismatches = []
        case_exact = True
        field_results = {}
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
            field_results[field] = {
                "expected": case["gold"].get(field, [] if field in SET_FIELDS else ""),
                "predicted": getattr(predicted, field),
                "exact": actual == expected,
            }
        rows.append({
            "id": case["id"],
            "category": _category(case["id"]),
            "query": case["query"],
            "gold_filter": case["gold"],
            "gold_filters": case["gold"],
            "system_output": {field: getattr(predicted, field) for field in FILTER_FIELDS},
            "system_filters": {field: getattr(predicted, field) for field in FILTER_FIELDS},
            "field_results": field_results,
            "matched_aliases": trace["matched_aliases"],
            "unresolved_terms": trace["unresolved_terms"],
            "semantic_matches": trace.get("semantic_matches", []),
            "semantic_filters": trace.get("semantic_filters", {}),
            "extraction_route": trace["extraction_route"],
            "llm_involved": trace.get("llm_involved", False),
            "llm_suggestion": trace.get("llm_suggestion"),
            "llm_provenance": trace.get("llm_provenance"),
            "exact_match": case_exact,
            "mismatched_fields": mismatches,
        })

    denominator = true_positive + false_positive
    precision = true_positive / denominator if denominator else 1.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    case_count = len(rows)
    category_summary = {}
    for category in ("canonical", "paraphrased"):
        category_rows = [row for row in rows if row["category"] == category]
        category_summary[category] = {
            "case_count": len(category_rows),
            "exact_case_accuracy": (
                sum(row["exact_match"] for row in category_rows) / len(category_rows)
                if category_rows else 0.0
            ),
        }
    return {
        "schema_version": 3,
        "evaluation_type": "study1_filter_extraction" if extractor == "baseline" else "study1_hybrid_filter_extraction",
        "extractor": extractor,
        "ontology_version": "hybrid-aliases-v1" if extractor == "hybrid" else None,
        "llm_mode": llm_mode,
        "alias_mode": alias_mode,
        "llm_batch_size": batch_size,
        "llm_provider": {
            "mode": llm_mode,
            "model": (
                os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
                if llm_mode == "gemini"
                else os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                if llm_mode == "openai"
                else "deterministic-mock"
                if llm_mode == "mock"
                else None
            ),
        },
        "semantic_mode": semantic_mode,
        "llm_summary": {
            "involved_case_count": sum(row["llm_involved"] for row in rows),
            "involved_case_ids": [row["id"] for row in rows if row["llm_involved"]],
            "suggestions": [
                {"id": row["id"], "suggestion": row["llm_suggestion"]}
                for row in rows if row["llm_involved"] and row["llm_suggestion"]
            ],
            "note": "Mock mode uses an injected deterministic provider; openai mode requires OPENAI_API_KEY; gemini mode uses the OpenAI-compatible Gemini endpoint and requires GEMINI_API_KEY; off mode makes no LLM calls.",
        },
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
            "by_category": category_summary,
            "unresolved_term_count": sum(len(row["unresolved_terms"]) for row in rows),
        },
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--resources", type=Path)
    parser.add_argument("--extractor", choices=("baseline", "hybrid"), default="baseline")
    parser.add_argument("--llm-mode", choices=("off", "mock", "openai", "gemini"), default="off")
    parser.add_argument("--semantic-mode", choices=("off", "on"), default="on")
    parser.add_argument("--alias-mode", choices=("off", "on"))
    parser.add_argument("--llm-batch-size", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(evaluate(args.cases, resources_path=args.resources, extractor=args.extractor, llm_mode=args.llm_mode, semantic_mode=args.semantic_mode, alias_mode=args.alias_mode, batch_size=args.llm_batch_size), indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
