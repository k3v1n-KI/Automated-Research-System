"""Pathways adapter for the existing ARS web-research pipeline."""

from __future__ import annotations

import asyncio
from typing import Any, Callable
from uuid import uuid4


def build_research_prompt(job: dict[str, Any]) -> str:
    if job["trigger_type"] == "missing_field":
        return (
            f"Find the current {job['field']} for the community resource identified as "
            f"{job.get('resource_id')}. {job['prompt']} Return the resource identity, "
            f"the requested field, source URL, and a short evidence excerpt. Do not guess."
        )
    return (
        f"Answer this Pathways community-resource Ask with evidence-backed services: "
        f"{job['prompt']} Return service name, location, relevant details, source URL, "
        f"and a short evidence excerpt. Do not invent services or facts."
    )


def _records_to_suggestions(job: dict[str, Any], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    suggestions = []
    for record in records:
        source_url = str(record.get("source_url") or record.get("url") or "").strip()
        value = record.get("value")
        field = job.get("field")
        if field:
            value = record.get(field, value)
        if not source_url or value in (None, ""):
            continue
        suggestions.append(
            {
                "suggestion_id": f"suggestion-{uuid4()}",
                "job_id": job["id"],
                "resource_id": job.get("resource_id"),
                "field": field,
                "value": str(value),
                "confidence": record.get("confidence"),
                "source_url": source_url,
                "evidence": str(record.get("evidence") or record.get("snippet") or ""),
            }
        )
    return suggestions


def default_executor(job: dict[str, Any]) -> list[dict[str, Any]]:
    """Run the existing LangGraph ARS pipeline and return extracted records."""
    from algorithm import build_research_algorithm

    async def invoke() -> dict[str, Any]:
        graph = build_research_algorithm()
        return await graph.ainvoke(
            {
                "initial_prompt": build_research_prompt(job),
                "column_specs": [job.get("field") or "service", "source_url", "evidence"],
                "queries": [],
                "search_results": [],
                "validated_urls": [],
                "validated_results": [],
                "scraped_content": [],
                "extracted_items": [],
                "final_dataset": [],
                "session_id": f"pathways-{job['id']}",
                "round": 1,
                "error": None,
                "previous_session_id": None,
                "tweak_instructions": None,
                "previous_queries": [],
                "previous_items": [],
                "columns": [],
                "priority_columns": [job.get("field") or "service"],
                "hard_identifier_columns": [],
                "soft_identifier_columns": [],
                "pathways_limits": {"max_queries": 2, "max_urls": 8},
            }
        )

    result = asyncio.run(invoke())
    return result.get("final_dataset") or result.get("extracted_items") or []


def run_research_job(
    database: Any,
    job_id: str,
    executor: Callable[[dict[str, Any]], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    jobs = [job for job in database.list_research_jobs() if job["id"] == job_id]
    if not jobs:
        return {"status": "not_found", "job_id": job_id}
    job = database.claim_research_job(job_id)
    if job is None:
        return {"status": "busy", "job_id": job_id}
    try:
        records = (executor or default_executor)(job)
        suggestions = _records_to_suggestions(job, records)
        for suggestion in suggestions:
            database.add_research_suggestion(**suggestion)
        if not suggestions:
            database.fail_research_job(job_id, "ARS returned no evidence-backed suggestions")
            return {"status": "failed", "job_id": job_id, "suggestions": []}
        return {"status": "needs_review", "job_id": job_id, "suggestions": suggestions}
    except Exception as error:
        database.fail_research_job(job_id, f"{type(error).__name__}: {error}")
        return {"status": "failed", "job_id": job_id, "error": str(error)}
