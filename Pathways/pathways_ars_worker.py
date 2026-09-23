"""Pathways adapter for the existing ARS web-research pipeline."""

from __future__ import annotations

import asyncio
import re
from typing import Any, Callable
from uuid import uuid4


REGION_LOCATION_HINTS = {
    "algoma oht": ("algoma", "sault", "sault ste marie", "sault ste. marie"),
}


def _ask_location_supported(job: dict[str, Any], evidence: str, source_url: str) -> bool:
    prompt = str(job.get("prompt") or "").casefold()
    region_match = re.search(r"region:\s*([^\.]+)", prompt)
    if not region_match or region_match.group(1).strip().casefold() in {"ontario", "ontario-wide"}:
        return True
    region = region_match.group(1).strip()
    hints = REGION_LOCATION_HINTS.get(region)
    if not hints:
        return True
    haystack = f"{evidence} {source_url}".casefold()
    return any(hint in haystack for hint in hints)


def build_research_prompt(job: dict[str, Any]) -> str:
    if job["trigger_type"] == "missing_field":
        resource_context = ", ".join(
            value for value in (
                job.get("resource_name"),
                job.get("resource_address"),
                job.get("resource_city"),
                job.get("resource_website"),
            ) if value
        )
        return (
            f"Find the current {job['field']} for this community resource: "
            f"{resource_context or job.get('resource_id')}. {job['prompt']} Return the resource identity, "
            f"the requested field, source URL, and a short evidence excerpt. Do not guess."
        )
    return (
        f"Answer this Pathways community-resource Ask with evidence-backed services: "
        f"{job['prompt']} Return service name, location, relevant details, source URL, "
        f"and a short evidence excerpt. Do not invent services or facts."
    )


def _records_to_suggestions(job: dict[str, Any], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    suggestions = []
    resource_terms = [
        str(value).casefold().strip()
        for value in (job.get("resource_name"), job.get("resource_address"), job.get("resource_city"))
        if value and str(value).strip()
    ]
    resource_address = str(job.get("resource_address") or "").casefold()
    address_terms = [
        term for term in re.findall(r"[a-z0-9]+", resource_address)
        if len(term) >= 3 and term not in {"street", "road", "avenue", "drive", "crescent", "ontario", "canada"}
    ]
    for record in records:
        source_url = str(record.get("source_url") or record.get("url") or "").strip()
        value = (
            record.get("value")
            or record.get("name")
            or record.get("service")
            or record.get("title")
            or record.get("hours")
            or record.get("opening_hours")
            or record.get("operating_hours")
        )
        field = job.get("field")
        if field:
            value = record.get(field, value)
        evidence = str(record.get("evidence") or record.get("snippet") or "").strip()
        if not source_url or value in (None, ""):
            continue
        if job.get("trigger_type") == "missing_field":
            normalized_evidence = evidence.casefold()
            normalized_value = str(value).casefold().strip()
            value_supported = normalized_value in normalized_evidence
            identity_supported = any(
                len(term) >= 4 and (term in normalized_evidence or term in source_url.casefold())
                for term in resource_terms
            )
            address_supported = not address_terms or sum(term in normalized_evidence for term in address_terms) >= min(2, len(address_terms))
            if not evidence or not value_supported or not identity_supported or not address_supported:
                continue
            confidence = record.get("confidence")
            if confidence is None:
                confidence = 0.85 if identity_supported and value_supported and address_supported else 0.0
        else:
            if not _ask_location_supported(job, evidence, source_url):
                continue
            confidence = record.get("confidence")
        suggestions.append(
            {
                "suggestion_id": f"suggestion-{uuid4()}",
                "job_id": job["id"],
                "resource_id": job.get("resource_id"),
                "field": field,
                "value": str(value),
                "confidence": confidence,
                "source_url": source_url,
                "evidence": evidence,
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
                "pathways_limits": {
                    "max_queries": 4,
                    "max_urls": 8,
                    "scrape_concurrency": 4,
                    "scrape_timeout_ms": 20000,
                    "max_extraction_documents": 4,
                    "max_extraction_chunks": 1,
                },
                "pathways_fallback_queries": [
                    f"{job.get('prompt', '')} {job.get('field', '')}".replace('site:', '').replace('"', '')
                ],
                "pathways_prefer_tavily": True,
            }
        )

    result = asyncio.run(invoke())
    records = result.get("final_dataset") or result.get("extracted_items") or []
    if job.get("trigger_type") in {"missing_field", "ask"}:
        from google_places import search_places

        query = " ".join(
            value for value in (
                job.get("resource_name"), job.get("resource_address"), job.get("resource_city")
            ) if value
        )
        if job.get("trigger_type") == "ask":
            query = job.get("prompt", "")
        try:
            records.extend(search_places(query))
        except Exception as error:
            print(f"Google Places lookup failed: {error}")
    return records


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
        if job.get("resource_id"):
            resource = database.get_resource(job["resource_id"])
            if resource:
                job.update({
                    "resource_name": resource.get("name", ""),
                    "resource_address": resource.get("address", ""),
                    "resource_city": resource.get("city", ""),
                    "resource_website": resource.get("website", ""),
                })
        records = (executor or default_executor)(job)
        suggestions = _records_to_suggestions(job, records)
        persisted_suggestions = []
        for suggestion in suggestions:
            persisted_suggestions.append(database.add_research_suggestion(**suggestion))
        if not suggestions:
            database.fail_research_job(job_id, "ARS returned no evidence-backed suggestions")
            return {"status": "failed", "job_id": job_id, "suggestions": []}
        if job.get("trigger_type") == "ask" and job.get("ask_id"):
            for suggestion in persisted_suggestions:
                database.add_reply(
                    ask_id=job["ask_id"],
                    author="ARS",
                    text=(
                        f"Suggested answer: {suggestion['value']}. "
                        f"Details: {suggestion['evidence'] or 'No additional details were extracted.'} "
                        "Please verify before relying on it. "
                        f"Source: {suggestion['source_url']}"
                    ),
                    candidate_name=suggestion["value"],
                )
        return {"status": "needs_review", "job_id": job_id, "suggestions": persisted_suggestions}
    except Exception as error:
        database.fail_research_job(job_id, f"{type(error).__name__}: {error}")
        return {"status": "failed", "job_id": job_id, "error": str(error)}
