"""Deterministic Find search over the normalized Pathways corpus."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from pathways_domain import PathwaysStore, Resource


STOP_WORDS = {
    "a", "an", "and", "any", "for", "find", "in", "me", "near", "of",
    "service", "services", "the", "to", "with",
}


@dataclass(frozen=True)
class SearchFilters:
    need: tuple[str, ...] = ()
    modality: tuple[str, ...] = ()
    population: tuple[str, ...] = ()
    coverage: tuple[str, ...] = ()
    location: str = ""
    language: tuple[str, ...] = ()
    urgency: str = ""


@dataclass(frozen=True)
class SearchLog:
    query: str
    filters: SearchFilters
    returned_ids: tuple[str, ...]
    scores: tuple[float, ...]
    created_at: datetime


@dataclass
class FindSearch:
    """Search adapter backed by a PathwaysStore and normalized resources."""

    store: PathwaysStore
    logs: list[SearchLog] = field(default_factory=list)

    @classmethod
    def from_jsonl(cls, path: Path) -> "FindSearch":
        resources = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                resources.append(
                    Resource(
                        id=record["candidate_id"],
                        name=record["name"],
                        category=record["category"],
                        city=record["city"],
                        domain=record["domain"],
                        address=record["address"],
                        phone=record["phone"],
                        website=record["website"],
                        postal_code=record["postal_code"],
                        source_url=record["source_url"],
                        source_dataset=record["source_dataset"],
                        source_row_number=record["source_row_number"],
                        tags=(record["domain"],),
                    )
                )
        return cls(PathwaysStore(resources))

    def extract_filters(self, query: str) -> SearchFilters:
        text = query.casefold()
        needs = []
        for phrase, value in (
            ("addiction", "addiction_services"),
            ("substance use", "addiction_services"),
            ("hospital", "hospitals"),
            ("pharmacy", "pharmacy_services"),
            ("medication", "pharmacy_services"),
        ):
            if phrase in text and value not in needs:
                needs.append(value)

        modalities = tuple(value for phrase, value in (("walk-in", "walk-in"), ("walk in", "walk-in"), ("same-day", "same-day"), ("same day", "same-day"), ("in person", "in-person"), ("virtual", "virtual")) if phrase in text)
        populations = tuple(value for phrase, value in (("adolescent", "adolescent"), ("youth", "youth"), ("senior", "senior"), ("child", "child")) if phrase in text)
        if "no ohip" in text:
            coverage = ("no-ohip",)
        elif "ohip only" in text or "ohip" in text:
            coverage = ("ohip",)
        else:
            coverage = ()
        languages = tuple(value for phrase, value in (("mandarin", "Mandarin"), ("cantonese", "Cantonese"), ("french", "French"), ("ojibwe", "Ojibwe")) if phrase in text)
        urgency = "same-day" if "same-day" in text or "same day" in text else ""

        location = ""
        known_cities = sorted({resource.city for resource in self.store._seed.values()}, key=len, reverse=True)
        for city in known_cities:
            if city and city.casefold() in text:
                location = city
                break
        return SearchFilters(
            need=tuple(needs),
            modality=modalities,
            population=populations,
            coverage=coverage,
            location=location,
            language=languages,
            urgency=urgency,
        )

    @staticmethod
    def _tokens(query: str) -> set[str]:
        return {
            token for token in re.findall(r"[a-z0-9]+", query.casefold())
            if token not in STOP_WORDS and len(token) > 1
        }

    def search(
        self,
        query: str,
        limit: int = 20,
        now: datetime | None = None,
        filters: SearchFilters | None = None,
    ) -> list[dict[str, Any]]:
        now = now or datetime.now(timezone.utc)
        filters = filters or self.extract_filters(query)
        terms = self._tokens(query)
        projected = self.store.project(now=now)
        ranked = []
        for record in projected.values():
            haystack = " ".join(
                [record["name"], record["category"], record["city"], record["address"], record.get("postal_code", ""), record.get("domain", "")]
            ).casefold()
            term_fit = sum(term in haystack for term in terms) / len(terms) if terms else 0.0
            need_fit = 1.0 if not filters.need or any(need in record.get("tags", ()) for need in filters.need) else 0.0
            location_fit = 1.0 if not filters.location or record["city"].casefold() == filters.location.casefold() else 0.0
            language_values = {language.casefold() for language in record.get("languages", ())}
            language_fit = 1.0 if not filters.language or any(language.casefold() in language_values for language in filters.language) else 0.0
            fit = (term_fit + need_fit + location_fit + language_fit) / 4
            field_states = record.get("field_states", {})
            relevant_fields = ["name", "address", "city", "phone", "website"]
            custody = record.get("chain_of_custody", {})
            freshness_values = [
                field_states[field]
                for field in relevant_fields
                if custody.get(field)
            ]
            freshness = (
                sum({"verified-fresh": 1.0, "verified-aging": 0.5}.get(state, 0.0) for state in freshness_values)
                / len(freshness_values)
                if freshness_values
                else 0.0
            )
            flagged = len(record.get("flagged_fields", [])) / len(relevant_fields)
            rank_score = max(0.0, 0.6 * fit + 0.3 * freshness - 0.4 * flagged)
            result = dict(record)
            result.update({
                "fit_score": round(fit, 4),
                "trust_score": round(freshness, 4),
                "flag_penalty": round(flagged, 4),
                "rank_score": round(rank_score, 4),
                "rationale": self._rationale(filters, fit, freshness, flagged),
            })
            ranked.append(result)
        ranked.sort(key=lambda result: (-result["rank_score"], result["name"].casefold(), result["id"]))
        results = ranked[:limit]
        self.logs.append(
            SearchLog(
                query=query,
                filters=filters,
                returned_ids=tuple(result["id"] for result in results),
                scores=tuple(result["rank_score"] for result in results),
                created_at=now,
            )
        )
        return results

    @staticmethod
    def _rationale(filters: SearchFilters, fit: float, freshness: float, flagged: float) -> str:
        reasons = [f"fit {fit:.0%}", f"trust {freshness:.0%}"]
        if filters.location:
            reasons.append(f"location {filters.location}")
        if filters.language:
            reasons.append("language " + "/".join(filters.language))
        if flagged:
            reasons.append(f"{flagged:.0%} flagged fields")
        return " · ".join(reasons)


def search_log_dict(log: SearchLog) -> dict[str, Any]:
    result = asdict(log)
    result["filters"] = asdict(log.filters)
    result["created_at"] = log.created_at.isoformat()
    return result
