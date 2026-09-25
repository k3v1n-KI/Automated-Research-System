"""Deterministic Find search over the normalized Pathways corpus."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:  # pragma: no cover - optional semantic stage dependency
    TfidfVectorizer = None
    cosine_similarity = None

from pathways_domain import PathwaysStore, Resource


STOP_WORDS = {
    "a", "an", "and", "any", "for", "find", "in", "me", "near", "of",
    "service", "services", "the", "to", "with",
}

RESIDUAL_STOP_WORDS = STOP_WORDS | {
    "need", "needs", "help", "looking", "find", "where", "can", "get",
    "someone", "something", "support", "care", "service", "services",
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


HYBRID_ALIASES = {
    "need": {
        "addiction_services": ("addiction", "substance use", "drug use"),
        "pharmacy_services": ("pharmacy", "medication", "prescriptions"),
        "hospitals": ("hospital",),
    },
    "modality": {
        "walk-in": ("walk-in", "walk in", "no appointment"),
        "same-day": ("same-day", "same day", "today", "right away"),
        "in-person": ("in person", "in-person", "face to face"),
        "virtual": ("virtual", "online", "remote", "by video"),
    },
    "population": {
        "adolescent": ("adolescent",),
        "youth": ("youth", "teenager", "teen", "young person", "young people"),
        "senior": ("senior", "older adult", "older person", "elderly", "my parent"),
        "child": ("child", "children", "my kid"),
    },
    "coverage": {
        "no-ohip": ("no ohip", "without ohip", "without insurance", "uninsured"),
        "ohip": ("ohip",),
    },
    "language": {
        "Mandarin": ("mandarin",),
        "Cantonese": ("cantonese",),
        "French": ("french", "francophone"),
        "Ojibwe": ("ojibwe",),
    },
}

SEMANTIC_ALIASES = {
    "need": {
        "addiction_services": ("substance abuse", "drug problem", "dependency support"),
        "pharmacy_services": ("meds", "medications", "drug store"),
    },
    "modality": {
        "same-day": ("urgent", "as soon as possible", "immediately"),
        "in-person": ("on site", "onsite", "in the office"),
        "virtual": ("remote consultation", "telehealth", "from home"),
    },
    "population": {
        "adolescent": ("young teenager", "minor teenager"),
        "senior": ("elder", "older person"),
        "youth": ("young person", "young people"),
    },
    "coverage": {
        "no-ohip": ("no health card", "no medical coverage"),
    },
    "language": {
        "French": ("French-speaking", "francophone"),
        "Ojibwe": ("Anishinaabemowin",),
    },
}


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

    def extract_filters_hybrid(self, query: str) -> SearchFilters:
        """Extract filters through a versioned alias ontology.

        This deterministic hybrid layer expands recognized concepts without
        allowing a language model to invent filter values or control ranking.
        An LLM can be added later as a proposal source for unresolved phrases.
        """
        text = query.casefold()

        def matches(field: str) -> tuple[str, ...]:
            return tuple(
                value
                for value, phrases in HYBRID_ALIASES[field].items()
                if any(phrase in text for phrase in phrases)
            )

        coverage = matches("coverage")
        if "no-ohip" in coverage:
            coverage = ("no-ohip",)
        locations = sorted(
            {resource.city for resource in self.store._seed.values()},
            key=len,
            reverse=True,
        )
        location = next((city for city in locations if city and city.casefold() in text), "")
        urgency = "same-day" if "same-day" in matches("modality") else ""
        return SearchFilters(
            need=matches("need"),
            modality=matches("modality"),
            population=matches("population"),
            coverage=coverage,
            location=location,
            language=matches("language"),
            urgency=urgency,
        )

    def _semantic_matches(self, query: str, matched_aliases: list[dict[str, str]]) -> list[dict[str, Any]]:
        """Rank unresolved query n-grams against a small ontology phrase index."""
        if TfidfVectorizer is None or cosine_similarity is None:
            return []
        text = query.casefold()
        tokens = re.findall(r"[a-z0-9-]+", text)
        query_phrases = {
            " ".join(tokens[start:end])
            for start in range(len(tokens))
            for end in range(start + 1, min(len(tokens), start + 4) + 1)
        }
        known_phrases = {entry["phrase"] for entry in matched_aliases}
        candidates = []
        for field, concepts in SEMANTIC_ALIASES.items():
            for concept, phrases in concepts.items():
                for phrase in phrases:
                    candidates.append((field, concept, phrase))
        if not candidates:
            return []
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        matrix = vectorizer.fit_transform([phrase for _, _, phrase in candidates] + list(query_phrases))
        matches = []
        for index, (field, concept, phrase) in enumerate(candidates):
            eligible = [
                (
                    float(cosine_similarity(matrix[index], matrix[len(candidates) + query_index])[0][0]),
                    query_phrase,
                )
                for query_index, query_phrase in enumerate(query_phrases)
                if query_phrase not in known_phrases
                and not (len(phrase.split()) > 1 and len(query_phrase.split()) == 1)
            ]
            best = max(eligible) if eligible else (0.0, "")
            if best[0] >= 0.58:
                matches.append({
                    "field": field,
                    "concept": concept,
                    "phrase": best[1],
                    "ontology_phrase": phrase,
                    "score": round(best[0], 4),
                })
        matches.sort(key=lambda item: (-item["score"], item["field"], item["concept"]))
        return matches

    def extract_filters_hybrid_trace(
        self,
        query: str,
        llm_proposer: Any | None = None,
        semantic_enabled: bool = True,
        aliases_enabled: bool = True,
    ) -> dict[str, Any]:
        """Return hybrid filters plus semantic and optional LLM evidence."""
        filters = self.extract_filters_hybrid(query) if aliases_enabled else self.extract_filters(query)
        text = query.casefold()
        matched_aliases = []
        if aliases_enabled:
            for field, concepts in HYBRID_ALIASES.items():
                for concept, phrases in concepts.items():
                    for phrase in phrases:
                        if phrase in text:
                            matched_aliases.append({"field": field, "phrase": phrase, "concept": concept})
        recognized_terms = {entry["phrase"] for entry in matched_aliases}
        known_cities = {
            resource.city.casefold()
            for resource in self.store._seed.values()
            if resource.city
        }
        residual_terms = [
            token for token in re.findall(r"[a-z0-9]+", text)
            if len(token) > 2
            and token not in RESIDUAL_STOP_WORDS
            and token not in recognized_terms
            and token not in known_cities
        ]
        semantic_matches = self._semantic_matches(query, matched_aliases) if semantic_enabled else []
        semantic_values = {field: [] for field in HYBRID_ALIASES}
        for match in semantic_matches:
            if match["concept"] not in semantic_values[match["field"]]:
                semantic_values[match["field"]].append(match["concept"])
        semantic_filters = SearchFilters(
            need=tuple(semantic_values["need"]),
            modality=tuple(semantic_values["modality"]),
            population=tuple(semantic_values["population"]),
            coverage=tuple(semantic_values["coverage"]),
            location=filters.location,
            language=tuple(semantic_values["language"]),
            urgency="same-day" if "same-day" in semantic_values["modality"] else filters.urgency,
        )
        llm_suggestion = None
        llm_provenance = None
        ambiguous_terms = {"teen", "teenager", "young person", "older adult"}
        has_ambiguous_phrase = any(term in text for term in ambiguous_terms)
        if llm_proposer is not None and (residual_terms or has_ambiguous_phrase):
            prompt = self._llm_filter_prompt(query, residual_terms)
            raw_suggestion = llm_proposer(prompt)
            llm_suggestion = self._validate_llm_suggestion(raw_suggestion)
            llm_provenance = {"prompt": prompt, "raw_response": raw_suggestion}
        combined = self._merge_filters(filters, semantic_filters, llm_suggestion or {})
        route_parts = []
        if aliases_enabled:
            route_parts.append("ontology_aliases")
        else:
            route_parts.append("baseline_rules")
        if semantic_matches:
            route_parts.append("semantic")
        if llm_suggestion:
            route_parts.append("llm")
        return {
            "filters": combined,
            "matched_aliases": matched_aliases,
            "unresolved_terms": sorted(set(residual_terms)),
            "semantic_matches": semantic_matches,
            "semantic_filters": asdict(semantic_filters),
            "llm_involved": llm_proposer is not None and bool(residual_terms),
            "llm_suggestion": llm_suggestion,
            "llm_provenance": llm_provenance,
            "extraction_route": "+".join(route_parts),
        }

    @staticmethod
    def _llm_filter_prompt(query: str, residual_terms: list[str]) -> str:
        allowed = {field: sorted(values) for field, values in HYBRID_ALIASES.items()}
        return json.dumps({
            "task": "Suggest only supported Pathways filter values for unresolved language.",
            "query": query,
            "unresolved_terms": residual_terms,
            "allowed_values": allowed,
            "instruction": "Return JSON with field arrays/scalars or null; abstain when unsupported.",
        }, sort_keys=True)

    @staticmethod
    def _validate_llm_suggestion(suggestion: Any) -> dict[str, Any] | None:
        if not isinstance(suggestion, dict):
            return None
        allowed = {field: set(values) for field, values in HYBRID_ALIASES.items()}
        validated = {}
        for field in HYBRID_ALIASES:
            value = suggestion.get(field)
            if isinstance(value, str):
                value = [value]
            if isinstance(value, list):
                valid = [item for item in value if item in allowed[field]]
                if valid:
                    validated[field] = valid
        urgency = suggestion.get("urgency")
        if urgency == "same-day":
            validated["urgency"] = urgency
        return validated or None

    @staticmethod
    def _merge_filters(base: SearchFilters, semantic: SearchFilters, suggestion: dict[str, Any]) -> SearchFilters:
        values = {}
        for field in HYBRID_ALIASES:
            existing = list(getattr(base, field)) if field != "location" else getattr(base, field)
            semantic_value = list(getattr(semantic, field)) if field != "location" else ""
            proposed = suggestion.get(field, [])
            if field == "location":
                values[field] = existing or proposed
            else:
                if proposed:
                    values[field] = tuple(dict.fromkeys(proposed))
                else:
                    values[field] = tuple(dict.fromkeys(existing or semantic_value))
        values["location"] = base.location or suggestion.get("location", "")
        values["urgency"] = base.urgency or semantic.urgency or suggestion.get("urgency", "")
        return SearchFilters(**values)

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
                [
                    record["name"],
                    record["category"],
                    record["city"],
                    record["address"],
                    record.get("postal_code", ""),
                    record.get("domain", ""),
                    *record.get("languages", ()),
                    *record.get("tags", ()),
                ]
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
