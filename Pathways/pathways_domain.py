"""Core Pathways domain model.

The module is deliberately framework-free so it can back the recovered browser
prototype now and a Flask/Postgres adapter later. Directory state is derived
from an append-only event ledger; no mutation is performed on resource records.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from math import log2
from collections import defaultdict
from typing import Any, Iterable
from uuid import uuid4


FIELD_STATES = {
    "verified-fresh",
    "verified-aging",
    "ai-only",
    "flagged-stale",
    "unknown",
}
DEFAULT_TRACKED_FIELDS = (
    "name",
    "address",
    "city",
    "phone",
    "website",
    "category",
)


@dataclass(frozen=True)
class Resource:
    id: str
    name: str
    category: str
    city: str
    domain: str = ""
    address: str = ""
    phone: str = ""
    website: str = ""
    postal_code: str = ""
    source_url: str = ""
    source_dataset: str = ""
    source_row_number: int = 0
    languages: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    fit_score: float = 0.0
    last_verified_at: datetime | None = None
    status: str = "active"


@dataclass(frozen=True)
class Event:
    id: str
    resource_id: str
    kind: str
    actor: str
    created_at: datetime
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Ask:
    id: str
    region: str
    author: str
    text: str
    tags: tuple[str, ...] = ()
    expires_at: datetime | None = None
    status: str = "open"
    linked_case_id: str | None = None
    watchers: tuple[str, ...] = ()
    created_at: datetime | None = None


@dataclass(frozen=True)
class AskReply:
    id: str
    ask_id: str
    author: str
    text: str
    attached_resource_id: str | None = None
    candidate_name: str | None = None
    created_at: datetime | None = None


@dataclass(frozen=True)
class FieldAsk:
    """Legacy field-level ask projected from Verify flags."""

    id: str
    resource_id: str
    field: str
    reason: str
    status: str = "open"
    created_at: datetime | None = None
    resolved_at: datetime | None = None


class PathwaysStore:
    """Append-only Pathways ledger with a materialized resource projection."""

    def __init__(
        self,
        resources: Iterable[Resource] = (),
        freshness_window_days: int = 30,
        aging_window_days: int = 90,
    ) -> None:
        if freshness_window_days <= 0 or aging_window_days <= freshness_window_days:
            raise ValueError("aging_window_days must be greater than freshness_window_days > 0")
        self._seed = {resource.id: resource for resource in resources}
        self.freshness_window_days = freshness_window_days
        self.aging_window_days = aging_window_days
        self.events: list[Event] = []
        self._asks: dict[str, Ask] = {}
        self._ask_replies: dict[str, list[AskReply]] = defaultdict(list)

    def append(
        self,
        resource_id: str,
        kind: str,
        actor: str,
        created_at: datetime | None = None,
        **payload: Any,
    ) -> Event:
        event = Event(
            id=str(uuid4()),
            resource_id=resource_id,
            kind=kind,
            actor=actor,
            created_at=created_at or datetime.now(timezone.utc),
            payload=payload,
        )
        self.events.append(event)
        return event

    def verify(
        self,
        resource_id: str,
        field: str,
        actor: str = "system",
        source: str = "unknown",
        display_actor: str | None = None,
        anonymous: bool = False,
        created_at: datetime | None = None,
    ) -> Event:
        return self.append(
            resource_id,
            "field_verified",
            actor,
            created_at=created_at,
            field=field,
            source=source,
            display_actor=display_actor or actor,
            anonymous=anonymous,
        )

    def flag(
        self,
        resource_id: str,
        field: str,
        reason: str,
        actor: str = "user",
        correction: str = "",
        source_url: str = "",
        source: str = "unknown",
        display_actor: str | None = None,
        anonymous: bool = False,
        created_at: datetime | None = None,
    ) -> Event:
        return self.append(
            resource_id,
            "field_flagged",
            actor,
            created_at=created_at,
            field=field,
            reason=reason,
            correction=correction,
            source_url=source_url,
            source=source,
            display_actor=display_actor or actor,
            anonymous=anonymous,
        )

    def accept_suggestion(
        self,
        resource_id: str,
        field: str,
        value: Any,
        source_url: str = "",
        actor: str = "operator",
        source: str = "web",
        display_actor: str | None = None,
        created_at: datetime | None = None,
    ) -> Event:
        return self.append(
            resource_id,
            "suggestion_accepted",
            actor,
            created_at=created_at,
            field=field,
            value=value,
            source_url=source_url,
            source=source,
            display_actor=display_actor or actor,
        )

    def create_ask(
        self,
        region: str,
        text: str,
        tags: Iterable[str] = (),
        author: str = "user",
        rural_extended: bool = False,
        linked_case_id: str | None = None,
        created_at: datetime | None = None,
    ) -> Ask:
        created_at = created_at or datetime.now(timezone.utc)
        expiry_days = 12 if rural_extended else 7
        ask = Ask(
            id=f"ask-{uuid4()}",
            region=region,
            author=author,
            text=text.strip(),
            tags=tuple(tags),
            expires_at=created_at + timedelta(days=expiry_days),
            linked_case_id=linked_case_id,
            created_at=created_at,
        )
        if not ask.text:
            raise ValueError("ask text cannot be empty")
        self._asks[ask.id] = ask
        self.append("", "ask_created", author, created_at=created_at, ask_id=ask.id, region=region, text=ask.text, tags=ask.tags)
        return ask

    def reply_to_ask(
        self,
        ask_id: str,
        text: str,
        author: str = "user",
        attached_resource_id: str | None = None,
        candidate_name: str | None = None,
        created_at: datetime | None = None,
    ) -> AskReply:
        ask = self._asks.get(ask_id)
        if ask is None:
            raise KeyError(f"unknown ask: {ask_id}")
        if ask.status != "open":
            raise ValueError("cannot reply to a closed or expired ask")
        if not text.strip():
            raise ValueError("reply text cannot be empty")
        reply = AskReply(
            id=f"reply-{uuid4()}",
            ask_id=ask_id,
            author=author,
            text=text.strip(),
            attached_resource_id=attached_resource_id,
            candidate_name=candidate_name,
            created_at=created_at or datetime.now(timezone.utc),
        )
        self._ask_replies[ask_id].append(reply)
        self.append(
            "",
            "ask_replied",
            author,
            created_at=reply.created_at,
            ask_id=ask_id,
            reply_id=reply.id,
            attached_resource_id=attached_resource_id,
            candidate_name=candidate_name,
        )
        return reply

    def watch_ask(self, ask_id: str, watcher: str) -> Ask:
        ask = self._asks[ask_id]
        if watcher in ask.watchers:
            return ask
        updated = Ask(**{**asdict(ask), "watchers": (*ask.watchers, watcher)})
        self._asks[ask_id] = updated
        self.append("", "ask_watched", watcher, ask_id=ask_id)
        return updated

    def resolve_ask(self, ask_id: str, actor: str = "operator", linked_case_id: str | None = None, created_at: datetime | None = None) -> Ask:
        ask = self._asks[ask_id]
        if ask.status != "open":
            return ask
        resolved = Ask(**{**asdict(ask), "status": "resolved", "linked_case_id": linked_case_id or ask.linked_case_id})
        self._asks[ask_id] = resolved
        self.append("", "ask_resolved", actor, created_at=created_at, ask_id=ask_id, linked_case_id=resolved.linked_case_id)
        return resolved

    def close_case(
        self,
        case_id: str,
        outcome: str,
        resource_id: str | None = None,
        referred_service_name: str | None = None,
        linked_ask_id: str | None = None,
        confirmed_fields: Iterable[str] = (),
        actor: str = "operator",
        created_at: datetime | None = None,
    ) -> dict[str, Any]:
        created_at = created_at or datetime.now(timezone.utc)
        confirmed_fields_list = list(dict.fromkeys(field for field in confirmed_fields if field))
        resolved_asks: list[str] = []

        if linked_ask_id is not None:
            ask = self._asks.get(linked_ask_id)
            if ask is not None and ask.status == "open":
                self.resolve_ask(linked_ask_id, actor=actor, linked_case_id=case_id, created_at=created_at)
                resolved_asks.append(linked_ask_id)

        if resource_id is not None:
            for field in confirmed_fields_list:
                self.verify(
                    resource_id,
                    field,
                    actor=actor,
                    source="case-close",
                    display_actor=actor,
                    created_at=created_at,
                )

        seeded_resource_id: str | None = None
        candidate_seeded = False
        if resource_id is None and referred_service_name:
            candidate_id = f"candidate-{uuid4()}"
            candidate = Resource(
                id=candidate_id,
                name=referred_service_name,
                category="candidate resource",
                city="Unknown",
                domain="candidate",
                address="",
                phone="",
                website="",
                status="candidate",
            )
            self._seed[candidate_id] = candidate
            self.append(
                candidate_id,
                "resource_seeded",
                actor,
                created_at=created_at,
                case_id=case_id,
                outcome=outcome,
                source="case-close",
                name=referred_service_name,
            )
            self.verify(
                candidate_id,
                "name",
                actor=actor,
                source="case-close",
                display_actor=actor,
                created_at=created_at,
            )
            for field in confirmed_fields_list:
                self.verify(
                    candidate_id,
                    field,
                    actor=actor,
                    source="case-close",
                    display_actor=actor,
                    created_at=created_at,
                )
            seeded_resource_id = candidate_id
            candidate_seeded = True

        summary = {
            "case_id": case_id,
            "outcome": outcome,
            "resource_id": resource_id,
            "referred_service_name": referred_service_name,
            "resolved_asks": resolved_asks,
            "confirmed_fields": confirmed_fields_list,
            "candidate_seeded": candidate_seeded,
            "seeded_resource_id": seeded_resource_id,
        }
        self.append(
            resource_id or "",
            "case_closed",
            actor,
            created_at=created_at,
            case_id=case_id,
            outcome=outcome,
            linked_resource_id=resource_id,
            referred_service_name=referred_service_name,
            linked_ask_id=linked_ask_id,
            resolved_ask_ids=resolved_asks,
            confirmed_fields=confirmed_fields_list,
            candidate_seeded=candidate_seeded,
            seeded_resource_id=seeded_resource_id,
        )
        return summary

    def asks_board(self, region: str | None = None, now: datetime | None = None) -> list[dict[str, Any]]:
        now = now or datetime.now(timezone.utc)
        board = []
        for ask in self._asks.values():
            current = ask
            if ask.status == "open" and ask.expires_at and ask.expires_at <= now:
                current = Ask(**{**asdict(ask), "status": "expired"})
                self._asks[ask.id] = current
            if region is not None and current.region != region:
                continue
            board.append({"ask": asdict(current), "replies": [asdict(reply) for reply in self._ask_replies.get(current.id, [])]})
        return board

    def _field_state(self, verified_at: datetime | None, flagged: bool, now: datetime) -> str:
        if flagged:
            return "flagged-stale"
        if verified_at is None:
            return "unknown"
        age_days = max(0.0, (now - verified_at).total_seconds() / 86400)
        if age_days <= self.freshness_window_days:
            return "verified-fresh"
        if age_days <= self.aging_window_days:
            return "verified-aging"
        return "unknown"

    def project(self, now: datetime | None = None) -> dict[str, dict[str, Any]]:
        now = now or datetime.now(timezone.utc)
        state = {resource_id: asdict(resource) for resource_id, resource in self._seed.items()}
        asks: dict[tuple[str, str], dict[str, Any]] = {}
        field_events: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        verified_at: dict[tuple[str, str], datetime] = {}
        flagged: set[tuple[str, str]] = set()
        for event in self.events:
            if event.resource_id not in state:
                continue
            record = state[event.resource_id]
            payload = event.payload
            field_name = payload.get("field")
            if field_name:
                field_events[(event.resource_id, field_name)].append(
                    {
                        "event_id": event.id,
                        "kind": event.kind,
                        "actor": event.actor,
                        "display_actor": "AOHT member" if payload.get("anonymous") else payload.get("display_actor", event.actor),
                        "anonymous": bool(payload.get("anonymous", False)),
                        "created_at": event.created_at,
                        "source": payload.get("source", "unknown"),
                        "source_url": payload.get("source_url", ""),
                        "reason": payload.get("reason", ""),
                        "correction": payload.get("correction", ""),
                    }
                )
            if event.kind == "field_verified":
                record.setdefault("verified_fields", set()).add(field_name)
                record["last_verified_at"] = event.created_at
                verified_at[(event.resource_id, field_name)] = event.created_at
                flagged.discard((event.resource_id, field_name))
            elif event.kind == "field_flagged":
                record.setdefault("flagged_fields", set()).add(field_name)
                flagged.add((event.resource_id, field_name))
                ask_key = (event.resource_id, field_name)
                asks[ask_key] = {
                    "id": f"ask-{event.id}",
                    "resource_id": event.resource_id,
                    "field": field_name,
                    "reason": payload.get("reason", ""),
                    "status": "open",
                    "created_at": event.created_at,
                }
            elif event.kind == "suggestion_accepted":
                if field_name:
                    record[field_name] = payload.get("value", "")
                    record.setdefault("verified_fields", set()).add(field_name)
                    record.setdefault("flagged_fields", set()).discard(field_name)
                    verified_at[(event.resource_id, field_name)] = event.created_at
                    flagged.discard((event.resource_id, field_name))
                record["last_verified_at"] = event.created_at
                ask_key = (event.resource_id, field_name)
                if ask_key in asks:
                    asks[ask_key]["status"] = "resolved"
                    asks[ask_key]["resolved_at"] = event.created_at
        for record in state.values():
            record["verified_fields"] = sorted(record.get("verified_fields", set()))
            record["flagged_fields"] = sorted(record.get("flagged_fields", set()))
            record["field_states"] = {
                field_name: self._field_state(
                    verified_at.get((record["id"], field_name)),
                    (record["id"], field_name) in flagged,
                    now,
                )
                for field_name in DEFAULT_TRACKED_FIELDS
            }
            record["chain_of_custody"] = {
                field_name: field_events.get((record["id"], field_name), [])
                for field_name in DEFAULT_TRACKED_FIELDS
            }
        self._last_asks = list(asks.values())
        return state

    def asks(self, status: str | None = None) -> list[dict[str, Any]]:
        self.project()
        items: list[dict[str, Any]] = []
        for ask in self._last_asks:
            if status is None or ask["status"] == status:
                items.append(ask)
        for ask in self._asks.values():
            if status is not None and ask.status != status:
                continue
            if any(existing.get("id") == ask.id for existing in items):
                continue
            payload = asdict(ask)
            payload.setdefault("field", None)
            items.append(payload)
        return items

    def search(self, query: str, limit: int = 20, now: datetime | None = None) -> list[dict[str, Any]]:
        now = now or datetime.now(timezone.utc)
        terms = {term.casefold() for term in query.split() if term.strip()}
        ranked: list[dict[str, Any]] = []
        for record in self.project().values():
            haystack = " ".join(
                [record["name"], record["category"], record["city"], record["address"]]
                + list(record.get("languages", ()))
                + list(record.get("tags", ()))
            ).casefold()
            matched = sum(term in haystack for term in terms)
            fit = matched / len(terms) if terms else 0.0
            verified_at = record.get("last_verified_at")
            if verified_at is None:
                freshness = 0.0
            else:
                age_days = max(0.0, (now - verified_at).total_seconds() / 86400)
                freshness = max(0.0, 1.0 - age_days / 90.0)
            result = dict(record)
            result["fit_score"] = round(fit, 4)
            result["trust_score"] = round(freshness, 4)
            result["rank_score"] = round(0.7 * fit + 0.3 * freshness, 4)
            ranked.append(result)
        ranked.sort(key=lambda item: item["rank_score"], reverse=True)
        return ranked[:limit]


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Compute binary Precision@K for an evaluated ranking."""
    if k <= 0:
        return 0.0
    top = retrieved[:k]
    return sum(item in relevant for item in top) / len(top) if top else 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Compute binary nDCG@K for a ranking."""
    top = retrieved[:k]
    if not top or not relevant:
        return 0.0
    dcg = sum((1.0 / log2(index + 2)) for index, item in enumerate(top) if item in relevant)
    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / log2(index + 2) for index in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def seed_resources() -> list[Resource]:
    """Small deterministic seed set for the recovered prototype and smoke tests."""
    return [
        Resource(
            id="res-yee-hong",
            name="Yee Hong Centre - Scarborough",
            category="PSW and home care",
            city="Scarborough",
            address="5 Crown Princess Crescent",
            phone="416-321-3000",
            website="https://www.yeehong.com",
            languages=("Mandarin", "Cantonese", "English"),
            tags=("OHIP", "meal support", "mobility"),
            fit_score=0.96,
        ),
        Resource(
            id="res-spectrum",
            name="Spectrum Home Health",
            category="PSW and home care",
            city="Scarborough",
            languages=("English", "Mandarin"),
            tags=("private", "mobility"),
            fit_score=0.88,
        ),
        Resource(
            id="res-meals",
            name="Mandarin Meals on Wheels",
            category="meal support",
            city="Scarborough",
            languages=("Mandarin", "English"),
            tags=("sliding scale", "meal support"),
            fit_score=0.91,
        ),
    ]
