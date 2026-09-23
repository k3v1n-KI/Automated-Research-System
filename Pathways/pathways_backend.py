import json
import hashlib
import os
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import psycopg
from flask import Flask, jsonify, request

DEFAULT_DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/postgres"


class PathwaysDatabase:
    """Minimal PostgreSQL-backed implementation of the Pathways event ledger."""

    def __init__(self, database_url: str | None = None, table_prefix: str = "pathways") -> None:
        self.database_url = database_url or os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
        if len(table_prefix) > 40:
            suffix = hashlib.sha1(table_prefix.encode("utf-8")).hexdigest()[:12]
            table_prefix = f"{table_prefix[:27]}_{suffix}"
        self.table_prefix = table_prefix
        self.resources_table = f"{table_prefix}_resources"
        self.events_table = f"{table_prefix}_events"
        self.asks_table = f"{table_prefix}_asks"
        self.ask_replies_table = f"{table_prefix}_ask_replies"
        self.cases_table = f"{table_prefix}_cases"
        self.research_jobs_table = f"{table_prefix}_research_jobs"
        self.research_suggestions_table = f"{table_prefix}_research_suggestions"
        self.init_schema()

    def _connect(self):
        return psycopg.connect(self.database_url, autocommit=False)

    def init_schema(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute('CREATE EXTENSION IF NOT EXISTS "pgcrypto";')
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.resources_table} (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        category TEXT DEFAULT '',
                        city TEXT DEFAULT '',
                        address TEXT DEFAULT '',
                        phone TEXT DEFAULT '',
                        website TEXT DEFAULT '',
                        languages JSONB DEFAULT '[]'::jsonb,
                        tags JSONB DEFAULT '[]'::jsonb,
                        fit_score NUMERIC(5,2) DEFAULT 0,
                        source_url TEXT DEFAULT '',
                        source_dataset TEXT DEFAULT '',
                        source_row_number INTEGER DEFAULT 0,
                        status TEXT DEFAULT 'active',
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    """
                )
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.events_table} (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        resource_id TEXT,
                        kind TEXT NOT NULL,
                        actor TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        payload JSONB NOT NULL DEFAULT '{{}}'::jsonb
                    );
                    """
                )
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.asks_table} (
                        id TEXT PRIMARY KEY,
                        region TEXT NOT NULL,
                        author TEXT NOT NULL,
                        text TEXT NOT NULL,
                        tags JSONB DEFAULT '[]'::jsonb,
                        expires_at TIMESTAMPTZ,
                        status TEXT NOT NULL DEFAULT 'open',
                        linked_case_id TEXT,
                        watchers JSONB DEFAULT '[]'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    """
                )
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.ask_replies_table} (
                        id TEXT PRIMARY KEY,
                        ask_id TEXT NOT NULL,
                        author TEXT NOT NULL,
                        text TEXT NOT NULL,
                        attached_resource_id TEXT,
                        candidate_name TEXT,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    """
                )
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.cases_table} (
                        id TEXT PRIMARY KEY,
                        outcome TEXT NOT NULL,
                        resource_id TEXT,
                        referred_service_name TEXT,
                        linked_ask_id TEXT,
                        note TEXT DEFAULT '',
                        confirmed_fields JSONB DEFAULT '[]'::jsonb,
                        actor TEXT NOT NULL,
                        candidate_seeded BOOLEAN NOT NULL DEFAULT FALSE,
                        seeded_resource_id TEXT,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    """
                )
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.research_jobs_table} (
                        id TEXT PRIMARY KEY,
                        trigger_type TEXT NOT NULL,
                        resource_id TEXT,
                        ask_id TEXT,
                        field TEXT,
                        prompt TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'queued',
                        attempts INTEGER NOT NULL DEFAULT 0,
                        error TEXT DEFAULT '',
                        requested_by TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    """
                )
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.research_suggestions_table} (
                        id TEXT PRIMARY KEY,
                        job_id TEXT NOT NULL,
                        resource_id TEXT,
                        field TEXT,
                        value TEXT NOT NULL,
                        confidence NUMERIC(5,4),
                        source_url TEXT NOT NULL,
                        evidence TEXT DEFAULT '',
                        status TEXT NOT NULL DEFAULT 'proposed',
                        reviewed_by TEXT,
                        reviewed_at TIMESTAMPTZ,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    """
                )
            conn.commit()

    def upsert_resource(self, resource: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "id": resource["id"],
            "name": resource.get("name", ""),
            "category": resource.get("category", ""),
            "city": resource.get("city", ""),
            "address": resource.get("address", ""),
            "phone": resource.get("phone", ""),
            "website": resource.get("website", ""),
            "languages": resource.get("languages", []) or [],
            "tags": resource.get("tags", []) or [],
            "fit_score": float(resource.get("fit_score", 0.0) or 0.0),
            "source_url": resource.get("source_url", ""),
            "source_dataset": resource.get("source_dataset", ""),
            "source_row_number": int(resource.get("source_row_number", 0) or 0),
            "status": resource.get("status", "active"),
        }
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {self.resources_table} (
                        id, name, category, city, address, phone, website,
                        languages, tags, fit_score, source_url, source_dataset,
                        source_row_number, status, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (id)
                    DO UPDATE SET
                        name = EXCLUDED.name,
                        category = EXCLUDED.category,
                        city = EXCLUDED.city,
                        address = EXCLUDED.address,
                        phone = EXCLUDED.phone,
                        website = EXCLUDED.website,
                        languages = EXCLUDED.languages,
                        tags = EXCLUDED.tags,
                        fit_score = EXCLUDED.fit_score,
                        source_url = EXCLUDED.source_url,
                        source_dataset = EXCLUDED.source_dataset,
                        source_row_number = EXCLUDED.source_row_number,
                        status = EXCLUDED.status,
                        updated_at = NOW()
                    RETURNING *;
                    """,
                    (
                        payload["id"],
                        payload["name"],
                        payload["category"],
                        payload["city"],
                        payload["address"],
                        payload["phone"],
                        payload["website"],
                        json.dumps(payload["languages"]),
                        json.dumps(payload["tags"]),
                        payload["fit_score"],
                        payload["source_url"],
                        payload["source_dataset"],
                        payload["source_row_number"],
                        payload["status"],
                    ),
                )
                row = cur.fetchone()
                conn.commit()
        return self._row_to_resource(row)

    def list_resources(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT * FROM {self.resources_table} ORDER BY name")
                return [self._row_to_resource(row) for row in cur.fetchall()]

    def get_resource(self, resource_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT * FROM {self.resources_table} WHERE id = %s", (resource_id,))
                row = cur.fetchone()
                return self._row_to_resource(row) if row else None

    def append_event(
        self,
        resource_id: str | None,
        kind: str,
        actor: str,
        payload: dict[str, Any] | None = None,
        created_at: datetime | None = None,
    ) -> dict[str, Any]:
        payload = payload or {}
        created_at = created_at or datetime.now(timezone.utc)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {self.events_table} (resource_id, kind, actor, created_at, payload)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING *;
                    """,
                    (resource_id, kind, actor, created_at, json.dumps(payload)),
                )
                row = cur.fetchone()
                conn.commit()
        return self._row_to_event(row)

    def list_events(self, limit: int | None = None) -> list[dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                if limit is None:
                    cur.execute(f"SELECT * FROM {self.events_table} ORDER BY created_at DESC")
                else:
                    cur.execute(
                        f"SELECT * FROM {self.events_table} ORDER BY created_at DESC LIMIT %s",
                        (limit,),
                    )
                return [self._row_to_event(row) for row in cur.fetchall()]

    def create_research_job(
        self,
        trigger_type: str,
        prompt: str,
        requested_by: str,
        resource_id: str | None = None,
        ask_id: str | None = None,
        field: str | None = None,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        if trigger_type not in {"missing_field", "ask"}:
            raise ValueError("trigger_type must be missing_field or ask")
        if not prompt.strip():
            raise ValueError("research prompt cannot be empty")
        if trigger_type == "missing_field" and (not resource_id or not field):
            raise ValueError("missing_field jobs require resource_id and field")
        if trigger_type == "ask" and not ask_id:
            raise ValueError("ask jobs require ask_id")
        job_id = job_id or f"research-{uuid4()}"
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {self.research_jobs_table} (
                        id, trigger_type, resource_id, ask_id, field, prompt, requested_by
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING *;
                    """,
                    (job_id, trigger_type, resource_id, ask_id, field, prompt, requested_by),
                )
                row = cur.fetchone()
                conn.commit()
        return self._row_to_research_job(row)

    def list_research_jobs(self, status: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                if status:
                    cur.execute(
                        f"SELECT * FROM {self.research_jobs_table} WHERE status = %s ORDER BY created_at DESC",
                        (status,),
                    )
                else:
                    cur.execute(f"SELECT * FROM {self.research_jobs_table} ORDER BY created_at DESC")
                return [self._row_to_research_job(row) for row in cur.fetchall()]

    def claim_research_job(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    UPDATE {self.research_jobs_table}
                    SET status = 'running', attempts = attempts + 1, updated_at = NOW()
                    WHERE id = %s AND status = 'queued'
                    RETURNING *;
                    """,
                    (job_id,),
                )
                row = cur.fetchone()
                conn.commit()
        return self._row_to_research_job(row) if row else None

    def fail_research_job(self, job_id: str, error: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    UPDATE {self.research_jobs_table}
                    SET status = 'failed', error = %s, updated_at = NOW()
                    WHERE id = %s AND status = 'running'
                    RETURNING *;
                    """,
                    (error[:2000], job_id),
                )
                row = cur.fetchone()
                conn.commit()
        return self._row_to_research_job(row) if row else None

    def add_research_suggestion(
        self,
        job_id: str,
        value: str,
        source_url: str,
        evidence: str = "",
        confidence: float | None = None,
        resource_id: str | None = None,
        field: str | None = None,
        suggestion_id: str | None = None,
    ) -> dict[str, Any]:
        if not value.strip() or not source_url.strip():
            raise ValueError("suggestions require a value and source_url")
        suggestion_id = suggestion_id or f"suggestion-{uuid4()}"
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {self.research_suggestions_table} (
                        id, job_id, resource_id, field, value, confidence, source_url, evidence
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING *;
                    """,
                    (suggestion_id, job_id, resource_id, field, value, confidence, source_url, evidence),
                )
                row = cur.fetchone()
                cur.execute(
                    f"UPDATE {self.research_jobs_table} SET status = 'needs_review', updated_at = NOW() WHERE id = %s",
                    (job_id,),
                )
                conn.commit()
        return self._row_to_research_suggestion(row)

    def list_research_suggestions(self, job_id: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                if job_id:
                    cur.execute(
                        f"SELECT * FROM {self.research_suggestions_table} WHERE job_id = %s ORDER BY created_at DESC",
                        (job_id,),
                    )
                else:
                    cur.execute(f"SELECT * FROM {self.research_suggestions_table} ORDER BY created_at DESC")
                return [self._row_to_research_suggestion(row) for row in cur.fetchall()]

    def review_research_suggestion(self, suggestion_id: str, status: str, reviewer: str) -> dict[str, Any] | None:
        if status not in {"accepted", "rejected"}:
            raise ValueError("suggestion status must be accepted or rejected")
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    UPDATE {self.research_suggestions_table}
                    SET status = %s, reviewed_by = %s, reviewed_at = NOW()
                    WHERE id = %s AND status = 'proposed'
                    RETURNING *;
                    """,
                    (status, reviewer, suggestion_id),
                )
                row = cur.fetchone()
                if row is None:
                    conn.commit()
                    return None
                if status == "accepted":
                    candidate_resource_id = row[2]
                    if candidate_resource_id is None and row[3] is None:
                        candidate_resource_id = f"candidate-{uuid4()}"
                        cur.execute(
                            f"""
                            INSERT INTO {self.resources_table} (
                                id, name, category, city, website, tags, source_url, source_dataset, status
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'candidate')
                            ON CONFLICT (id) DO NOTHING;
                            """,
                            (
                                candidate_resource_id,
                                row[4],
                                "candidate resource",
                                "Unknown",
                                row[6],
                                json.dumps(["candidate", "ask-research"]),
                                row[6],
                                "ars-ask",
                            ),
                        )
                        cur.execute(
                            f"UPDATE {self.research_suggestions_table} SET resource_id = %s WHERE id = %s",
                            (candidate_resource_id, suggestion_id),
                        )
                        self._append_event_in_cursor(
                            cur,
                            candidate_resource_id,
                            "resource_seeded",
                            reviewer,
                            {"suggestion_id": suggestion_id, "source_url": row[6], "source": "ask-research"},
                        )
                    self._append_event_in_cursor(
                        cur,
                        candidate_resource_id,
                        "suggestion_accepted",
                        reviewer,
                        {"field": row[3], "value": row[4], "source_url": row[6], "suggestion_id": row[0]},
                    )
                cur.execute(
                    f"UPDATE {self.research_jobs_table} SET status = 'completed', updated_at = NOW() WHERE id = %s",
                    (row[1],),
                )
                cur.execute(f"SELECT * FROM {self.research_suggestions_table} WHERE id = %s", (suggestion_id,))
                row = cur.fetchone()
                conn.commit()
        return self._row_to_research_suggestion(row)

    def _append_event_in_cursor(self, cur: Any, resource_id: str | None, kind: str, actor: str, payload: dict[str, Any]) -> None:
        cur.execute(
            f"INSERT INTO {self.events_table} (resource_id, kind, actor, payload) VALUES (%s, %s, %s, %s)",
            (resource_id, kind, actor, json.dumps(payload)),
        )

    @staticmethod
    def _row_to_research_job(row: Any) -> dict[str, Any]:
        return {
            "id": row[0],
            "trigger_type": row[1],
            "resource_id": row[2],
            "ask_id": row[3],
            "field": row[4],
            "prompt": row[5],
            "status": row[6],
            "attempts": row[7],
            "error": row[8],
            "requested_by": row[9],
            "created_at": row[10].isoformat() if row[10] else None,
            "updated_at": row[11].isoformat() if row[11] else None,
        }

    @staticmethod
    def _row_to_research_suggestion(row: Any) -> dict[str, Any]:
        return {
            "id": row[0],
            "job_id": row[1],
            "resource_id": row[2],
            "field": row[3],
            "value": row[4],
            "confidence": float(row[5]) if row[5] is not None else None,
            "source_url": row[6],
            "evidence": row[7],
            "status": row[8],
            "reviewed_by": row[9],
            "reviewed_at": row[10].isoformat() if row[10] else None,
            "created_at": row[11].isoformat() if row[11] else None,
        }

    def create_ask(
        self,
        region: str,
        text: str,
        author: str,
        tags: list[str] | tuple[str, ...] | None = None,
        expires_at: datetime | None = None,
        linked_case_id: str | None = None,
        ask_id: str | None = None,
    ) -> dict[str, Any]:
        if not text.strip():
            raise ValueError("ask text cannot be empty")
        ask_id = ask_id or f"ask-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
        expires_at = expires_at or (datetime.now(timezone.utc).replace(microsecond=0) + __import__('datetime').timedelta(days=7))
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {self.asks_table} (
                        id, region, author, text, tags, expires_at, status, linked_case_id, watchers, created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    RETURNING *;
                    """,
                    (
                        ask_id,
                        region,
                        author,
                        text,
                        json.dumps(list(tags or [])),
                        expires_at,
                        "open",
                        linked_case_id,
                        json.dumps([author]),
                    ),
                )
                row = cur.fetchone()
                conn.commit()
        return self._row_to_ask(row)

    def list_asks(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT * FROM {self.asks_table} ORDER BY created_at DESC")
                return [self._row_to_ask(row) for row in cur.fetchall()]

    def watch_ask(self, ask_id: str, watcher: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    UPDATE {self.asks_table}
                    SET watchers = CASE
                        WHEN watchers ? %s::text THEN watchers
                        ELSE watchers || jsonb_build_array(%s::text)
                    END
                    WHERE id = %s
                    RETURNING *;
                    """,
                    (watcher, watcher, ask_id),
                )
                row = cur.fetchone()
                conn.commit()
        return self._row_to_ask(row) if row else None

    def add_reply(
        self,
        ask_id: str,
        author: str,
        text: str,
        attached_resource_id: str | None = None,
        candidate_name: str | None = None,
    ) -> dict[str, Any]:
        reply_id = f"reply-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {self.ask_replies_table} (
                        id, ask_id, author, text, attached_resource_id, candidate_name, created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, NOW())
                    RETURNING *;
                    """,
                    (reply_id, ask_id, author, text, attached_resource_id, candidate_name),
                )
                row = cur.fetchone()
                conn.commit()
        return self._row_to_reply(row)

    def list_replies(self, ask_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT * FROM {self.ask_replies_table} WHERE ask_id = %s ORDER BY created_at ASC",
                    (ask_id,),
                )
                return [self._row_to_reply(row) for row in cur.fetchall()]

    def resolve_ask(self, ask_id: str, linked_case_id: str | None = None) -> dict[str, Any] | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    UPDATE {self.asks_table}
                    SET status = 'resolved', linked_case_id = %s
                    WHERE id = %s AND status = 'open'
                    RETURNING *;
                    """,
                    (linked_case_id, ask_id),
                )
                row = cur.fetchone()
                conn.commit()
        return self._row_to_ask(row) if row else None

    def close_case(
        self,
        case_id: str,
        outcome: str,
        actor: str,
        resource_id: str | None = None,
        referred_service_name: str | None = None,
        linked_ask_id: str | None = None,
        confirmed_fields: list[str] | tuple[str, ...] | None = None,
        note: str = "",
    ) -> dict[str, Any]:
        confirmed_fields = list(dict.fromkeys(confirmed_fields or []))
        candidate_seeded = False
        seeded_resource_id = None

        if resource_id is not None:
            for field in confirmed_fields:
                self.append_event(resource_id, "field_verified", actor, {"field": field, "source": "case-close"})
        elif referred_service_name:
            candidate_id = f"candidate-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
            self.upsert_resource(
                {
                    "id": candidate_id,
                    "name": referred_service_name,
                    "category": "candidate resource",
                    "city": "Unknown",
                    "address": "",
                    "phone": "",
                    "website": "",
                    "languages": [],
                    "tags": ["candidate"],
                    "fit_score": 0.0,
                    "source_dataset": "case-close",
                    "status": "candidate",
                }
            )
            for field in confirmed_fields:
                self.append_event(candidate_id, "field_verified", actor, {"field": field, "source": "case-close"})
            candidate_seeded = True
            seeded_resource_id = candidate_id

        if linked_ask_id:
            self.resolve_ask(linked_ask_id, linked_case_id=case_id)

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {self.cases_table} (
                        id, outcome, resource_id, referred_service_name,
                        linked_ask_id, note, confirmed_fields, actor,
                        candidate_seeded, seeded_resource_id, created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (id)
                    DO UPDATE SET
                        outcome = EXCLUDED.outcome,
                        resource_id = EXCLUDED.resource_id,
                        referred_service_name = EXCLUDED.referred_service_name,
                        linked_ask_id = EXCLUDED.linked_ask_id,
                        note = EXCLUDED.note,
                        confirmed_fields = EXCLUDED.confirmed_fields,
                        actor = EXCLUDED.actor,
                        candidate_seeded = EXCLUDED.candidate_seeded,
                        seeded_resource_id = EXCLUDED.seeded_resource_id
                    RETURNING *;
                    """,
                    (
                        case_id,
                        outcome,
                        resource_id,
                        referred_service_name,
                        linked_ask_id,
                        note,
                        json.dumps(confirmed_fields),
                        actor,
                        candidate_seeded,
                        seeded_resource_id,
                    ),
                )
                row = cur.fetchone()
                conn.commit()
        summary = self._row_to_case(row)
        self.append_event(resource_id, "case_closed", actor, {"case_id": case_id, "outcome": outcome, "linked_ask_id": linked_ask_id, "confirmed_fields": confirmed_fields})
        return summary

    def list_cases(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT * FROM {self.cases_table} ORDER BY created_at DESC")
                return [self._row_to_case(row) for row in cur.fetchall()]

    @staticmethod
    def _row_to_resource(row: Any) -> dict[str, Any]:
        if row is None:
            return {}
        return {
            "id": row[0],
            "name": row[1],
            "category": row[2],
            "city": row[3],
            "address": row[4],
            "phone": row[5],
            "website": row[6],
            "languages": row[7] or [],
            "tags": row[8] or [],
            "fit_score": float(row[9] or 0.0),
            "source_url": row[10],
            "source_dataset": row[11],
            "source_row_number": row[12],
            "status": row[13],
            "created_at": row[14].isoformat() if row[14] else None,
            "updated_at": row[15].isoformat() if row[15] else None,
        }

    @staticmethod
    def _row_to_event(row: Any) -> dict[str, Any]:
        if row is None:
            return {}
        return {
            "id": str(row[0]),
            "resource_id": row[1],
            "kind": row[2],
            "actor": row[3],
            "created_at": row[4].isoformat() if row[4] else None,
            "payload": row[5] or {},
        }

    @staticmethod
    def _row_to_ask(row: Any) -> dict[str, Any]:
        if row is None:
            return {}
        return {
            "id": row[0],
            "region": row[1],
            "author": row[2],
            "text": row[3],
            "tags": row[4] or [],
            "expires_at": row[5].isoformat() if row[5] else None,
            "status": row[6],
            "linked_case_id": row[7],
            "watchers": row[8] or [],
            "created_at": row[9].isoformat() if row[9] else None,
        }

    @staticmethod
    def _row_to_reply(row: Any) -> dict[str, Any]:
        if row is None:
            return {}
        return {
            "id": row[0],
            "ask_id": row[1],
            "author": row[2],
            "text": row[3],
            "attached_resource_id": row[4],
            "candidate_name": row[5],
            "created_at": row[6].isoformat() if row[6] else None,
        }

    @staticmethod
    def _row_to_case(row: Any) -> dict[str, Any]:
        if row is None:
            return {}
        return {
            "id": row[0],
            "outcome": row[1],
            "resource_id": row[2],
            "referred_service_name": row[3],
            "linked_ask_id": row[4],
            "note": row[5],
            "confirmed_fields": row[6] or [],
            "actor": row[7],
            "candidate_seeded": row[8],
            "seeded_resource_id": row[9],
            "created_at": row[10].isoformat() if row[10] else None,
        }


def create_app(database_url: str | None = None, table_prefix: str = "pathways") -> Flask:
    app = Flask(__name__)
    db = PathwaysDatabase(database_url=database_url, table_prefix=table_prefix)

    @app.after_request
    def add_dev_cors(response):
        response.headers["Access-Control-Allow-Origin"] = os.getenv("PATHWAYS_WEB_ORIGIN", "http://localhost:8765")
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return response

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "database": "postgres"})

    @app.get("/api/resources")
    def list_resources():
        return jsonify(db.list_resources())

    @app.post("/api/resources")
    def create_resource():
        payload = request.get_json(silent=True) or {}
        return jsonify(db.upsert_resource(payload)), 201

    @app.get("/api/events")
    def list_events():
        limit = request.args.get("limit", type=int)
        return jsonify(db.list_events(limit=limit))

    @app.post("/api/events")
    def append_event():
        payload = request.get_json(silent=True) or {}
        event = db.append_event(
            resource_id=payload.get("resource_id"),
            kind=payload["kind"],
            actor=payload.get("actor", "system"),
            payload=payload.get("payload", {}),
        )
        return jsonify(event), 201

    @app.get("/api/research/jobs")
    def list_research_jobs():
        return jsonify(db.list_research_jobs(status=request.args.get("status")))

    @app.post("/api/research/jobs")
    def create_research_job():
        payload = request.get_json(silent=True) or {}
        try:
            job = db.create_research_job(
                trigger_type=payload["trigger_type"],
                prompt=payload["prompt"],
                requested_by=payload.get("requested_by", "system"),
                resource_id=payload.get("resource_id"),
                ask_id=payload.get("ask_id"),
                field=payload.get("field"),
                job_id=payload.get("id"),
            )
        except (KeyError, ValueError) as error:
            return jsonify({"error": str(error)}), 400
        return jsonify(job), 201

    @app.post("/api/research/jobs/<job_id>/run")
    def run_research_job(job_id: str):
        from pathways_ars_worker import run_research_job

        result = run_research_job(db, job_id)
        status = 200 if result.get("status") not in {"failed", "not_found", "busy"} else 409
        return jsonify(result), status

    @app.get("/api/research/suggestions")
    def list_research_suggestions():
        return jsonify(db.list_research_suggestions(job_id=request.args.get("job_id")))

    @app.post("/api/research/suggestions")
    def add_research_suggestion():
        payload = request.get_json(silent=True) or {}
        try:
            suggestion = db.add_research_suggestion(
                job_id=payload["job_id"],
                value=payload["value"],
                source_url=payload["source_url"],
                evidence=payload.get("evidence", ""),
                confidence=payload.get("confidence"),
                resource_id=payload.get("resource_id"),
                field=payload.get("field"),
                suggestion_id=payload.get("id"),
            )
        except (KeyError, ValueError) as error:
            return jsonify({"error": str(error)}), 400
        return jsonify(suggestion), 201

    @app.post("/api/research/suggestions/<suggestion_id>/review")
    def review_research_suggestion(suggestion_id: str):
        payload = request.get_json(silent=True) or {}
        try:
            suggestion = db.review_research_suggestion(
                suggestion_id=suggestion_id,
                status=payload["status"],
                reviewer=payload.get("reviewer", "system"),
            )
        except (KeyError, ValueError) as error:
            return jsonify({"error": str(error)}), 400
        return (jsonify(suggestion), 200) if suggestion else (jsonify({"error": "suggestion not found or already reviewed"}), 404)

    @app.get("/api/asks")
    def list_asks():
        return jsonify(db.list_asks())

    @app.post("/api/asks")
    def create_ask():
        payload = request.get_json(silent=True) or {}
        ask = db.create_ask(
            region=payload.get("region", "Ontario") or "Ontario",
            text=payload["text"],
            author=payload.get("author", "system"),
            tags=payload.get("tags", []),
            linked_case_id=payload.get("linked_case_id"),
            ask_id=payload.get("id"),
        )
        return jsonify(ask), 201

    @app.post("/api/asks/<ask_id>/research")
    def research_ask(ask_id: str):
        asks = [ask for ask in db.list_asks() if ask["id"] == ask_id]
        if not asks:
            return jsonify({"error": "ask not found"}), 404
        ask = asks[0]
        job_id = f"ask-research-{ask_id}"
        existing = [job for job in db.list_research_jobs() if job["id"] == job_id]
        if existing and existing[0]["status"] in {"queued", "running", "needs_review", "completed"}:
            return jsonify(existing[0]), 200
        if existing and existing[0]["status"] == "failed":
            job_id = f"{job_id}-retry-{int(datetime.now(timezone.utc).timestamp())}"
        job = db.create_research_job(
            job_id=job_id,
            trigger_type="ask",
            ask_id=ask_id,
            prompt=f"Region: {ask['region']}. Need: {ask['text']}. Tags: {', '.join(ask.get('tags', []))}",
            requested_by=ask["author"],
        )
        return jsonify(job), 201

    @app.post("/api/asks/<ask_id>/watchers")
    def watch_ask(ask_id: str):
        payload = request.get_json(silent=True) or {}
        ask = db.watch_ask(ask_id, payload.get("watcher", "system"))
        return (jsonify(ask), 200) if ask else (jsonify({"error": "ask not found"}), 404)

    @app.post("/api/asks/<ask_id>/replies")
    def add_reply(ask_id: str):
        payload = request.get_json(silent=True) or {}
        reply = db.add_reply(
            ask_id=ask_id,
            author=payload.get("author", "system"),
            text=payload["text"],
            attached_resource_id=payload.get("attached_resource_id"),
            candidate_name=payload.get("candidate_name"),
        )
        return jsonify(reply), 201

    @app.get("/api/asks/<ask_id>/replies")
    def get_replies(ask_id: str):
        return jsonify(db.list_replies(ask_id))

    @app.post("/api/asks/<ask_id>/resolve")
    def resolve_ask(ask_id: str):
        payload = request.get_json(silent=True) or {}
        ask = db.resolve_ask(ask_id, linked_case_id=payload.get("linked_case_id"))
        return (jsonify(ask), 200) if ask else (jsonify({"error": "ask not found or already resolved"}), 404)

    @app.get("/api/cases")
    def list_cases():
        return jsonify(db.list_cases())

    @app.post("/api/cases/close")
    def close_case_route():
        payload = request.get_json(silent=True) or {}
        case = db.close_case(
            case_id=payload["case_id"],
            outcome=payload["outcome"],
            actor=payload.get("actor", "system"),
            resource_id=payload.get("resource_id"),
            referred_service_name=payload.get("referred_service_name"),
            linked_ask_id=payload.get("linked_ask_id"),
            confirmed_fields=payload.get("confirmed_fields", []),
            note=payload.get("note", ""),
        )
        return jsonify(case), 201

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5001")), debug=True)
