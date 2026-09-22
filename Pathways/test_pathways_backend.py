from uuid import uuid4

from pathways_backend import PathwaysDatabase


def test_research_job_suggestion_review_writes_ledger_event():
    prefix = f"pathways_test_{uuid4().hex}"
    database = PathwaysDatabase(table_prefix=prefix)
    try:
        job = database.create_research_job(
            trigger_type="missing_field",
            resource_id="resource-1",
            field="phone",
            prompt="Find the current phone number for Resource One",
            requested_by="navigator-1",
            job_id="job-1",
        )
        assert job["status"] == "queued"

        suggestion = database.add_research_suggestion(
            job_id="job-1",
            resource_id="resource-1",
            field="phone",
            value="555-0100",
            confidence=0.91,
            source_url="https://example.org/contact",
            evidence="Contact page lists 555-0100.",
            suggestion_id="suggestion-1",
        )
        assert suggestion["status"] == "proposed"
        assert database.list_research_jobs(status="needs_review")[0]["id"] == "job-1"

        reviewed = database.review_research_suggestion("suggestion-1", "accepted", "coordinator-1")
        assert reviewed["status"] == "accepted"
        assert database.list_research_jobs(status="completed")[0]["id"] == "job-1"

        events = database.list_events()
        assert events[0]["kind"] == "suggestion_accepted"
        assert events[0]["resource_id"] == "resource-1"
        assert events[0]["payload"]["source_url"] == "https://example.org/contact"
    finally:
        with database._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {database.research_suggestions_table}")
                cur.execute(f"DROP TABLE IF EXISTS {database.research_jobs_table}")
                cur.execute(f"DROP TABLE IF EXISTS {database.events_table}")
                cur.execute(f"DROP TABLE IF EXISTS {database.cases_table}")
                cur.execute(f"DROP TABLE IF EXISTS {database.ask_replies_table}")
                cur.execute(f"DROP TABLE IF EXISTS {database.asks_table}")
                cur.execute(f"DROP TABLE IF EXISTS {database.resources_table}")
            conn.commit()
