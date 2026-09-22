from uuid import uuid4

from pathways_ars_worker import run_research_job
from pathways_backend import PathwaysDatabase


def make_database():
    return PathwaysDatabase(table_prefix=f"pathways_worker_test_{uuid4().hex}")


def drop_database(database):
    with database._connect() as conn:
        with conn.cursor() as cur:
            for table in (
                database.research_suggestions_table,
                database.research_jobs_table,
                database.events_table,
                database.cases_table,
                database.ask_replies_table,
                database.asks_table,
                database.resources_table,
            ):
                cur.execute(f"DROP TABLE IF EXISTS {table}")
        conn.commit()


def test_worker_translates_ars_record_to_reviewable_suggestion():
    database = make_database()
    try:
        database.create_research_job(
            job_id="worker-job-1",
            trigger_type="missing_field",
            resource_id="resource-1",
            field="phone",
            prompt="Find the phone number",
            requested_by="tester",
        )
        result = run_research_job(
            database,
            "worker-job-1",
            executor=lambda job: [{
                "phone": "555-0100",
                "source_url": "https://example.org/contact",
                "evidence": "The contact page lists 555-0100.",
                "confidence": 0.88,
            }],
        )
        assert result["status"] == "needs_review"
        assert result["suggestions"][0]["value"] == "555-0100"
        assert database.list_research_suggestions()[0]["status"] == "proposed"
    finally:
        drop_database(database)


def test_worker_records_empty_ars_result_as_failure():
    database = make_database()
    try:
        database.create_research_job(
            job_id="worker-job-2",
            trigger_type="ask",
            ask_id="ask-1",
            prompt="Find addiction support",
            requested_by="tester",
        )
        result = run_research_job(database, "worker-job-2", executor=lambda job: [])
        assert result["status"] == "failed"
        assert database.list_research_jobs()[0]["status"] == "failed"
        assert "no evidence" in database.list_research_jobs()[0]["error"]
    finally:
        drop_database(database)
