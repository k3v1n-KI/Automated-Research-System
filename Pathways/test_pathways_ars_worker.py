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
        database.upsert_resource({
            "id": "resource-1",
            "name": "Resource One",
            "address": "1 Main Street",
            "city": "Toronto",
        })
        result = run_research_job(
            database,
            "worker-job-1",
            executor=lambda job: [{
                "phone": "555-0100",
                "source_url": "https://example.org/contact",
                "evidence": "Resource One at 1 Main Street, Toronto lists phone 555-0100.",
                "confidence": 0.88,
            }],
        )
        assert result["status"] == "needs_review"
        assert result["suggestions"][0]["value"] == "555-0100"
        assert database.list_research_suggestions()[0]["status"] == "proposed"
    finally:
        drop_database(database)


def test_worker_rejects_wrong_entity_evidence():
    database = make_database()
    try:
        database.create_research_job(
            job_id="worker-job-wrong-entity",
            trigger_type="missing_field",
            resource_id="resource-2",
            field="phone",
            prompt="Find the phone number",
            requested_by="tester",
        )
        database.upsert_resource({
            "id": "resource-2",
            "name": "Synthetic Care Centre - East",
            "address": "123 Synthetic Avenue",
            "city": "Testville",
        })
        result = run_research_job(
            database,
            "worker-job-wrong-entity",
            executor=lambda job: [{
                "phone": "416-412-4571",
                "source_url": "https://www.yeehong.com/centre/community-services/information-referral",
                "evidence": "Synthetic Care Centre - West at 999 Other Road lists phone 416-412-4571.",
            }],
        )
        assert result["status"] == "failed"
        assert database.list_research_suggestions() == []
    finally:
        drop_database(database)


def test_worker_rejects_wrong_branch_with_same_organization():
    database = make_database()
    try:
        database.create_research_job(
            job_id="worker-job-wrong-branch",
            trigger_type="missing_field",
            resource_id="resource-3",
            field="phone",
            prompt="Find the phone number",
            requested_by="tester",
        )
        database.upsert_resource({
            "id": "resource-3",
            "name": "Synthetic Care Centre - East",
            "address": "123 Synthetic Avenue",
            "city": "Testville",
        })
        result = run_research_job(
            database,
            "worker-job-wrong-branch",
            executor=lambda job: [{
                "phone": "416-412-4571",
                "source_url": "https://maps.google.com/example",
                "evidence": "Synthetic Care Centre - West at 456 Different Road, Otherville lists phone 416-412-4571.",
            }],
        )
        assert result["status"] == "failed"
        assert database.list_research_suggestions() == []
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


def test_worker_preserves_ask_service_suggestions_for_review():
    database = make_database()
    try:
        database.create_research_job(
            job_id="worker-ask-1",
            trigger_type="ask",
            ask_id="ask-1",
            prompt="Find addiction support in Sault Ste. Marie",
            requested_by="tester",
        )
        result = run_research_job(
            database,
            "worker-ask-1",
            executor=lambda job: [{
                "value": "Algoma Recovery Hub",
                "source_url": "https://example.org/recovery",
                "evidence": "Algoma Recovery Hub provides addiction support in Sault Ste. Marie.",
                "confidence": 0.9,
            }],
        )
        assert result["status"] == "needs_review"
        assert result["suggestions"][0]["value"] == "Algoma Recovery Hub"
        assert result["suggestions"][0]["resource_id"] is None
        reply = database.list_replies("ask-1")[0]
        assert reply["author"] == "ARS"
        assert "Suggested answer: Algoma Recovery Hub" in reply["text"]
        assert "Algoma Recovery Hub provides addiction support" in reply["text"]
        reviewed = database.review_research_suggestion(result["suggestions"][0]["id"], "accepted", "coordinator")
        assert reviewed["resource_id"].startswith("candidate-")
        candidate = database.get_resource(reviewed["resource_id"])
        assert candidate["name"] == "Algoma Recovery Hub"
        assert candidate["status"] == "candidate"
    finally:
        drop_database(database)


def test_worker_rejects_ask_result_outside_region():
    database = make_database()
    try:
        database.create_research_job(
            job_id="worker-ask-wrong-region",
            trigger_type="ask",
            ask_id="ask-region-1",
            prompt="Region: Algoma OHT. Need: pharmacy hours",
            requested_by="tester",
        )
        result = run_research_job(
            database,
            "worker-ask-wrong-region",
            executor=lambda job: [{
                "value": "Affirming Care Pharmacy",
                "source_url": "https://maps.google.com/mississauga",
                "evidence": "Affirming Care Pharmacy is located in Mississauga.",
            }],
        )
        assert result["status"] == "failed"
        assert database.list_research_suggestions() == []
    finally:
        drop_database(database)
