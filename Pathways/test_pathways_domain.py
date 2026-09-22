from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

from pathways_domain import PathwaysStore, ndcg_at_k, precision_at_k, seed_resources


def test_verify_ask_close_replays_to_projection():
    store = PathwaysStore(seed_resources())
    store.verify("res-yee-hong", "phone", actor="nurse")
    store.flag("res-yee-hong", "phone", "number may be stale", actor="nurse")
    store.accept_suggestion(
        "res-yee-hong",
        "phone",
        "416-321-3001",
        "https://www.yeehong.com/contact",
        actor="operator",
    )

    record = store.project()["res-yee-hong"]

    assert record["phone"] == "416-321-3001"
    assert record["verified_fields"] == ["phone"]
    assert record["flagged_fields"] == []
    assert store.asks("open") == []
    assert store.asks("resolved")[0]["field"] == "phone"


def test_search_combines_fit_and_freshness():
    store = PathwaysStore(seed_resources())
    old = datetime.now(timezone.utc) - timedelta(days=120)
    store.append("res-spectrum", "field_verified", "nurse", field="website")
    store.events[-1] = store.events[-1].__class__(
        id=store.events[-1].id,
        resource_id=store.events[-1].resource_id,
        kind=store.events[-1].kind,
        actor=store.events[-1].actor,
        created_at=old,
        payload=store.events[-1].payload,
    )

    results = store.search("Mandarin Scarborough", limit=3)

    assert results[0]["id"] == "res-yee-hong"
    assert results[0]["trust_score"] == 0.0


def test_ir_metrics():
    retrieved = ["a", "c", "b", "d"]
    relevant = {"a", "b"}

    assert precision_at_k(retrieved, relevant, 2) == 0.5
    assert 0 < ndcg_at_k(retrieved, relevant, 4) < 1


def test_field_states_decay_flag_and_recover():
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store = PathwaysStore(seed_resources(), freshness_window_days=30, aging_window_days=90)

    assert store.project(now=base)["res-yee-hong"]["field_states"]["phone"] == "unknown"
    store.verify(
        "res-yee-hong",
        "phone",
        actor="user-123",
        source="call",
        display_actor="Rita",
        anonymous=True,
        created_at=base,
    )
    assert store.project(now=base + timedelta(days=30))["res-yee-hong"]["field_states"]["phone"] == "verified-fresh"
    assert store.project(now=base + timedelta(days=31))["res-yee-hong"]["field_states"]["phone"] == "verified-aging"
    assert store.project(now=base + timedelta(days=91))["res-yee-hong"]["field_states"]["phone"] == "unknown"

    store.flag(
        "res-yee-hong",
        "phone",
        "wrong",
        actor="user-123",
        correction="416-321-3001",
        source="web",
        source_url="https://example.org/contact",
        created_at=base + timedelta(days=91),
    )
    flagged = store.project(now=base + timedelta(days=91))["res-yee-hong"]
    assert flagged["field_states"]["phone"] == "flagged-stale"
    assert flagged["chain_of_custody"]["phone"][0]["display_actor"] == "AOHT member"
    assert flagged["chain_of_custody"]["phone"][1]["correction"] == "416-321-3001"

    store.accept_suggestion(
        "res-yee-hong",
        "phone",
        "416-321-3001",
        source_url="https://example.org/contact",
        created_at=base + timedelta(days=92),
    )
    resolved = store.project(now=base + timedelta(days=92))["res-yee-hong"]
    assert resolved["field_states"]["phone"] == "verified-fresh"
    assert resolved["flagged_fields"] == []
    assert len(resolved["chain_of_custody"]["phone"]) == 3


def test_event_records_are_frozen():
    event = PathwaysStore(seed_resources()).verify("res-yee-hong", "phone")

    try:
        event.kind = "changed"
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError("event records must be immutable")


def test_ask_board_lifecycle_and_expiry():
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store = PathwaysStore(seed_resources())
    ask = store.create_ask(
        "Algoma OHT",
        "Need addiction support in Sault Ste. Marie",
        tags=("addiction", "Sault Ste. Marie"),
        author="rita",
        created_at=base,
    )
    store.watch_ask(ask.id, "jamie")
    reply = store.reply_to_ask(
        ask.id,
        "Try this community program.",
        author="jamie",
        attached_resource_id="res-yee-hong",
        candidate_name="Algoma Recovery Hub",
        created_at=base + timedelta(hours=1),
    )

    board = store.asks_board("Algoma OHT", now=base + timedelta(days=1))
    assert board[0]["ask"]["status"] == "open"
    assert board[0]["ask"]["watchers"] == ("jamie",)
    assert board[0]["replies"][0]["candidate_name"] == "Algoma Recovery Hub"
    assert reply.attached_resource_id == "res-yee-hong"
    assert store.resolve_ask(ask.id, actor="rita").status == "resolved"

    rural = store.create_ask("North OHT", "Need a rural pharmacy", rural_extended=True, created_at=base)
    assert store.asks_board("North OHT", now=base + timedelta(days=12, seconds=1))[0]["ask"]["status"] == "expired"


def test_closed_ask_rejects_new_replies():
    store = PathwaysStore(seed_resources())
    ask = store.create_ask("Algoma OHT", "Need a service", created_at=datetime.now(timezone.utc))
    store.resolve_ask(ask.id)

    try:
        store.reply_to_ask(ask.id, "Too late")
    except ValueError as error:
        assert "closed or expired" in str(error)
    else:
        raise AssertionError("closed asks must reject replies")


def test_close_case_records_outcome_and_seeds_candidate():
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store = PathwaysStore(seed_resources())
    ask = store.create_ask("Algoma OHT", "Need addiction support", created_at=base)

    summary = store.close_case(
        case_id="case-42",
        outcome="referred",
        resource_id="res-yee-hong",
        referred_service_name="Algoma Recovery Hub",
        linked_ask_id=ask.id,
        confirmed_fields=("phone", "website"),
        actor="rita",
        created_at=base + timedelta(days=1),
    )

    assert summary["outcome"] == "referred"
    assert summary["resolved_asks"] == [ask.id]
    assert summary["confirmed_fields"] == ["phone", "website"]
    assert store.asks("resolved")[0]["id"] == ask.id
    assert summary["candidate_seeded"] is False

    candidate = store.close_case(
        case_id="case-43",
        outcome="referred",
        referred_service_name="North Shore Community Service",
        linked_ask_id=None,
        confirmed_fields=("name",),
        actor="rita",
        created_at=base + timedelta(days=2),
    )

    assert candidate["candidate_seeded"] is True
    assert candidate["seeded_resource_id"].startswith("candidate-")
    resource = store.project(now=base + timedelta(days=2))[candidate["seeded_resource_id"]]
    assert resource["name"] == "North Shore Community Service"
    assert resource["field_states"]["name"] == "verified-fresh"
