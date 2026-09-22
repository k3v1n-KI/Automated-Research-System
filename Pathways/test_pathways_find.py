from datetime import datetime, timedelta, timezone
from pathlib import Path

from pathways_domain import PathwaysStore, Resource, seed_resources
from pathways_find import FindSearch, SearchFilters, search_log_dict


PROCESSED = Path(__file__).parent / "data" / "processed" / "normalized_resources.jsonl"


def test_extract_filters_from_case_text():
    search = FindSearch(PathwaysStore(seed_resources()))

    filters = search.extract_filters("same-day addiction support for an adolescent, no OHIP, Scarborough")

    assert filters.need == ("addiction_services",)
    assert filters.modality == ("same-day",)
    assert filters.population == ("adolescent",)
    assert filters.coverage == ("no-ohip",)
    assert filters.location == "Scarborough"
    assert filters.urgency == "same-day"


def test_corpus_loader_and_provenance():
    search = FindSearch.from_jsonl(PROCESSED)
    results = search.search("hospitals Toronto", limit=5)

    assert len(search.store._seed) == 2325
    assert results
    assert results[0]["source_dataset"] == "Hospital_Dataset.csv"
    assert results[0]["source_row_number"] >= 2
    assert results[0]["source_url"].startswith("http")
    assert len(search.logs) == 1


def test_search_logs_filters_and_scores():
    search = FindSearch(PathwaysStore(seed_resources()))

    results = search.search("Mandarin Scarborough", limit=2)
    log = search.logs[0]
    serialized = search_log_dict(log)

    assert len(results) == 2
    assert serialized["query"] == "Mandarin Scarborough"
    assert serialized["filters"]["location"] == "Scarborough"
    assert serialized["returned_ids"] == tuple(result["id"] for result in results)
    assert serialized["scores"] == tuple(result["rank_score"] for result in results)
    assert "fit" in results[0]["rationale"]
    assert "trust" in results[0]["rationale"]


def test_flags_reduce_rank_score():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    resources = [
        Resource(id="a", name="Mandarin Hospital", category="Hospital", city="Ottawa", languages=("Mandarin",)),
        Resource(id="b", name="Mandarin Hospital", category="Hospital", city="Ottawa", languages=("Mandarin",)),
    ]
    search = FindSearch(PathwaysStore(resources))
    clean = search.search("Mandarin Hospital Ottawa", now=now)
    search.store.flag("a", "phone", "wrong", created_at=now)
    flagged = search.search("Mandarin Hospital Ottawa", now=now)

    assert clean[0]["id"] == "a"
    assert flagged[0]["id"] == "b"
    assert flagged[1]["flag_penalty"] > 0


def test_freshness_is_visible_in_ranking():
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    resources = [
        Resource(id="fresh", name="Mandarin Hospital", category="Hospital", city="Ottawa", languages=("Mandarin",)),
        Resource(id="aging", name="Mandarin Hospital", category="Hospital", city="Ottawa", languages=("Mandarin",)),
    ]
    search = FindSearch(PathwaysStore(resources))
    search.store.verify("fresh", "name", created_at=base)
    search.store.verify("aging", "name", created_at=base - timedelta(days=46))

    results = search.search("Mandarin Hospital Ottawa", now=base + timedelta(days=45))

    assert results[0]["id"] == "fresh"
    assert results[0]["trust_score"] > results[1]["trust_score"]
