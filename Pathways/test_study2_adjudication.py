import csv
import json
from pathlib import Path

from evaluation.adjudicate_study2 import (
    bundle_from_csv,
    export_bundle_csv,
    generate_adjudication_bundle,
    score_adjudication_labels,
)


CORPUS = Path(__file__).parent / "evaluation" / "study2_corpus.json"


def test_study2_adjudication_bundle_includes_all_cases_and_candidates():
    bundle = generate_adjudication_bundle(CORPUS)

    assert bundle["evaluation_type"] == "independent_adjudication_bundle"
    assert bundle["case_count"] == 8
    assert bundle["resource_count"] == 12
    assert len(bundle["items"]) == 8 * 12
    assert all(item["label"] is None for item in bundle["items"])
    assert {item["candidate_id"] for item in bundle["items"]} == {
        "r-yee-hong",
        "r-spectrum",
        "r-meals",
        "r-access",
        "r-scarborough-chc",
        "r-woodgreen",
        "r-camh",
        "r-connex",
        "r-youth",
        "r-pharmacy",
        "r-hospital",
        "r-french",
    }


def test_study2_adjudication_bundle_is_serializable():
    bundle = generate_adjudication_bundle(CORPUS)
    json.dumps(bundle)


def test_study2_adjudication_scores_labels_and_summarizes_case_quality():
    bundle = generate_adjudication_bundle(CORPUS)
    for item in bundle["items"]:
        item["label"] = 0
    top_case = bundle["items"][0]
    top_case["label"] = 3
    summary = score_adjudication_labels(bundle)

    assert summary["case_count"] == 8
    assert summary["resource_count"] == 12
    assert 0.0 <= summary["mean_case_label"] <= 3.0
    assert summary["items_labeled"] == len(bundle["items"])


def test_study2_adjudication_csv_roundtrip_preserves_rows(tmp_path):
    bundle = generate_adjudication_bundle(CORPUS)
    csv_path = tmp_path / "adjudication.csv"
    export_bundle_csv(bundle, csv_path)

    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    assert len(rows) == len(bundle["items"])
    assert rows[0]["case_id"] == bundle["items"][0]["case_id"]

    loaded = bundle_from_csv(csv_path)
    assert len(loaded["items"]) == len(bundle["items"])
    assert loaded["items"][0]["candidate_id"] == bundle["items"][0]["candidate_id"]
