import json
from pathlib import Path

from evaluation.evaluate_study1 import evaluate


CASES = Path(__file__).parent / "evaluation" / "study1_extraction_cases.json"
BROAD_CASES = Path(__file__).parent / "evaluation" / "study1_extraction_broad_cases.json"
RESOURCES = Path(__file__).parent / "data" / "processed" / "normalized_resources.jsonl"


def test_study1_extraction_report_is_reproducible_and_complete():
    first = evaluate(CASES)
    second = evaluate(CASES)

    assert first == second
    assert first["evaluation_type"] == "study1_filter_extraction"
    assert first["case_count"] == 8
    assert 0.0 <= first["summary"]["micro_f1"] <= 1.0
    assert first["summary"]["exact_case_accuracy"] >= 0.75


def test_study1_extraction_report_is_serializable():
    json.dumps(evaluate(CASES))


def test_study1_broad_benchmark_covers_canonical_and_paraphrased_queries():
    report = evaluate(BROAD_CASES, resources_path=RESOURCES)

    assert report["case_count"] == 50
    assert report["resources"].endswith("normalized_resources.jsonl")
    assert 0.0 <= report["summary"]["micro_f1"] <= 1.0
    assert report["summary"]["micro_f1"] < 1.0


def test_study1_hybrid_ontology_improves_paraphrase_exact_accuracy():
    baseline = evaluate(BROAD_CASES, resources_path=RESOURCES)
    hybrid = evaluate(BROAD_CASES, resources_path=RESOURCES, extractor="hybrid")

    assert hybrid["evaluation_type"] == "study1_hybrid_filter_extraction"
    assert hybrid["summary"]["exact_case_accuracy"] > baseline["summary"]["exact_case_accuracy"]
    assert hybrid["summary"]["micro_recall"] > baseline["summary"]["micro_recall"]


def test_study1_hybrid_report_contains_auditable_case_schema():
    report = evaluate(BROAD_CASES, resources_path=RESOURCES, extractor="hybrid")

    assert report["schema_version"] == 3
    assert report["ontology_version"] == "hybrid-aliases-v1"
    assert report["summary"]["by_category"]["canonical"]["case_count"] == 25
    assert report["summary"]["by_category"]["paraphrased"]["case_count"] == 25
    assert all(case["category"] in {"canonical", "paraphrased"} for case in report["cases"])
    assert all(set(case["field_results"]) == set(report["fields"]) for case in report["cases"])
    assert all(case["extraction_route"] in {"ontology_aliases", "ontology_aliases+semantic"} for case in report["cases"])
    assert report["llm_summary"]["involved_case_count"] == 0
    assert report["llm_summary"]["involved_case_ids"] == []
    assert all(case["llm_involved"] is False for case in report["cases"])
    assert all(case["llm_suggestion"] is None for case in report["cases"])


def test_study1_mock_llm_fallback_is_validated_and_audited():
    report = evaluate(BROAD_CASES, resources_path=RESOURCES, extractor="hybrid", llm_mode="mock")

    assert report["llm_mode"] == "mock"
    assert report["llm_summary"]["involved_case_count"] > 0
    assert all(case["llm_involved"] for case in report["cases"] if case["unresolved_terms"])
    population_suggestions = {
        tuple(suggestion["suggestion"]["population"])
        for suggestion in report["llm_summary"]["suggestions"]
        if "population" in suggestion["suggestion"]
    }
    assert ("adolescent",) in population_suggestions
    assert ("youth",) in population_suggestions
