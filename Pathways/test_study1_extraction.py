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
