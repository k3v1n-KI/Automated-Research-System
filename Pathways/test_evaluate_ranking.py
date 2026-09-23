import json
from pathlib import Path

from evaluation.evaluate_ranking import evaluate


def test_ranking_report_is_reproducible():
    cases = Path(__file__).parent / "evaluation" / "ranking_cases.json"
    first = evaluate(cases)
    second = evaluate(cases)

    assert first["evaluation_type"] == "development_smoke_fixture"
    assert first["case_count"] == 2
    assert first["summary"] == second["summary"]
    assert first["cases"] == second["cases"]
    assert first["summary"]["pathways_ndcg_at_5"] >= 0.0
    assert first["summary"]["fit_only_ndcg_at_5"] >= 0.0
