import json
from pathlib import Path

from evaluation.evaluate_study2 import evaluate


CORPUS = Path(__file__).parent / "evaluation" / "study2_corpus.json"
ADJUDICATION = Path(__file__).parent / "evaluation" / "study2_adjudication_scores.csv"


def test_study2_corpus_is_reproducible():
    first = evaluate(CORPUS)
    second = evaluate(CORPUS)

    assert first["evaluation_type"] == "controlled_development_corpus"
    assert first["corpus_size"] == 12
    assert first["case_count"] == 8
    assert first["summary"] == second["summary"]
    assert first["cases"] == second["cases"]


def test_study2_reports_all_baselines_and_metrics():
    report = evaluate(CORPUS)

    for baseline in ("pathways", "fit_only", "bm25"):
        for metric in ("precision_at_5", "ndcg_at_5", "mrr", "map"):
            assert 0.0 <= report["summary"][f"{baseline}_{metric}"] <= 1.0


def test_study2_report_can_be_serialized():
    report = evaluate(CORPUS)
    json.dumps(report)


def test_study2_evaluation_can_use_adjudicated_csv_labels():
    report = evaluate(CORPUS, adjudication_path=ADJUDICATION)

    assert report["evaluation_type"] == "independent_adjudication"
    assert report["judgment_count"] == 96
    assert report["summary"]["pathways_ndcg_at_5"] >= 0.0
