from pathlib import Path

from evaluation.sweep_study2_weights import evaluate_weights


CORPUS = Path(__file__).parent / "evaluation" / "study2_corpus.json"


def test_weight_sweep_is_reproducible_and_selects_a_valid_blend():
    first = evaluate_weights(CORPUS, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    second = evaluate_weights(CORPUS, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

    assert first == second
    assert first["best_by_ndcg_at_5"]["fit_weight"] == 0.9
    assert first["best_by_ndcg_at_5"]["freshness_weight"] == 0.1
    assert 0.0 <= first["best_by_ndcg_at_5"]["summary"]["ndcg_at_5"] <= 1.0
