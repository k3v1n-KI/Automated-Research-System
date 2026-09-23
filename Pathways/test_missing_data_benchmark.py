import json
from pathlib import Path

from evaluation.missing_data_benchmark import generate_benchmark, score_predictions


def test_generate_keeps_gold_separate(tmp_path: Path):
    manifest = generate_benchmark(Path(__file__).parent, tmp_path, sample_size=8, seed=7)
    public = (tmp_path / "missing_data_public.jsonl").read_text()
    gold = (tmp_path / "missing_data_gold.jsonl").read_text()

    assert manifest["benchmark_rows"] == 8
    assert public
    assert gold
    assert "expected" not in public
    assert all(json.loads(line)["missing_fields"] for line in public.splitlines())


def test_score_counts_exact_wrong_entity_and_unexpected_fill(tmp_path: Path):
    generate_benchmark(Path(__file__).parent, tmp_path, sample_size=2, seed=11)
    public_rows = [json.loads(line) for line in (tmp_path / "missing_data_public.jsonl").read_text().splitlines()]
    gold_rows = [json.loads(line) for line in (tmp_path / "missing_data_gold.jsonl").read_text().splitlines()]
    predictions = []
    for index, public in enumerate(public_rows):
        field = public["missing_fields"][0]
        expected = gold_rows[index]["expected"][field]
        predictions.append({
            "benchmark_id": public["benchmark_id"],
            "name": public["name"],
            "address": public["address"],
            "city": public["city"],
            field: expected,
            "website": "unexpected-value",
        })
    predictions[1]["name"] = "Different organization"
    prediction_path = tmp_path / "predictions.jsonl"
    prediction_path.write_text("".join(json.dumps(row) + "\n" for row in predictions))

    report = score_predictions(
        tmp_path / "missing_data_public.jsonl",
        tmp_path / "missing_data_gold.jsonl",
        prediction_path,
    )
    metrics = report["metrics"]
    assert metrics["requested"] == 2
    assert metrics["exact"] == 1
    assert metrics["wrong_entity"] == 1
    assert metrics["unexpected_fills"] == 0
