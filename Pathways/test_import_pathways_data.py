import json
from pathlib import Path

from import_pathways_data import import_datasets


EXPECTED_ROWS = {
    "Addiction_Dataset.csv": 540,
    "Hospital_Dataset.csv": 801,
    "Pharmacy_Dataset.csv": 984,
}


def test_import_preserves_rows_and_provenance(tmp_path: Path):
    report = import_datasets(Path(__file__).parent, tmp_path)

    assert report["total_rows"] == sum(EXPECTED_ROWS.values())
    assert report["candidate_records"] == 2325
    assert report["domains"] == {
        "addiction_services": 540,
        "hospitals": 801,
        "pharmacy_services": 984,
    }

    records = [
        json.loads(line)
        for line in (tmp_path / "normalized_resources.jsonl").read_text().splitlines()
    ]
    assert len(records) == 2325
    assert len({record["candidate_id"] for record in records}) == len(records)
    assert {record["source_dataset"] for record in records} == set(EXPECTED_ROWS)
    assert all(record["source_row_number"] >= 2 for record in records)
    assert all(record["verification_state"] == "unknown" for record in records)


def test_import_is_repeatable(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    import_datasets(Path(__file__).parent, first)
    import_datasets(Path(__file__).parent, second)

    for filename in ("normalized_resources.jsonl", "duplicate_candidates.jsonl", "import_report.json"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


def test_duplicate_report_retains_source_rows(tmp_path: Path):
    import_datasets(Path(__file__).parent, tmp_path)
    groups = [
        json.loads(line)
        for line in (tmp_path / "duplicate_candidates.jsonl").read_text().splitlines()
    ]

    assert groups
    assert all(group["candidate_ids"] for group in groups)
    assert all(group["reasons"] for group in groups)
    assert all(member["source_row_number"] >= 2 for group in groups for member in group["records"])
    assert any("name_city" in group["reasons"] for group in groups)
