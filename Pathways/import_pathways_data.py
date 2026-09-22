"""Import Pathways CSV sources into a normalized, provenance-preserving corpus.

The importer never merges or deletes source rows. It writes normalized records,
duplicate candidate groups, and a report so duplicate decisions can be reviewed
before stable directory entities are created.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse


COMMON_FIELDS = (
    "name",
    "address",
    "city",
    "phone",
    "website",
    "postal_code",
    "category",
    "source_url",
)

DATASET_SPECS = {
    "Addiction_Dataset.csv": {
        "domain": "addiction_services",
        "category": "Addiction services",
    },
    "Hospital_Dataset.csv": {
        "domain": "hospitals",
        "category": "Hospital",
    },
    "Pharmacy_Dataset.csv": {
        "domain": "pharmacy_services",
        "category": "Pharmacy services",
    },
}

FIELD_ALIASES = {
    "name": "Name",
    "address": "Address",
    "city": "City",
    "phone": "Phone Number",
    "website": "Website",
    "postal_code": "Zip Code",
    "source_url": "source_url",
}

REQUIRED_COLUMNS = {"Name", "Address", "City", "source_url"}


@dataclass(frozen=True)
class NormalizedRecord:
    candidate_id: str
    domain: str
    category: str
    name: str
    address: str
    city: str
    phone: str
    website: str
    postal_code: str
    source_url: str
    source_dataset: str
    source_row_number: int
    verification_state: str = "unknown"


class ImportErrorDetail(ValueError):
    """Raised when a source file cannot satisfy its declared import contract."""


def clean_text(value: str | None) -> str:
    value = unicodedata.normalize("NFKC", value or "")
    return re.sub(r"\s+", " ", value).strip()


def normalize_match(value: str | None) -> str:
    value = clean_text(value).casefold()
    return re.sub(r"[^a-z0-9]+", "", value)


def normalize_phone(value: str | None) -> str:
    return re.sub(r"\D", "", clean_text(value))


def normalize_postal_code(value: str | None) -> str:
    return re.sub(r"\s+", "", clean_text(value)).upper()


def normalize_url(value: str | None) -> str:
    return clean_text(value).rstrip("/")


def is_valid_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def candidate_id(
    domain: str,
    name: str,
    address: str,
    city: str,
    source_dataset: str,
    source_row_number: int,
) -> str:
    identity = "|".join(
        normalize_match(value)
        for value in (domain, name, address, city, source_dataset, str(source_row_number))
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"candidate-{digest}"


def source_files(input_dir: Path) -> list[Path]:
    missing = [name for name in DATASET_SPECS if not (input_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing Pathways dataset(s): {', '.join(missing)}")
    return [input_dir / name for name in DATASET_SPECS]


def read_dataset(path: Path) -> tuple[list[NormalizedRecord], dict]:
    spec = DATASET_SPECS[path.name]
    records: list[NormalizedRecord] = []
    missing_values: dict[str, int] = defaultdict(int)
    malformed_urls = 0
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        headers = set(reader.fieldnames or ())
        missing_columns = REQUIRED_COLUMNS - headers
        if missing_columns:
            raise ImportErrorDetail(f"{path.name} is missing columns: {sorted(missing_columns)}")
        for row_number, row in enumerate(reader, start=2):
            values = {
                field: clean_text(row.get(source_field, ""))
                for field, source_field in FIELD_ALIASES.items()
            }
            for field, value in values.items():
                if not value:
                    missing_values[field] += 1
            if values["source_url"] and not is_valid_url(values["source_url"]):
                malformed_urls += 1
            record = NormalizedRecord(
                candidate_id=candidate_id(
                    spec["domain"],
                    values["name"],
                    values["address"],
                    values["city"],
                    path.name,
                    row_number,
                ),
                domain=spec["domain"],
                category=spec["category"],
                name=values["name"],
                address=values["address"],
                city=values["city"],
                phone=values["phone"],
                website=values["website"],
                postal_code=values["postal_code"],
                source_url=values["source_url"],
                source_dataset=path.name,
                source_row_number=row_number,
            )
            records.append(record)
    return records, {
        "rows": len(records),
        "missing_values": dict(sorted(missing_values.items())),
        "malformed_source_urls": malformed_urls,
    }


def duplicate_groups(records: Iterable[NormalizedRecord]) -> list[dict]:
    buckets: dict[tuple[str, str], set[str]] = defaultdict(set)
    records_by_id = {record.candidate_id: record for record in records}
    for record in records:
        fields = {
            "name_city": (normalize_match(record.name), normalize_match(record.city)),
            "address_city": (normalize_match(record.address), normalize_match(record.city)),
            "phone": (normalize_phone(record.phone),),
            "website": (normalize_match(record.website),),
        }
        for reason, values in fields.items():
            if all(values):
                buckets[(reason, "|".join(values))].add(record.candidate_id)

    groups: dict[frozenset[str], set[str]] = defaultdict(set)
    for (reason, _), candidate_ids in buckets.items():
        if len(candidate_ids) > 1:
            groups[frozenset(candidate_ids)].add(reason)

    output = []
    for candidate_ids, reasons in sorted(groups.items(), key=lambda item: sorted(item[0])):
        members = [records_by_id[candidate_id] for candidate_id in sorted(candidate_ids)]
        output.append(
            {
                "candidate_ids": sorted(candidate_ids),
                "reasons": sorted(reasons),
                "records": [
                    {
                        "candidate_id": member.candidate_id,
                        "name": member.name,
                        "address": member.address,
                        "city": member.city,
                        "source_dataset": member.source_dataset,
                        "source_row_number": member.source_row_number,
                    }
                    for member in members
                ],
            }
        )
    return output


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
            count += 1
    return count


def import_datasets(input_dir: Path, output_dir: Path) -> dict:
    all_records: list[NormalizedRecord] = []
    by_dataset: dict[str, dict] = {}
    for path in source_files(input_dir):
        records, report = read_dataset(path)
        all_records.extend(records)
        by_dataset[path.name] = report

    all_records.sort(key=lambda record: (record.source_dataset, record.source_row_number))
    candidates = duplicate_groups(all_records)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "normalized_resources.jsonl", (asdict(record) for record in all_records))
    write_jsonl(output_dir / "duplicate_candidates.jsonl", candidates)

    report = {
        "schema_version": 1,
        "total_rows": len(all_records),
        "candidate_records": len(all_records),
        "duplicate_candidate_groups": len(candidates),
        "datasets": by_dataset,
        "domains": {
            domain: sum(record.domain == domain for record in all_records)
            for domain in sorted({record.domain for record in all_records})
        },
    }
    (output_dir / "import_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "data" / "processed",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = import_datasets(args.input_dir, args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
