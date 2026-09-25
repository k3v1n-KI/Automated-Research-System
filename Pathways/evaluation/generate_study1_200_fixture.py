"""Generate the structured 200-query Study 1 benchmark."""

from __future__ import annotations

import json
from pathlib import Path


OUTPUT = Path(__file__).with_name("study1_extraction_200_cases.json")
CITIES = (
    "Toronto", "Ottawa", "Brampton", "Mississauga", "Hamilton",
    "London", "Windsor", "Markham", "Scarborough", "Vaughan",
)

# Ten filter families crossed with ten locations yield 100 canonical and
# 100 paraphrased queries. Every family has one canonical and one natural-language form.
FAMILIES = (
    ("need", "addiction services {city}", "help with substance abuse in {city}", {"need": ["addiction_services"]}),
    ("need", "pharmacy medication {city}", "help getting meds from a drug store in {city}", {"need": ["pharmacy_services"]}),
    ("need", "hospital services {city}", "find hospital care in {city}", {"need": ["hospitals"]}),
    ("modality", "walk-in care {city}", "a clinic accepting patients without an appointment in {city}", {"modality": ["walk-in"]}),
    ("modality", "virtual services {city}", "online support from home in {city}", {"modality": ["virtual"]}),
    ("modality", "same-day care {city}", "care needed today in {city}", {"modality": ["same-day"], "urgency": "same-day"}),
    ("population", "adolescent services {city}", "support for a young teenager in {city}", {"population": ["adolescent"]}),
    ("population", "senior services {city}", "support for an older person in {city}", {"population": ["senior"]}),
    ("coverage", "no OHIP pharmacy {city}", "medication help for someone without a health card in {city}", {"need": ["pharmacy_services"], "coverage": ["no-ohip"]}),
    ("language", "French hospital services {city}", "French-speaking hospital support in {city}", {"need": ["hospitals"], "language": ["French"]}),
)


def gold_filter(values: dict[str, object], city: str) -> dict[str, object]:
    result = {
        "need": [], "modality": [], "population": [], "coverage": [],
        "location": city, "language": [], "urgency": "",
    }
    result.update(values)
    return result


def build_cases() -> list[dict[str, object]]:
    canonical = []
    paraphrased = []
    canonical_index = 1
    paraphrase_index = 1
    for _, canonical_template, paraphrase_template, values in FAMILIES:
        for city in CITIES:
            canonical.append({
                "id": f"c{canonical_index:03d}",
                "category": "canonical",
                "query": canonical_template.format(city=city),
                "gold": gold_filter(values, city),
            })
            canonical_index += 1
            paraphrased.append({
                "id": f"p{paraphrase_index:03d}",
                "category": "paraphrased",
                "query": paraphrase_template.format(city=city),
                "gold": gold_filter(values, city),
            })
            paraphrase_index += 1
    return canonical + paraphrased


if __name__ == "__main__":
    payload = {
        "schema_version": 2,
        "description": "Structured Study 1 benchmark: 10 filter families crossed with 10 cities, yielding 100 canonical and 100 paraphrased queries.",
        "formula": {
            "families": 10,
            "locations": list(CITIES),
            "queries_per_category": "10 families * 10 locations",
            "total_queries": 200,
        },
        "cases": build_cases(),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote={OUTPUT} cases={len(payload['cases'])}")
