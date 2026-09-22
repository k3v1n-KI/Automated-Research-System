# Pathways dataset inventory

The supplied datasets are source inputs for the M0 import phase. They should be treated as immutable raw inputs; normalized or deduplicated output belongs in a separate processed-data directory.

| File | Domain | Rows | Columns | Missing values | Duplicate normalized names |
|---|---:|---:|---:|---:|---:|
| `Addiction_Dataset.csv` | Addiction services | 540 | 5 | 0 detected | 1 |
| `Hospital_Dataset.csv` | Hospitals | 801 | 4 | 0 detected | 236 |
| `Pharmacy_Dataset.csv` | Pharmacy services | 984 | 7 | 0 detected | 3 |

## Common fields

- `Name`: organization or facility name
- `Address`: source address text
- `City`: source city text
- `source_url`: evidence/provenance URL

## Domain-specific fields

- Addiction: `Phone Number`
- Hospitals: no phone or website field in the supplied source
- Pharmacy: `Phone Number`, `Website`, `Zip Code`

## Import rules

1. Keep the original files unchanged.
2. Attach `source_dataset` and `source_row_number` to every imported record.
3. Normalize values for matching, but retain the original value for display and provenance.
4. Do not merge solely on normalized name. Hospitals require address and source evidence because repeated names may represent separate facilities or duplicate records.
5. Report missing values, duplicate candidates, malformed URLs, and conflicting field values.
6. Assign stable resource IDs only after normalization and duplicate review.

The initial profiling found no empty cells in the three files. This is input completeness, not evidence that every value is correct or current; the Verify workflow remains necessary.

The import report currently identifies one source URL using the `ftp` scheme rather
than `http` or `https` (Addiction row 255, Inner City Health Associates). The importer
preserves it and reports it as an unsupported web URL for later source review.
