# Pathways recovery

This folder contains the recovered Pathways design studies and a runnable prototype.

## Planning documents

- [FEATURE_BACKLOG.md](FEATURE_BACKLOG.md) is the sequential implementation plan.
- [DATASETS.md](DATASETS.md) records the supplied dataset schema and import rules.
- [EVALUATION.md](EVALUATION.md) describes the evaluation studies and current limits.

The three CSV files are currently kept at the root as immutable raw inputs while the
import contract is being implemented. They should move to `data/raw/` only as part of
a verified mechanical reorganization. The recovered HTML/JSX design studies remain
separate from the runnable prototype conceptually and should not be edited as part of
feature implementation unless a design reference needs correction.

## Import the datasets

From the repository root:

```bash
python Pathways/import_pathways_data.py \
	--input-dir Pathways \
	--output-dir Pathways/data/processed
```

The importer writes `normalized_resources.jsonl`, `duplicate_candidates.jsonl`, and
`import_report.json`. It does not merge or delete source rows. The current report
contains 2,325 candidates, 324 duplicate candidate groups, and one unsupported `ftp`
source URL in the Addiction dataset; the raw value is preserved for review.

## Run the prototype

From the repository root:

```bash
python -m http.server 8765 --directory Pathways
```

Open `http://127.0.0.1:8765/` in a browser. The prototype uses no build step and stores its append-only browser event ledger in `localStorage` under `pathways.event-ledger.v1`.

## Recovered workflow

- **Find** searches the seeded community-resource projection and shows Fit and Trust scores separately.
- **Verify** appends a field verification event.
- **Ask** appends a stale-field event and places it in the ARS queue.
- **Close** accepts a candidate suggestion and updates the derived projection.

The browser app is a recovery prototype, not a clinical production system. The framework-free domain implementation in `pathways_domain.py` is the reference behavior for a future Flask/PostgreSQL adapter.

## M1 event ledger

`PathwaysStore` derives directory state from immutable `Event` records. Each tracked
field exposes `unknown`, `verified-fresh`, `verified-aging`, or `flagged-stale` state;
accepted ARS suggestions return a field to `verified-fresh`. Freshness and aging windows
are configurable and can be evaluated with a fixed clock for reproducible tests.

The projection also includes per-field `chain_of_custody` entries containing the event,
actor, display identity, source channel, source URL, reason, and correction. Anonymous
events display as `AOHT member` while retaining their internal actor ID in the event.

## M2 Find search

`pathways_find.py` loads `data/processed/normalized_resources.jsonl` into the domain
store and provides deterministic free-text filter extraction, corpus search, transparent
Fit/Trust/Flag scoring, rationales, provenance in results, and in-memory `SearchLog`
records. It is intentionally dependency-free so keyword fallback remains available when
an LLM extractor is unavailable.

The browser prototype's Verify phase now exposes each tracked field, its current state,
Confirm and Flag actions, source provenance, and the append-only custody history. Flagging
supports a reason, optional correction, evidence URL, and anonymous presentation.

## M4 Ask board

The Ask phase supports region-scoped requests with seven-day expiry, tags, watchers,
replies, existing-resource attachments, and candidate service mentions. Ask creation,
replies, watching, and resolution are persisted in local browser state and recorded in
the shared event ledger.

Example:

```python
from pathlib import Path
from pathways_find import FindSearch

search = FindSearch.from_jsonl(Path("Pathways/data/processed/normalized_resources.jsonl"))
results = search.search("pharmacy services in Brampton", limit=10)
```

## Evaluate

```bash
python -m pytest Pathways/test_pathways_domain.py -q
```

See `EVALUATION.md` for the four planned studies and the boundaries of the current benchmark.
