# Pathways evaluation

This recovery has two executable evaluation surfaces:

- `test_pathways_domain.py` checks ledger replay, Fit + Freshness ranking, and IR metrics.
- `pathways_domain.py` exposes `precision_at_k` and `ndcg_at_k` for comparison against a BM25 or other baseline.

## Planned studies

1. **Flywheel simulation**: seed stale fields, run Verify/Flag/Accept events over T=0, 30, and 90 days, and compare freshness against an unmaintained control.
2. **Information retrieval**: build a relevance set for each case and report Precision@K and nDCG@K for Pathways ranking versus BM25.
3. **Observability**: record search, resource-open, verify, flag, ask, and accept events; report verification rate and time-to-resolution.
4. **Missing-data restoration**: reuse the existing 500-row benchmark protocol, count only requested fields as fills, and retain evidence URLs and field-level correctness labels.

The browser prototype stores its event log in `localStorage` for recovery testing. It is intentionally not a clinical source of truth; a production adapter should persist the same events in PostgreSQL and require authenticated operators for accepting suggestions.
