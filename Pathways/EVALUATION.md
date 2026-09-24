# Pathways evaluation

## Study 1 filter extraction

The expanded extraction benchmark is available at
`Pathways/evaluation/study1_extraction_broad_cases.json`. It contains 50 queries:
25 canonical requests and 25 natural-language paraphrases. Run it against the
full normalized resource vocabulary with:

```bash
PYTHONPATH=.:Pathways python Pathways/evaluation/evaluate_study1.py \
	--cases Pathways/evaluation/study1_extraction_broad_cases.json \
	--resources Pathways/data/processed/normalized_resources.jsonl \
	--output Pathways/evaluation/study1_broad_report.json
```

The current result is 0.917 micro-F1 and 0.68 exact-case accuracy. Canonical
queries are 1.00 exact, while paraphrases are 0.36 exact. This establishes a
clear generalization limitation: the current rule-based extractor recognizes the
controlled vocabulary reliably but needs synonym and discourse handling for
natural-language requests, especially population and modality expressions.

This recovery has two executable evaluation surfaces:

- `test_pathways_domain.py` checks ledger replay, Fit + Freshness ranking, and IR metrics.
- `pathways_domain.py` exposes `precision_at_k` and `ndcg_at_k` for comparison against a BM25 or other baseline.

## Study 2 smoke harness

The first reproducible ranking harness is available at:

```bash
PYTHONPATH=Pathways python Pathways/evaluation/evaluate_ranking.py \
	--output Pathways/evaluation/ranking_report.json
```

It uses the versioned development fixture in
`Pathways/evaluation/ranking_cases.json` and compares Pathways ranking against a
fit-only baseline using Precision@5, graded nDCG@5, MRR, and MAP. The generated
report is explicitly labeled `development_smoke_fixture`; its hand-authored cases
are not thesis ground truth. The fixture includes fixed-clock fresh and aging
verification events, so it exercises the freshness weighting path. A thesis-quality
study still requires a larger independently adjudicated relevance corpus, BM25
comparison, repeated queries/seeds, confidence intervals, and stale-result metrics.

An expanded controlled development corpus is available at
`Pathways/evaluation/study2_corpus.json`. It contains 12 resources, 8 queries, graded
relevance labels, and deterministic verification events. Run it with:

```bash
PYTHONPATH=.:Pathways python Pathways/evaluation/evaluate_study2.py \
	--output Pathways/evaluation/study2_report.json
```

The evaluator compares Pathways, fit-only, and a local BM25 implementation using
Precision@5, graded nDCG@5, MRR, and MAP. The current controlled-corpus result is a
diagnostic only: Pathways nDCG@5 is 0.804, fit-only is 0.879, and BM25 is 0.916;
Pathways Precision@5 is 0.350 versus 0.300 for fit-only and 0.325 for BM25. This
indicates that freshness can improve some cases while also promoting recently verified
lower-grade resources above lexically stronger results. The labels are hand-authored
and the corpus is not yet suitable as thesis evidence.

A sensitivity sweep is available at
`Pathways/evaluation/sweep_study2_weights.py`. On the current corpus, the best
in-sample blend among the tested settings is 0.9 fit / 0.1 freshness, with nDCG@5
0.933, compared with 0.804 for the current freshness-heavy setting and 0.916 for
BM25. This is calibration evidence only; the weight must be selected and evaluated
on held-out or independently adjudicated queries before changing production ranking.

## Automated missing-data benchmark

The sealed benchmark foundation is available at
`Pathways/evaluation/missing_data_benchmark.py`.

Generate the reproducible 500-row fixture:

```bash
PYTHONPATH=Pathways python Pathways/evaluation/missing_data_benchmark.py generate \
	--input-dir Pathways \
	--output-dir Pathways/evaluation/missing_data_benchmark \
	--sample-size 500 \
	--seed 20260922
```

The public JSONL contains identifying context and declared `missing_fields`; the
separate gold JSONL contains expected values and must not be exposed to ARS during
prediction. Score an ARS prediction JSONL with the same runner's `score` command.
The report separates fill rate, entity-supported fill rate, exact/normalized match
rate, wrong-entity rate, unresolved fields, and unexpected fills. This is an
automated benchmark foundation, not a replacement for a stratified audit sample.

The first Places-only baseline showed high raw coverage but substantial wrong-entity
selection because it chose the first API result. The evaluator now uses fuzzy matching
over returned name, address, city, postal code, and website domain, with abstention when
no candidate clears the support threshold. This intentionally trades fill rate for
entity precision and should be rerun as a separate fuzzy-matching experiment rather
than mixed with the original baseline.

## Planned studies

1. **Flywheel simulation**: seed stale fields, run Verify/Flag/Accept events over T=0, 30, and 90 days, and compare freshness against an unmaintained control.
2. **Information retrieval**: build a relevance set for each case and report Precision@K and nDCG@K for Pathways ranking versus BM25.
3. **Observability**: record search, resource-open, verify, flag, ask, and accept events; report verification rate and time-to-resolution.
4. **Missing-data restoration**: reuse the existing 500-row benchmark protocol, count only requested fields as fills, and retain evidence URLs and field-level correctness labels.

## Hybrid unresolved-field fallback

`Pathways/evaluation/run_hybrid_missing_data.py` preserves the Places v2 predictions
and runs only unresolved rows through the original bounded ARS pipeline. Its merged
output is a separate condition, so Places-only and ARS-assisted results remain
auditable. A pilot recovered one additional validated field from an official web
source without introducing a wrong-entity result. The full unresolved-row run should
be executed only after inspecting provider quota and latency, because it invokes the
LLM, search, scraping, and extraction stages per unresolved row.

The 357 entity-supported but non-exact hybrid fills are classified by
`Pathways/evaluation/analyze_missing_data_mismatches.py`. The resulting
`Pathways/evaluation/missing_data_benchmark/mismatch_audit.json` reports 155 high-
confidence representation differences, 75 likely valid alternates, and 127 cases
that need manual verification. These tiers are review triage heuristics, not a
replacement for source-level adjudication; in particular, name-and-city agreement
can support an entity match even when an address or contact value has materially
changed.

The browser prototype stores its event log in `localStorage` for recovery testing. It is intentionally not a clinical source of truth; a production adapter should persist the same events in PostgreSQL and require authenticated operators for accepting suggestions.
