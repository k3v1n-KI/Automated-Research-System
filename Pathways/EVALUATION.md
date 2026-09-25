# Pathways evaluation

## Study 1 filter extraction

### Version 1: original 8-case smoke test

The original fixture remains available at
`Pathways/evaluation/study1_extraction_cases.json`. It contains 8 controlled
queries and is retained as a regression check. The original report is
`Pathways/evaluation/study1_extraction_report.json`.

| Metric | Result |
|---|---:|
| Cases | 8 |
| Exact-case accuracy | 1.00 |
| Micro-precision | 1.00 |
| Micro-recall | 1.00 |
| Micro-F1 | 1.00 |

This result is a smoke-test baseline, not evidence of broad natural-language
generalization.

### Version 2: expanded rule-based baseline

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

The original expanded report is retained as
`Pathways/evaluation/study1_broad_report.json`.

| Category | Cases | Exact-case accuracy |
|---|---:|---:|
| Canonical | 25 | 1.00 |
| Paraphrased | 25 | 0.36 |
| **Overall** | **50** | **0.68** |

### Version 3: hybrid ontology-alias extractor

The development hybrid condition adds an ontology alias layer while preserving
the deterministic structured filter output. It is generated with:

```bash
PYTHONPATH=.:Pathways python Pathways/evaluation/evaluate_study1.py \
	--cases Pathways/evaluation/study1_extraction_broad_cases.json \
	--resources Pathways/data/processed/normalized_resources.jsonl \
	--extractor hybrid \
	--output Pathways/evaluation/study1_hybrid_report.json
```

The enriched report retains all 50 case rows and records the category, gold
filter, system output, per-field results, matched aliases, unresolved terms,
extraction route, and ontology version (`hybrid-aliases-v1`).

| Category | Cases | Exact-case accuracy |
|---|---:|---:|
| Canonical | 25 | 1.00 |
| Paraphrased | 25 | 0.88 |
| **Overall** | **50** | **0.94** |

| Metric | Result |
|---|---:|
| Micro-precision | 0.9857 |
| Micro-recall | 0.9787 |
| Micro-F1 | 0.9822 |

The hybrid extractor failed on three paraphrases:

- `p10`: “A hospital serving Indigenous language speakers in Thunder Bay”; the
  gold label expects `Ojibwe`, but the query does not name that language and the
  ontology has no supported mapping for the phrase.
- `p14`: “Addiction support for a teenager in London”; the system returned
  `youth` while the fixture expects `adolescent`.
- `p19`: “A walk-in clinic for a teenager in Vaughan”; the same
  `youth`/`adolescent` ontology mismatch occurred.

This hybrid score is calibration/development evidence because the alias mappings
were created while examining this benchmark. It must not replace the Version 2
result as the primary thesis claim until a held-out paraphrase fixture is
evaluated. The progression nevertheless demonstrates the engineering effect of
alias coverage: paraphrase exact accuracy increased from 0.36 to 0.88 while
canonical accuracy remained 1.00.

### Version 4: semantic and constrained-fallback ablation

The extractor now includes an optional semantic candidate stage and an optional
constrained LLM fallback. The semantic stage ranks unresolved query n-grams
against ontology phrase entries using TF-IDF character similarity. It only fills
an empty field and cannot overwrite a deterministic alias. The LLM stage receives
the unresolved or ambiguous residue plus the allowed ontology values, returns a
JSON proposal, and is validated before it can replace an ambiguous field. Prompt,
raw response, validated proposal, and route are recorded in the report.

The evaluator supports:

```bash
# Semantic stage, no LLM
PYTHONPATH=.:Pathways python Pathways/evaluation/evaluate_study1.py \
	--extractor hybrid --semantic-mode on --llm-mode off \
	--cases Pathways/evaluation/study1_extraction_broad_cases.json \
	--resources Pathways/data/processed/normalized_resources.jsonl \
	--output Pathways/evaluation/study1_semantic_report.json

# Deterministic mock LLM for reproducible control-flow testing
PYTHONPATH=.:Pathways python Pathways/evaluation/evaluate_study1.py \
	--extractor hybrid --semantic-mode on --llm-mode mock \
	--cases Pathways/evaluation/study1_extraction_broad_cases.json \
	--resources Pathways/data/processed/normalized_resources.jsonl \
	--output Pathways/evaluation/study1_semantic_llm_mock_report.json

# Real OpenAI-compatible fallback; requires OPENAI_API_KEY
PYTHONPATH=.:Pathways python Pathways/evaluation/evaluate_study1.py \
	--extractor hybrid --semantic-mode on --llm-mode openai \
	--cases Pathways/evaluation/study1_extraction_broad_cases.json \
	--resources Pathways/data/processed/normalized_resources.jsonl \
	--output Pathways/evaluation/study1_semantic_llm_openai_report.json

# Real Gemini fallback; loads GEMINI_API_KEY from searxng/.env
PYTHONPATH=.:Pathways python Pathways/evaluation/evaluate_study1.py \
	--extractor hybrid --semantic-mode on --llm-mode gemini \
	--cases Pathways/evaluation/study1_extraction_broad_cases.json \
	--resources Pathways/data/processed/normalized_resources.jsonl \
	--output Pathways/evaluation/study1_semantic_llm_gemini_report.json
```

| Condition | Exact-case accuracy | Paraphrased exact | Micro-F1 |
|---|---:|---:|---:|
| Alias-only | 0.94 | 0.88 | 0.9822 |
| Semantic, no LLM | 0.94 | 0.88 | 0.9822 |
| Semantic + mock LLM | 0.98 | 0.96 | 0.9964 |
| Semantic + Gemini | 0.96 | 0.92 | 0.9894 |

The semantic stage accepted 11 candidates without changing the aggregate score
on this fixture. The mock fallback was involved in 33 cases and improved the
context-sensitive population cases, including the distinction between general
teenage support and adolescent addiction or walk-in care. This is a viability
test of the constrained architecture, not a claim about an external model's
performance: the mock provider is deterministic, and the OpenAI condition must
be evaluated separately with a recorded model, prompt, cost, and held-out data.
The real Gemini run used the configured OpenAI-compatible endpoint and model
`gemini-3.8-flash`. It involved 33 of 50 cases and produced 22 validated
suggestion payloads. Its result is still development-fixture evidence, not
held-out thesis evidence; the report records each gold filter, system output,
route, and LLM provenance in
`Pathways/evaluation/study1_semantic_llm_gemini_report.json`.

### Version 5: structured 200-query five-condition study

To increase query coverage, a structured fixture was generated from ten filter
families crossed with ten Ontario locations. Each family contributes one
canonical and one paraphrased form, producing 100 canonical and 100
paraphrased queries. The fixture and generation formula are recorded in
`Pathways/evaluation/study1_extraction_200_cases.json` and
`Pathways/evaluation/generate_study1_200_fixture.py`.

The five conditions are independently switchable:

```bash
# Basic rules
PYTHONPATH=.:Pathways python -m evaluation.evaluate_study1 \
	--cases Pathways/evaluation/study1_extraction_200_cases.json \
	--extractor baseline --alias-mode off --semantic-mode off --llm-mode off \
	--output Pathways/evaluation/study1_200_basic_rules_report.json

# Hybrid aliases
PYTHONPATH=.:Pathways python -m evaluation.evaluate_study1 \
	--cases Pathways/evaluation/study1_extraction_200_cases.json \
	--extractor hybrid --alias-mode on --semantic-mode off --llm-mode off \
	--output Pathways/evaluation/study1_200_hybrid_aliases_report.json

# Semantic-only
PYTHONPATH=.:Pathways python -m evaluation.evaluate_study1 \
	--cases Pathways/evaluation/study1_extraction_200_cases.json \
	--extractor hybrid --alias-mode off --semantic-mode on --llm-mode off \
	--output Pathways/evaluation/study1_200_semantic_only_report.json

# Gemini-only, batched in groups of 50
PYTHONPATH=.:Pathways python -m evaluation.evaluate_study1 \
	--cases Pathways/evaluation/study1_extraction_200_cases.json \
	--extractor hybrid --alias-mode off --semantic-mode off --llm-mode gemini \
	--llm-batch-size 50 \
	--output Pathways/evaluation/study1_200_gemini_only_report.json

# Semantic + Gemini, batched in groups of 50
PYTHONPATH=.:Pathways python -m evaluation.evaluate_study1 \
	--cases Pathways/evaluation/study1_extraction_200_cases.json \
	--extractor hybrid --alias-mode on --semantic-mode on --llm-mode gemini \
	--llm-batch-size 50 \
	--output Pathways/evaluation/study1_200_semantic_gemini_report.json
```

| Condition | Overall exact | Canonical exact | Paraphrased exact | Micro-F1 |
|---|---:|---:|---:|---:|
| Basic rules | 0.600 | 1.000 | 0.200 | 0.8916 |
| Hybrid aliases | 0.750 | 1.000 | 0.500 | 0.9318 |
| Semantic-only | 0.900 | 1.000 | 0.800 | 0.9663 |
| Gemini-only | 0.950 | 1.000 | 0.900 | 0.9890 |
| Semantic + Gemini | 1.000 | 1.000 | 1.000 | 1.0000 |

The two Gemini conditions used four batched requests of 50 cases rather than
200 sequential requests. These remain same-fixture development results, not
held-out evidence. The semantic-plus-Gemini condition should therefore be
treated as evidence that the combined control flow resolves this structured
fixture, not as a general claim about open-ended language understanding.

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
