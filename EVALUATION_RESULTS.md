# Automated Research System Evaluation Results

**Evaluation date:** 2026-09-22  
**Environment:** Python 3.12, `auto-research` conda environment  
**Primary application:** Pathways healthcare-resource discovery and missing-data restoration system

## 1. Purpose and Scope

This document records the tests conducted on the Automated Research System and the results available for use when revising the thesis. It distinguishes executable software validation from thesis-level empirical evidence.

The evaluation covered:

1. Domain and backend regression tests.
2. Ranking behavior using a small development smoke fixture.
3. Automated missing-data restoration on a sealed 500-row benchmark.
4. Google Places retrieval with fuzzy entity matching.
5. Hybrid restoration that sends unresolved Places rows through the original ARS pipeline.
6. Analysis of values that matched the correct entity but differed from the benchmark's historical gold value.
7. Search, LLM, Places, and scraping provider smoke tests.
8. An earlier real-URL scraper failure study.

The results show that the system is operational and capable of high-coverage automated restoration while preserving a conservative wrong-entity rate in the evaluated conditions. They do not yet constitute a complete validation of every thesis study or a clinical safety evaluation.

## 2. Executive Summary

The strongest completed result is the hybrid missing-data experiment:

| Metric | Result |
|---|---:|
| Benchmark fields requested | 500 |
| Fields filled | 499 |
| Validated entity-supported fills | 499 |
| Exact or normalized gold matches | 142 |
| Non-exact but entity-supported fills | 357 |
| Unresolved fields | 1 |
| Wrong-entity fills | 0 |
| Unexpected fills | 0 |
| Validated fill rate | 99.8% |
| Exact match rate | 28.4% |

The Places-only fuzzy condition produced 340 entity-supported fills and 84 exact matches, with zero wrong-entity fills. Adding the ARS fallback increased coverage from 68.0% to 99.8% and exact matches from 16.8% to 28.4%.

The central interpretation is therefore two-dimensional:

- **Coverage and entity safety were strong:** the hybrid condition filled 499 of 500 requested fields without an evaluated wrong-entity result.
- **Exact agreement with the historical gold fixture was substantially lower:** only 142 values matched the gold values exactly after field normalization.

The 357 non-exact values should not be treated as automatically incorrect. An audit classified 155 as high-confidence representation differences, 75 as likely valid alternates, and 127 as requiring manual verification. Those categories are automated triage heuristics and should not be reported as independently verified truth.

## 3. Evaluation Artifacts

The principal artifacts are:

- [Pathways evaluation guide](Pathways/EVALUATION.md)
- [Hybrid predictions](Pathways/evaluation/missing_data_benchmark/predictions_hybrid.jsonl)
- [Hybrid benchmark report](Pathways/evaluation/missing_data_benchmark/report_hybrid.json)
- [Fuzzy Places predictions](Pathways/evaluation/missing_data_benchmark/predictions_places_fuzzy_v2.jsonl)
- [Mismatch audit](Pathways/evaluation/missing_data_benchmark/mismatch_audit.json)
- [Ranking smoke report](Pathways/evaluation/ranking_report.json)
- [Sealed benchmark scorer](Pathways/evaluation/missing_data_benchmark.py)
- [Mismatch audit program](Pathways/evaluation/analyze_missing_data_mismatches.py)
- [Hybrid runner](Pathways/evaluation/run_hybrid_missing_data.py)

The public benchmark fixture contains identifying context and missing-field declarations. The separate gold fixture contains expected values and must not be passed to the ARS during prediction.

## 4. Software and Regression Tests

### 4.1 Test suite

The Pathways test suite was run after the final hybrid-runner and mismatch-audit changes:

```text
26 passed in 1.92s
```

The suite covers the Pathways domain, search/ranking behavior, backend behavior, ARS worker behavior, the missing-data benchmark, and the ranking evaluator. The tests are designed to be deterministic and synthetic-safe where external providers are not required.

### 4.2 Event-loop regression

The original hybrid runner reused asynchronous provider clients across multiple `asyncio.run()` calls. During a two-row pilot, the first row completed but the second failed with:

```text
RuntimeError: Event loop is closed
```

The runner was changed to execute each ARS fallback row in a fresh subprocess. The post-fix two-row pilot completed both rows successfully, and the full test suite remained green.

This isolation is important for reproducibility because the hybrid runner invokes asynchronous Gemini, search, scraping, and extraction stages repeatedly.

### 4.3 Backend and application checks

The application-level implementation checks verified:

- Flask backend imports and route wiring.
- PostgreSQL-backed event and projection paths.
- Research job creation, claiming, failure, and suggestion review paths.
- Ask/reply and case-closing flows.
- ARS candidate seeding and human-review gates.
- Frontend-to-backend research setup flow.
- Dataset and diversity-report plumbing.

The local Flask server was already running on port 5001 during later checks; a second attempt to start it returned `Address already in use`. This was an environment-state observation, not an application failure.

## 5. Study 2: Ranking Smoke Evaluation

### 5.1 Design

The ranking evaluator uses two hand-authored development cases and compares Pathways ranking with a fit-only baseline. Pathways includes fit and freshness behavior, while the baseline uses fit without freshness weighting.

The fixture includes fixed-clock fresh and aging verification events. This exercises the freshness path but does not provide a statistically meaningful estimate of general ranking performance.

### 5.2 Results

| Metric | Pathways | Fit-only baseline |
|---|---:|---:|
| Precision@5 | 0.667 | 0.667 |
| nDCG@5 | 1.000 | 0.904 |
| MRR | 1.000 | 1.000 |
| MAP | 1.000 | 1.000 |

At the case level:

- For `Mandarin Scarborough`, both systems achieved Precision@5 of 1.0 and MRR of 1.0. Pathways achieved nDCG@5 of 1.0 versus 0.808 for fit-only.
- For `Mandarin meals Scarborough`, both systems achieved Precision@5 of 0.333, nDCG@5 of 1.0, MRR of 1.0, and MAP of 1.0.

### 5.3 Interpretation

The smoke fixture indicates that freshness-aware ranking can improve graded ordering in at least one controlled case without changing first-relevant-result metrics. However, the two-case fixture is not thesis-quality evidence.

The result should be described as:

> In a two-case development smoke fixture, the freshness-aware Pathways ranker matched the fit-only baseline on Precision@5, MRR, and MAP, while achieving higher mean graded nDCG@5.

It should not be described as proof that Pathways generally outperforms BM25 or a fit-only ranker. A larger independently adjudicated relevance corpus, BM25 baseline, repeated queries, and confidence intervals are still required.

## 6. Study 4: Sealed Missing-Data Benchmark

### 6.1 Benchmark design

The benchmark contains 500 requested missing fields sampled with a fixed seed. Each row identifies an organization and location context while withholding one field such as address, phone, website, or postal code.

The gold fixture stores the expected value from the source dataset. The ARS prediction process receives only the public fixture and live/provider-derived evidence. The scorer evaluates only the requested missing field for each row.

The scorer records:

- `filled`: a non-empty candidate value was returned.
- `entity_supported`: the returned record passed the fuzzy entity-support rule.
- `exact`: the returned value matched the gold value after field-specific normalization.
- `wrong_entity`: a non-empty candidate failed entity support.
- `unresolved`: no value was returned.
- `unexpected_fills`: unrelated withheld fields were filled.

The entity-support matcher uses weighted name, address, city, postal-code, and website-domain similarity. Support requires a weighted score of at least 0.72, name similarity of at least 0.60, and city similarity of at least 0.70.

### 6.2 Conditions

Three conceptual conditions were examined:

1. **Original first-result Places baseline:** useful for diagnosing the wrong-entity problem but contaminated by identity leakage and unsuitable for accurate entity-precision claims.
2. **Fuzzy Places v2:** Google Places Text Search with fuzzy matching and abstention.
3. **Hybrid:** starts with fuzzy Places v2 and sends unresolved rows through the original bounded ARS pipeline, including query expansion, web search, scraping, and extraction.

### 6.3 Places-only fuzzy condition

| Metric | Fuzzy Places v2 |
|---|---:|
| Requested fields | 500 |
| Filled | 340 |
| Entity-supported | 340 |
| Exact | 84 |
| Unresolved | 160 |
| Wrong entity | 0 |
| Unexpected fills | 0 |
| Fill rate | 68.0% |
| Exact rate | 16.8% |
| Wrong-entity rate | 0.0% |

The fuzzy matcher intentionally abstained when the first Places result did not meet the entity-support threshold. This reduced coverage but eliminated the wrong-entity results observed in the original first-result condition.

### 6.4 Hybrid condition

| Metric | Hybrid ARS fallback |
|---|---:|
| Requested fields | 500 |
| Filled | 499 |
| Entity-supported | 499 |
| Exact | 142 |
| Unresolved | 1 |
| Wrong entity | 0 |
| Unexpected fills | 0 |
| Fill rate | 99.8% |
| Exact rate | 28.4% |
| Wrong-entity rate | 0.0% |

The hybrid runner processed the unresolved Places rows individually in isolated subprocesses. The merged output contains all 500 benchmark rows. One row remained unresolved:

- `Mental Health and Addictions Program - Humber River Health`, Toronto, Ontario, postal code `M3M 0B2`.
- Requested field: phone number.

The ARS did not return a sufficiently reliable phone value for this row and correctly left it unresolved rather than inventing one.

### 6.5 Change from Places-only to hybrid

| Metric | Fuzzy Places v2 | Hybrid | Change |
|---|---:|---:|---:|
| Filled | 340 | 499 | +159 |
| Exact | 84 | 142 | +58 |
| Unresolved | 160 | 1 | -159 |
| Wrong entity | 0 | 0 | 0 |
| Unexpected fills | 0 | 0 | 0 |

The hybrid fallback increased validated coverage by 31.8 percentage points and exact match rate by 11.6 percentage points, while preserving a zero wrong-entity rate under the automated evaluator.

## 7. Analysis of the 357 Non-Exact but Entity-Supported Values

### 7.1 Why exact and validated are different

A value can be entity-supported without matching the historical gold string. For example, a current official website may differ from an old directory URL, or a phone number may include an extension that was absent from the source dataset.

Therefore:

```text
499 entity-supported fills = 142 exact matches + 357 non-exact matches
```

The 357 are not automatically correct. They are values associated with an entity that the matcher judged compatible with the benchmark row.

### 7.2 Mismatch categories

| Field | Category | Count |
|---|---|---:|
| Address | Near-equivalent wording | 53 |
| Address | Same street number, expanded or abbreviated | 91 |
| Address | Same street number, partial overlap | 57 |
| Address | Materially different address text | 57 |
| Website | Same domain, different URL path | 3 |
| Website | Different domain | 56 |
| Phone | Same base number, extension or format difference | 8 |
| Phone | Same area code, different number | 18 |
| Phone | Different number or area code | 8 |
| Postal code | Different postal code | 6 |
| **Total** |  | **357** |

### 7.3 Review tiers

The mismatch audit assigns review tiers using transparent field-specific heuristics:

| Tier | Count | Meaning |
|---|---:|---|
| High-confidence representation | 155 | Likely formatting, expansion, abbreviation, extension, or URL-path variation |
| Likely valid alternate | 75 | Plausibly current or alternate information, but not equivalent enough to accept automatically |
| Needs manual verification | 127 | Material address, domain, phone, or postal-code difference |

The high-confidence tier includes near-equivalent addresses, same-street-number address expansions, same-domain website paths, and phone extension/format variations. The likely-alternate tier includes partial address matches and same-area-code phone changes. The manual tier includes materially different address text, different website domains, different phone numbers or area codes, and different postal codes.

### 7.4 Interpretation and risk

The audit makes the coverage result more informative, but it does not convert the 357 values into verified truth. The matcher can give a high entity score when name and city agree even if the address differs substantially. This is especially important for organizations with multiple branches, hospitals with several locations, mailing addresses, administrative offices, and directory pages.

The most defensible statement is:

> The hybrid system produced 499 entity-supported values, of which 142 matched the sealed historical gold values exactly. The remaining 357 values were associated with supported entities but differed from the historical values; automated triage classified 155 as likely representation differences, 75 as likely alternates, and 127 for manual verification.

## 8. Provider and ARS Operational Tests

The following provider behaviors were observed during live integration checks:

| Component | Result | Observation |
|---|---|---|
| Gemini | Working | Query generation and extraction completed in the ARS pipeline after credits were configured |
| Tavily | Working | Used successfully as a search fallback |
| Google Places | Working | Text Search returned names, addresses, phones, websites, hours, Maps URLs, and place IDs |
| SearXNG | Working locally | SearXNG responded, but upstream engines were intermittently rate-limited or CAPTCHA-blocked |
| Google Custom Search JSON API | Not available | New-customer access returned HTTP 403; it was not used as a dependable condition |
| Crawl4AI | Working after repair | Chromium/dependencies were installed; overlay removal was disabled because it reduced extracted content to a few characters |

The ARS pipeline was bounded by query, search, scraping, and extraction limits. This reduced cost and latency but means the experiment measures the configured bounded pipeline, not unlimited web research.

## 9. Earlier Scraper Failure Study

An earlier real-URL scraper test evaluated 12 URLs from hospital-related search results:

| Result | Count | Rate |
|---|---:|---:|
| Successful HTML pages | 3 | 25.0% |
| Failed or unusable URLs | 9 | 75.0% |
| PDFs detected | 2 | 16.7% |
| Combined HTML/PDF content recovery | 5 | 41.7% |

The observed failure categories were:

- DNS resolution failures: 4
- PDF downloads requiring a separate processor: 2
- Timeout: 1
- Connection reset: 1
- SSL certificate error: 1

The test motivated DNS validation, PDF routing, retries, longer timeouts, rate limiting, and caching. It should be treated as an engineering diagnostic sample, not as a general estimate of web-crawling success.

## 10. What the Results Support

The completed evidence supports the following claims:

1. The Pathways implementation has a functioning automated test surface; the current suite passes 26 tests.
2. The ARS can retrieve and restore missing resource fields using Places and web-search fallback providers.
3. Fuzzy entity matching with abstention materially reduces wrong-entity risk compared with blindly accepting the first Places result.
4. A hybrid strategy substantially increases coverage over Places-only retrieval.
5. The hybrid condition achieved 499/500 entity-supported fills and 0/500 evaluated wrong-entity fills on this sealed benchmark.
6. Exact historical-value agreement is lower than entity-supported coverage and must be reported separately.
7. Freshness-aware ranking improves graded ordering in at least one case in the development smoke fixture.

## 11. What the Results Do Not Support

The results do not establish:

- That all 499 entity-supported values are factually current.
- That the system is clinically safe or ready for unsupervised production use.
- That the hybrid pipeline generally outperforms all alternative search systems.
- That Pathways generally outperforms BM25 or other rankers.
- That the 127 manual-review tier contains only incorrect values.
- That the 155 high-confidence representation tier is fully correct without source verification.
- General web-scraping success rates from the 12-URL diagnostic test.
- Statistical significance or confidence intervals for the ranking smoke test.

The benchmark is based on historical source values. A current provider value can be more useful operationally while still being counted as a mismatch against the historical gold fixture. Conversely, a value can pass automated entity matching while being outdated or associated with the wrong branch.

## 12. Recommended Thesis Presentation

The Study 4 results should be presented as separate conditions rather than a single headline number:

| Condition | Coverage | Exact agreement | Wrong entity |
|---|---:|---:|---:|
| Fuzzy Places v2 | 68.0% | 16.8% | 0.0% |
| Hybrid ARS fallback | 99.8% | 28.4% | 0.0% |

The paper should report both exact agreement and entity-supported coverage. A suggested wording is:

> On a sealed 500-field missing-data benchmark, fuzzy Google Places retrieval produced 340 entity-supported fills, including 84 exact matches, with no wrong-entity results under the automated matcher. Applying the bounded ARS pipeline to unresolved rows increased entity-supported coverage to 499 fields and exact matches to 142, again with no wrong-entity results. Of the 357 non-exact but entity-supported values, automated audit heuristics classified 155 as likely representation differences, 75 as likely valid alternates, and 127 as requiring manual verification. These classifications are triage signals rather than adjudicated correctness labels.

For Study 2, the paper should use cautious wording:

> In a two-case development smoke fixture, Pathways matched the fit-only baseline on Precision@5, MRR, and MAP and achieved higher graded nDCG@5. The fixture is too small and hand-authored to support a general performance claim.

## 13. Reproduction Commands

Run the full Pathways test suite:

```bash
cd /home/Kageshi/Documents/Projects/Automated-Research-System
PYTHONPATH=.:Pathways /home/Kageshi/anaconda3/envs/auto-research/bin/python -m pytest Pathways -q
```

Run the ranking smoke evaluator:

```bash
PYTHONPATH=.:Pathways python Pathways/evaluation/evaluate_ranking.py \
  --output Pathways/evaluation/ranking_report.json
```

Score the fuzzy Places condition:

```bash
PYTHONPATH=.:Pathways python Pathways/evaluation/missing_data_benchmark.py score \
  --public Pathways/evaluation/missing_data_benchmark/missing_data_public.jsonl \
  --gold Pathways/evaluation/missing_data_benchmark/missing_data_gold.jsonl \
  --predictions Pathways/evaluation/missing_data_benchmark/predictions_places_fuzzy_v2.jsonl \
  --output /tmp/report_places_fuzzy.json
```

Score the hybrid condition:

```bash
PYTHONPATH=.:Pathways python Pathways/evaluation/missing_data_benchmark.py score \
  --public Pathways/evaluation/missing_data_benchmark/missing_data_public.jsonl \
  --gold Pathways/evaluation/missing_data_benchmark/missing_data_gold.jsonl \
  --predictions Pathways/evaluation/missing_data_benchmark/predictions_hybrid.jsonl \
  --output Pathways/evaluation/missing_data_benchmark/report_hybrid.json
```

Regenerate the mismatch audit:

```bash
PYTHONPATH=.:Pathways python Pathways/evaluation/analyze_missing_data_mismatches.py \
  --report Pathways/evaluation/missing_data_benchmark/report_hybrid.json \
  --gold Pathways/evaluation/missing_data_benchmark/missing_data_gold.jsonl \
  --output Pathways/evaluation/missing_data_benchmark/mismatch_audit.json
```

## 14. Remaining Evaluation Work

The following work remains before the evaluation can be described as complete:

1. Build a larger independently adjudicated ranking corpus and add a BM25 baseline.
2. Repeat ranking experiments across queries, seeds, and time conditions.
3. Run the planned flywheel simulation at multiple time points.
4. Aggregate observability events and report verification and time-to-resolution metrics.
5. Manually adjudicate a stratified sample of the 357 mismatches, prioritizing the 127 manual-review cases.
6. Compare current provider evidence against historical gold values to distinguish stale gold data from incorrect predictions.
7. Add confidence intervals and, where appropriate, significance tests.
8. Report provider cost, latency, rate limits, and failure rates for the hybrid experiment.
9. Preserve the benchmark public fixture hash and provider configuration for the final paper appendix.

Until these steps are complete, the current results should be presented as a strong engineering and benchmark evaluation of the ARS-assisted workflow, not as a complete clinical or population-level validation.
