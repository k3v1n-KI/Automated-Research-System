# Pathways feature backlog

This is the implementation order for rebuilding Pathways from the thesis, recovered design studies, and the three supplied datasets. Each phase should be completed and tested before the next one begins.

## Priority order

### M0. Data foundation and clean import
**Goal:** Turn the three CSV datasets into one normalized resource corpus without losing provenance.

Features:

- Import Addiction, Hospital, and Pharmacy records into a common resource schema.
- Preserve source dataset, source row number, and `source_url` for every record.
- Normalize names, phone numbers, websites, addresses, cities, postal codes, and categories.
- Detect duplicate candidates using name + address + phone/website evidence.
- Keep an explicit duplicate decision history rather than silently deleting rows.
- Add deterministic seed/import commands and an import report.
- Store field-level provenance and initial `ai-only`/`unknown` verification state.

Dataset notes:

- Addiction: 540 rows, 5 columns, no missing values, 1 duplicate normalized name.
- Hospitals: 801 rows, 4 columns, no missing values, 236 duplicate normalized names. This requires the strongest deduplication review.
- Pharmacy: 984 rows, 7 columns, no missing values, 3 duplicate normalized names.

Exit criteria: repeatable import, stable resource IDs, provenance retained, duplicate report generated, and no source row silently discarded.

### M1. Event ledger and field state — implemented
**Goal:** Make directory state derived from immutable events.

Features:

- Append-only events with actor, display identity, timestamp, source/channel, and payload.
- Resource fields: name, category, address, city, phone, website, postal code, hours, eligibility, fees, referral path, accessibility, languages, and wait time.
- Field states: `verified-fresh`, `verified-aging`, `ai-only`, `flagged-stale`, and `unknown`.
- Deterministic projection/replay from the event ledger.
- Configurable freshness decay with a seedable clock.
- Chain-of-custody view for every field.
- No event mutation or deletion.

Implemented in `pathways_domain.py` with fixed-clock tests in `test_pathways_domain.py`.
Exit criteria met: replaying the same event stream produces the same projection; freshness boundary tests pass.

### M2. Find: directory search and ranking — implemented
**Goal:** Make the imported corpus useful for real service discovery.

Features:

- Direct directory search by keyword, category, city/region, language, and postal code.
- Free-text case input.
- Structured filter extraction into editable chips: need, modality, population, coverage, location, language, and urgency.
- Keyword fallback when extraction is unavailable.
- Transparent ranking with separate Fit, Trust/Freshness, and stale-flag effects.
- Result rationale, verification aggregate, active flags, source links, and provenance.
- Search logging: query, extracted filters, returned IDs, scores, and click-throughs.
- Empty/weak-result path that offers to create an Ask or add a known resource.

Implemented in `pathways_find.py` with corpus-loader, filter-extraction, ranking, flag,
freshness, and search-log tests in `test_pathways_find.py`. Exit criteria met for the
deterministic search layer; BM25 and LLM extraction remain later evaluation extensions.

### M3. Verify: frictionless field validation — implemented
**Goal:** Let navigators improve records during normal use.

Features:

- Resource detail page with field-by-field state badges.
- One-click Confirm action.
- Flag action with reason: stale, wrong, closed, or unsure.
- Optional correction text and source attachment.
- Anonymous presentation mode with internal attribution retained.
- Optimistic UI with durable event append.
- Verification changes ranking freshness and field state.

Implemented in the domain projection and browser resource detail view. Exit criteria met:
confirm/flag actions update the projection, field state, custody history, and ranking inputs.

### M4. Ask: regional knowledge network — implemented
**Goal:** Recover missing or uncertain services through community input.

Features:

- Create an Ask from free text and tags.
- Region and expiry rules: urban and rural-extended windows.
- Reply to an Ask.
- Attach an existing resource to a reply.
- Mention a not-yet-indexed service and create a candidate resource.
- Watch an Ask and notify the asker/watchers.
- Resolve an Ask and preserve the resolution event.

Implemented in `pathways_domain.py` and the browser Ask board. Exit criteria met:
Ask → reply → candidate/resource link → resolved state is tested end to end, including
urban/rural expiry behavior.

### M5. Close: cases, referrals, and the flywheel
**Goal:** Complete Find → Verify → Ask → Close.

Features:

- Create a case without patient identifiers.
- Save structured needs and search history to a case.
- Close with outcome: referred, self-managed, no-fit, or declined.
- Record referred resource when applicable.
- Resolve linked Ask and re-confirm the used resource atomically.
- Seed an off-index referral as a candidate resource.
- Display a contribution summary: fields confirmed, candidate seeded, and Ask resolved.

Exit criteria: a close action changes all linked states atomically and is replayable.

### M6. Accounts, regions, and moderation
**Goal:** Add controlled multi-user behavior.

Features:

- Authentication boundary and home OHT membership.
- Stable internal user ID separated from display identity.
- Coordinator role for flags, candidates, merges, and disputes.
- Candidate merge workflow.
- Abuse controls and verification rate limiting.
- Region update feed and actionable notifications.

Exit criteria: permissions and attribution are covered by integration tests.

### M7. Evaluation harnesses
**Goal:** Prepare the reconstructed system for thesis evaluation.

Features:

- Case-vignette and gold-filter corpus.
- Extraction metrics: precision, recall, F1, exact match.
- Ranking metrics: nDCG@5/@10, MRR, Precision@K, MAP, stale-result rate, trust-weighted nDCG.
- Baselines: keyword/BM25 and fit-only ranking.
- Ablation over freshness weights and half-life.
- Search latency and event persistence telemetry.
- Reproducible reports with confidence intervals.

Exit criteria: every metric can be rerun from versioned fixtures and saved event logs.

### M8. Flywheel simulation
**Goal:** Evaluate directory maintenance over time.

Features:

- Synthetic navigator personas and usage patterns.
- Seeded data decay and incorrect fields.
- Verification, flag, Ask, ARS suggestion, and Close events through the real API.
- Static-directory control condition.
- Time points at 0, 30, and 90 days plus participation sweeps.
- Accuracy, freshness, stale-result rate, mean time to correction, and participation threshold outputs.

Exit criteria: RQ3/RQ4 results can be reproduced from a simulation seed.

## Cross-cutting requirements

- No patient names or PHI in case text by design.
- Preserve source URLs and event provenance.
- Keep browser prototype, backend/domain code, datasets, design studies, and evaluation artifacts in separate directories.
- Prefer deterministic, framework-free domain logic for tests; adapters can be added around it.
- The static directory must remain searchable when extraction or ARS services are unavailable.
- UI must remain keyboard accessible and usable on narrow/low-bandwidth screens.

## Proposed clean layout

```text
Pathways/
  app/                 # runnable browser entry and UI assets
  domain/              # event ledger, projections, ranking, schemas
  data/raw/            # supplied CSV files, unchanged
  data/processed/      # normalized imports and generated reports
  evaluation/          # fixtures, runners, metrics, result reports
  design/              # HTML, JSX, wireframes, screenshots, design canvas
  docs/                # requirements, presentation, recovery notes
  tests/               # domain, integration, and browser tests
```

The current files are being kept intact while this backlog is established. Directory moves should happen as a separate mechanical cleanup step before M0 implementation so design-study paths and existing links are not broken accidentally.
