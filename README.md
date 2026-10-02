# Automated Research System and Pathways

This repository contains two connected systems for discovering, verifying, and maintaining community healthcare-resource information:

- **Automated Research System (ARS):** a Python web-research server that expands search queries, collects open-web evidence, extracts structured records, resolves duplicate entities, and supports verification.
- **Pathways:** a browser-based directory-maintenance prototype that exposes verified resources through a Find--Verify--Ask--Close workflow and records maintenance events in the browser.

The project is a research prototype. It is intended for reproducible experimentation and human-assisted information work, not autonomous clinical decision-making or a production referral service.

## Repository Structure

```text
.
├── server.py                 # ARS Flask/Socket.IO server
├── algorithm.py              # ARS discovery and processing logic
├── dataset_builder.py        # Dataset construction utilities
├── vector_store.py           # Optional vector-store integration
├── nodes/                    # ARS pipeline nodes
├── DSPy/                     # Optional DSPy programs and services
├── Pathways/                 # Pathways browser prototype and evaluation code
├── datasets/                 # Current dataset inputs and outputs
├── datasets_history/         # Historical dataset snapshots
├── logs/                     # Run logs
├── templates/                # ARS server templates
├── test/                     # Root-level tests
├── requirements.txt          # Python dependencies
├── quickstart.sh             # Interactive ARS setup/start helper
└── Thesis/ and Journal/      # Publication manuscript sources
```

## System Overview

### ARS Discovery Pipeline

ARS separates search-space construction from language-model interpretation:

1. **Query Expansion Matrix (QEM):** enumerates combinations of entity type, geographic scope, service attribute, and source type.
2. **Web collection and conversion:** retrieves candidate pages and converts content into a form suitable for structured processing.
3. **Schema-constrained extraction:** extracts organization, service, location, contact, and operational fields while preserving source evidence.
4. **Entity resolution:** combines exact identifiers, normalized fields, similarity methods, and contextual review to identify duplicates or ambiguous records.
5. **Evidence verification:** routes records through registry matching, domain-scoped search, provider signals, and manual review.

The ARS server is implemented in `server.py` and exposes the research workflow through Flask and Flask-SocketIO.

### Pathways Maintenance Workflow

Pathways turns the resulting resource inventory into a human-mediated maintenance system:

- **Find:** search resources using structured filters and transparent result information.
- **Verify:** confirm fields that appear current.
- **Ask:** report stale or missing information and route it for restoration.
- **Close:** review and accept, reject, or leave a proposed correction unresolved.

Pathways uses an append-only browser event ledger and a derived projection. The prototype records field verification, issue reports, restoration requests, replies, and accepted corrections in browser `localStorage`.

## Requirements

- Python 3.10 or newer is recommended.
- A virtual environment is recommended for Python dependencies.
- ARS features may require external services or credentials, depending on the workflow.
- Pathways runs as a static browser application without PostgreSQL or an LLM provider.

## Installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

## Run the ARS Server

The ARS runs through `server.py`:

```bash
cd /home/Kageshi/Documents/Projects/Automated-Research-System
python server.py
```

Open `http://127.0.0.1:8000/` in a browser.

The port can be changed with `PORT`:

```bash
PORT=8080 python server.py
```

The server may report a vector-store connection warning when PostgreSQL credentials or the database service are unavailable. The web server can still start for workflows that do not require the vector store, but vector-backed features will not be available until the database configuration is corrected.

### ARS Environment Variables

| Variable | Purpose |
|---|---|
| `PORT` | HTTP server port; defaults to `8000`. |
| `GEMINI_API_KEY` | API key for the real Gemini extraction fallback. |
| `GEMINI_OPENAI_BASE_URL` | OpenAI-compatible Gemini endpoint. |
| `GEMINI_MODEL` | Gemini model used by the evaluator or fallback. |
| `SEARXNG_URL` | SearXNG endpoint for search verification; commonly `http://localhost:8888`. |

Never commit API keys, database passwords, or other secrets.

### Optional ARS Quickstart

```bash
cd /home/Kageshi/Documents/Projects/Automated-Research-System
./quickstart.sh
```

The helper is interactive. For reproducible runs, starting `python server.py` directly is clearer.

## Run Pathways

Pathways is served as a static application from `Pathways`:

```bash
cd /home/Kageshi/Documents/Projects/Automated-Research-System
python -m http.server 8765 --directory Pathways
```

Open `http://127.0.0.1:8765/` in a browser.

The prototype uses no npm or Vite build. Its browser event ledger is stored under `pathways.event-ledger.v1` in `localStorage`. Clear local site data to reset prototype state.

## Import Pathways Data

From the repository root:

```bash
python Pathways/import_pathways_data.py \
  --input-dir Pathways \
  --output-dir Pathways/data/processed
```

The importer writes normalized resources, duplicate candidates, and an import report without deleting or merging source files.

## Tests

Focused Pathways domain tests:

```bash
python -m pytest Pathways/test_pathways_domain.py -q
```

Broader Pathways test set:

```bash
python -m pytest Pathways -q
```

Study 1 and current Study 2 evaluation tests:

```bash
PYTHONPATH=.:Pathways python -m pytest \
  Pathways/test_study1_extraction.py \
  Pathways/test_study2.py \
  Pathways/test_study2_adjudication.py \
  -q
```

Some historical tests and scripts refer to earlier experimental studies. Check the corresponding evaluation file before treating those artifacts as part of the current journal manuscript.

## Current Evaluation Evidence

### Study 1: Filter Extraction

Study 1 uses 200 structured requests: ten filter families crossed with ten Ontario locations, with canonical and paraphrased variants. It compares basic rules, aliases, semantic matching, constrained Gemini fallback, and the combined semantic-plus-Gemini pipeline. The principal metrics are exact-case accuracy and micro-averaged precision, recall, and F1.

These are development-fixture results because the ontology and query families were available during implementation. They should not be presented as held-out estimates of open-ended language understanding.

### Study 2: Missing-Data Restoration

Study 2 uses a sealed 500-field benchmark containing addresses, phone numbers, postal codes, and websites. Results are classified as Exact, Supported non-exact, Weakly supported, or Unresolved. The benchmark also includes regional and service-category breakdowns and a mismatch audit covering representation drift, likely valid alternatives, ambiguous evidence, and malformed extraction outputs.

## Data, Code, and Reproducibility

Important evaluation artifacts are located under `Pathways/evaluation/`, including:

- Study 1 fixtures and extraction reports
- Missing-data public and gold fixtures
- Hybrid restoration reports
- Entity-matching utilities
- Mismatch audits and manual-review documentation
- Simulation and historical evaluation scripts

The thesis source is under `Thesis/`, and the journal manuscript is under `Journal/`. For reproducible work, record the Python version, dependency versions, model/provider configuration, fixture paths, search/provider availability, random seeds, and dates of external-source retrieval.

## Limitations and Responsible Use

ARS and Pathways process public organizational information and are not clinical decision-support systems. They do not guarantee complete geographic coverage, current operational status, or correctness of every source-derived field.

Known limitations include shallow or source-dependent web coverage, external provider dependence, development-fixture evidence for Study 1, manual judgment in Study 2, geographic imbalance in the restoration benchmark, source drift, and the lack of live clinical deployment or user-adoption evidence. Human review is required before using restored information in a trusted referral workflow.

## Documentation Map

| Document | Purpose |
|---|---|
| `README.md` | This combined setup and usage guide. |
| `ARCHITECTURE.md` | System architecture notes. |
| `QUICK_REFERENCE.md` | ARS operational reference. |
| `EVALUATION_RESULTS.md` | Evaluation-result summaries and caveats. |
| `Pathways/README.md` | Pathways implementation and workflow notes. |
| `Pathways/EVALUATION.md` | Pathways evaluation design and limits. |
| `Thesis/Thesis.tex` | Thesis source. |
| `Journal/Journal.tex` | MDPI Computers article draft. |

## License and Publication Status

This repository is a research workspace. Check licensing and data-use terms before redistributing code, datasets, screenshots, or provider-derived outputs. The thesis and journal manuscript are draft research documents and should not be treated as published clinical guidance.
