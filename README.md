# Automated Research System (ARS)

**AI-Driven Information Gathering & Agentic Curation of Community Health Services**

## Project Overview

Traditional social service directories (like Ontario 211) suffer from incomplete coverage and missing granular community programs due to the bottleneck of manual curation.

The **Automated Research System (ARS)** is an autonomous, domain-agnostic information-gathering pipeline that uses Large Language Models (LLMs) and advanced search strategies to dynamically scrape, map, and structurally verify community health data across Ontario (e.g., Addiction Counseling, Mental Health Clinics).

* **Author:** Kevin Igweh
* **Program:** Master of Science in Computer Science (MSc)
* **Institution:** Algoma University

## System Architecture & Data Flow

The core extraction pipeline is built on four sequential computational nodes:

### Node 1: Query Expansion Matrix (QEM)

To prevent "Generative Convergence" (LLMs defaulting to average search phrases like *"Hospitals in Ontario"*), the QEM uses a combinatorial formula to enforce structural variance:
`Query = Entity + Scope + Attribute + Source`
The system permutes these variables into specific strategies (*The Broad Net*, *The Deep Dive*, *The Artifact Hunter*) to sweep both high-level domains and hidden unstructured files (PDFs/CSVs).

### Node 2: Web Scraping & Conversion (Crawl4AI)

The system navigates to generated URL targets and utilizes `Crawl4AI` to scrape raw HTML, subsequently converting the unstructured web data into clean, structured Markdown for LLM processing.

### Node 3: Chunking & LLM Extraction Engine

Markdown documents are split into processable context chunks. Strict alignment prompting is applied to parse the text into structured JSON arrays, while actively filtering out out-of-scope data (e.g., private for-profit entities).

### Node 4: Multi-Layer Deduplication (Entity Resolution)

A custom 5-layer deduplication pipeline resolves complex, overlapping entities:

1. **Hard Identifier Match:** Exact matches on rigid columns (e.g., Phone Numbers).
2. **Vector Similarity Index:** Soft identifiers (Name + Address) are embedded into 384-D vectors using `all-MiniLM-L6-v2`. Pairs with a cosine similarity > `0.8` are flagged as candidates.
3. **Hybrid String Math:** Applies `Jaro-Winkler` (prefix matching) and `TokenSetRatio` (word swaps) per column to calculate string similarity.
4. **Threshold Decision Routing:** Auto-merges duplicates (>90%), auto-separates distinct entities (<60%), and routes the "Gray Zone" (60-89%) to Layer 5.
5. **LLM Context Judge:** An LLM acts as the final arbiter for ambiguous gray-zone pairs, applying human-level contextual reasoning.

## Automated Verification Pipeline

To evaluate the system without relying on brittle UI automation or unscalable manual validation, we engineered a programmatic verification pipeline against a manually curated Ground Truth (GT) of 403 samples.

1. **SearXNG Database Querying:** Locally hosted SearXNG queries official databases (e.g., `site:211ontario.ca "Generated Name"`) to retrieve JSON-formatted search snippets.
2. **Matching Algorithm:** The deduplication logic (Layer 3) is reverse-engineered to compare the `SearXNG Result Title` against the `Generated Entity Name`.

### Current Performance Metrics (Addiction Counseling)

* **Generated Samples:** 573
* **Successfully Validated:** 266 (~46.5% auto-verified against official registries)
* **Precision:** 80.80%
* **Recall:** 49.67%
* **F1-Score:** 61.51%

**Breakthrough:** The algorithm successfully extracted over 450 unique, valid entities absent from the official Ground Truth, proving its capacity to outperform manual curation in hyper-local discovery.

## Future Work

* **Domain Adaptation:** Prove the algorithm is domain-agnostic by adapting the extraction logic to entirely new taxonomies of social services (e.g., Food Banks, Disability Services).
* **Semantic Web Verification:** Pivot from reliance on Google Places tags (which struggle with ontological classification like `point_of_interest`) to an **LLM-driven Search Snippet Classifier**. This will analyze open-web meta descriptions to accurately verify domain alignment (e.g., distinguishing a clinical health service from a commercial retail store).

## Installation & Setup

*(Add specific instructions for setting up the Python environment, `.env` API keys for the LLM, Crawl4AI requirements, and the SearXNG docker container here).*

```bash
# Clone the repository
git clone [https://github.com/yourusername/automated-research-system.git](https://github.com/yourusername/automated-research-system.git)
cd automated-research-system

# Install dependencies
pip install -r requirements.txt
