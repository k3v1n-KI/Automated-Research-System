# Dataset Builder - Architecture & Design

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         WEB BROWSER                                 │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │            chat_interface.html (Modern UI)                    │ │
│  │  ┌──────────────────┐            ┌──────────────────────────┐ │ │
│  │  │  Input Panel     │            │  Progress Panel          │ │ │
│  │  │                  │            │  ┌────────────────────┐  │ │ │
│  │  │ • Dataset desc   │  ━━━━━━━━> │  │ 🔍 Query Gen       │  │ │ │
│  │  │ • Start button   │  WebSocket │  │ 🔎 Search          │  │ │ │
│  │  └──────────────────┘            │  │ ✅ Validate        │  │ │ │
│  │                                  │  │ 📄 Scrape          │  │ │ │
│  │                                  │  │ 🎯 Extract         │  │ │ │
│  │                                  │  │ 🗑️  Deduplicate    │  │ │ │
│  │                                  │  └────────────────────┘  │ │ │
│  │                                  │  • Real-time updates     │ │ │
│  │                                  │  • Statistics            │ │ │
│  │                                  │  • Export buttons        │ │ │
│  │                                  └──────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                              ▲  ▼
                          WebSocket
                              ▲  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        FLASK SERVER                                 │
│                     (server.py)                                     │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Route: /        Renders HTML interface                       │ │
│  │  Route: /health  Health check endpoint                        │ │
│  │  WS: connect/disconnect  Connection management               │ │
│  │  WS: start_research  Launch algorithm                         │ │
│  │  WS: export_dataset  Download results                         │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              ▲  ▼                                    │
│                         Python asyncio                              │
│                              ▲  ▼                                    │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │              LangGraph Algorithm                              │ │
│  │                  (algorithm.py)                                │ │
│  │                                                                │ │
│  │  ResearchState (Dict-based)                                   │ │
│  │  ┌──────────────────────────────────────────────────────────┐ │ │
│  │  │ • initial_prompt                                         │ │ │
│  │  │ • queries → [4 strings]                                  │ │ │
│  │  │ • search_results → [60 URLs]                             │ │ │
│  │  │ • validated_urls → [45 URLs]                             │ │ │
│  │  │ • scraped_content → [40 HTML docs]                       │ │ │
│  │  │ • extracted_items → [120 records]                        │ │ │
│  │  │ • final_dataset → [95 unique records]                    │ │ │
│  │  │ • session_id, round, error                               │ │ │
│  │  └──────────────────────────────────────────────────────────┘ │ │
│  │                                                                │ │
│  │  Node Pipeline:                                                │ │
│  │  ┌───────────────────────────────────────────────────────┐   │ │
│  │  │  1. QUERY_GENERATION_NODE                            │   │ │
│  │  │     Input: initial_prompt                            │   │ │
│  │  │     Process: LLM generates 4 diverse queries          │   │ │
│  │  │     Output: queries = [4 strings]                     │   │ │
│  │  │     LLM: gpt-4o-mini, temp=0.7                        │   │ │
│  │  └───────────────────────────────────────────────────────┘   │ │
│  │                          ↓                                     │ │
│  │  ┌───────────────────────────────────────────────────────┐   │ │
│  │  │  2. SEARCH_NODE                                       │   │ │
│  │  │     Input: queries[4]                                 │   │ │
│  │  │     Process: SEARXNG + Google API (fallback)          │   │ │
│  │  │     Output: search_results = [60 URLs]                │   │ │
│  │  │     Per query: 15 results                             │   │ │
│  │  │     Deduplication: By URL                             │   │ │
│  │  └───────────────────────────────────────────────────────┘   │ │
│  │                          ↓                                     │ │
│  │  ┌───────────────────────────────────────────────────────┐   │ │
│  │  │  3. VALIDATE_NODE                                     │   │ │
│  │  │     Input: search_results[60]                         │   │ │
│  │  │     Process: Cosine similarity (SemanticRanker)       │   │ │
│  │  │     Threshold: 0.5                                    │   │ │
│  │  │     Output: validated_urls = [45 URLs]                │   │ │
│  │  │     Filter: Against source query                      │   │ │
│  │  └───────────────────────────────────────────────────────┘   │ │
│  │                          ↓                                     │ │
│  │  ┌───────────────────────────────────────────────────────┐   │ │
│  │  │  4. SCRAPE_NODE                                       │   │ │
│  │  │     Input: validated_urls[45]                         │   │ │
│  │  │     Process: Playwright browser automation            │   │ │
│  │  │     Timeout: 15 seconds per URL                       │   │ │
│  │  │     Output: scraped_content = [40 HTML docs]          │   │ │
│  │  │     Includes: HTML + extracted text                   │   │ │
│  │  └───────────────────────────────────────────────────────┘   │ │
│  │                          ↓                                     │ │
│  │  ┌───────────────────────────────────────────────────────┐   │ │
│  │  │  5. EXTRACT_NODE                                      │   │ │
│  │  │     Input: scraped_content[40]                        │   │ │
│  │  │     Process: LLM extracts structured data             │   │ │
│  │  │     Per doc: First 3000 characters                    │   │ │
│  │  │     Output: extracted_items = [120 records]           │   │ │
│  │  │     LLM: gpt-4o-mini, temp=0.3                        │   │ │
│  │  │     Prompt: Original task + columns                   │   │ │
│  │  └───────────────────────────────────────────────────────┘   │ │
│  │                          ↓                                     │ │
│  │  ┌───────────────────────────────────────────────────────┐   │ │
│  │  │  6. DEDUPLICATE_NODE                                  │   │ │
│  │  │     Input: extracted_items[120]                       │   │ │
│  │  │     Process: LLM identifies duplicates                │   │ │
│  │  │     Criteria: Same entity (name, location, etc)       │   │ │
│  │  │     Output: final_dataset = [95 unique records]       │   │ │
│  │  │     JSON format: [{col1, col2, col3, ...}]            │   │ │
│  │  │     LLM: gpt-4o-mini, temp=0.3                        │   │ │
│  │  └───────────────────────────────────────────────────────┘   │ │
│  │                                                                │ │
│  │  ProgressTracker:                                              │ │
│  │  • Logs to terminal with detailed formatting                  │ │
│  │  • Emits "progress" events to WebSocket                       │ │
│  │  • Tracks: step, detail, data                                 │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  External APIs:                                                   │
│  ├─ OpenAI API (gpt-4o-mini) ← LLM for all text processing       │
│  ├─ SEARXNG API ← Primary search                                  │
│  ├─ Google Custom Search ← Fallback search                        │
│  ├─ SemanticRanker ← Cosine similarity validation                │
│  └─ Playwright ← Web scraping                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow Diagram

```
User Input (Prompt)
       │
       ▼
╔════════════════════════════════════════════════════════════════╗
║  INPUT VALIDATION & STATE INITIALIZATION                       ║
║  • Parse prompt                                                ║
║  • Create ResearchState                                        ║
║  • Generate session_id                                         ║
╚════════════════════════════════════════════════════════════════╝
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ 1️⃣  QUERY GENERATION                                          │
│                                                               │
│ Input: initial_prompt (e.g., "Find hospitals in Ontario")    │
│ Process:                                                       │
│   LLM (gpt-4o-mini) analyzes prompt                          │
│   ✓ Identifies key entities                                  │
│   ✓ Extracts requirements (columns, filters)                │
│   ✓ Generates 4 diverse query phrasings                      │
│ Output: queries = [                                           │
│   "hospitals in Ontario",                                     │
│   "Ontario medical centers and facilities",                  │
│   "healthcare institutions Ontario Canada",                  │
│   "Ontario hospital directory list"                          │
│ ]                                                             │
│                                                               │
│ Progress Update: "Generated 4 diverse queries"               │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ 2️⃣  SEARCH                                                    │
│                                                               │
│ Input: queries[4]                                             │
│ Process:                                                       │
│   FOR each query:                                             │
│     ✓ SEARXNG query (primary)                                │
│       • Request: format=json, language=en, categories=general │
│       • Returns: title, url, content (snippet)               │
│       • Take: top 15 results                                 │
│     ✓ Google API query (fallback if < 15 results)            │
│       • Request: key, cx, q, num=10                          │
│       • Returns: title, link, snippet                        │
│   Deduplication:                                              │
│     ✓ By URL (set-based)                                     │
│     ✓ Preserve metadata (title, snippet)                     │
│ Output: search_results = [                                    │
│   {url, title, snippet, source_query},                        │
│   ... (up to 60 total)                                        │
│ ]                                                             │
│                                                               │
│ Progress Update: "Found 60 unique URLs"                      │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ 3️⃣  VALIDATE                                                 │
│                                                               │
│ Input: search_results[60]                                     │
│ Process:                                                       │
│   FOR each URL:                                               │
│     ✓ Extract text from title + snippet                      │
│     ✓ Compute cosine similarity vs source_query              │
│     ✓ IF similarity >= 0.5: KEEP                             │
│     ✓ ELSE: DISCARD                                          │
│ Output: validated_urls = [                                    │
│   "hospital1.com",                                            │
│   "hospital2.com",                                            │
│   ... (typically 45-50 URLs)                                  │
│ ]                                                             │
│                                                               │
│ Progress Update: "Validated 45/60 URLs (threshold: 0.5)"    │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ 4️⃣  SCRAPE                                                   │
│                                                               │
│ Input: validated_urls[45]                                     │
│ Process:                                                       │
│   FOR each URL:                                               │
│     ✓ Launch Playwright browser instance                     │
│     ✓ Navigate to URL (timeout: 15s)                         │
│     ✓ Extract: page.content() → HTML                         │
│     ✓ Extract: page.innerText → Clean text                   │
│     ✓ Close browser instance                                 │
│     ✗ Timeout/Error: Skip and log                            │
│ Output: scraped_content = [                                   │
│   {url, html, text, timestamp},                               │
│   ... (typically 40-42 successful)                            │
│ ]                                                             │
│                                                               │
│ Progress Update: "Scraped 40/45 URLs"                        │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ 5️⃣  EXTRACT                                                  │
│                                                               │
│ Input: scraped_content[40]                                    │
│ Process:                                                       │
│   FOR each document:                                           │
│     ✓ Truncate text to first 3000 characters                 │
│     ✓ Create LLM prompt:                                      │
│       • System: "Extract structured data based on task"      │
│       • User: "Extract from this content: [text]"            │
│       • Include: original prompt for context                 │
│       • Model: gpt-4o-mini, temp=0.3                         │
│     ✓ Parse JSON from response                               │
│     ✓ Add source_url to each record                          │
│     ✗ Parse error: Skip and continue                         │
│ Output: extracted_items = [                                   │
│   {Name, Address, Website, source_url},                       │
│   ... (typically 120-150 records)                             │
│ ]                                                             │
│                                                               │
│ Progress Update: "Extracted 120 records"                     │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ 6️⃣  DEDUPLICATE                                              │
│                                                               │
│ Input: extracted_items[120]                                   │
│ Process:                                                       │
│   ✓ Create LLM prompt:                                        │
│     • System: "Remove duplicates (same entity = duplicate)"  │
│     • User: "Remove duplicates from: [json of all items]"    │
│     • Model: gpt-4o-mini, temp=0.3                           │
│   ✓ Parse JSON array response                                │
│   ✓ Validate structure matches input columns                 │
│ Output: final_dataset = [                                     │
│   {Name, Address, Website},                                   │
│   ... (typically 95-110 unique records)                       │
│ ]                                                             │
│                                                               │
│ Progress Update: "Final dataset: 95 unique records"          │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
╔════════════════════════════════════════════════════════════════╗
║  RESULTS DELIVERY                                              ║
║  • Emit "research_complete" WebSocket event                   ║
║  • Include: final_dataset + statistics                        ║
║  • Frontend: Display results & export options                 ║
║  • User can: Download JSON/CSV or request modifications       ║
╚════════════════════════════════════════════════════════════════╝
```

## State Progression

```
┌──────────────────────────────────────────────────────────────────┐
│ State Initialization                                             │
│ {                                                                │
│   initial_prompt: "Find hospitals in Ontario...",               │
│   queries: [],                                                   │
│   search_results: [],                                            │
│   validated_urls: [],                                            │
│   scraped_content: [],                                           │
│   extracted_items: [],                                           │
│   final_dataset: [],                                             │
│   session_id: "uuid",                                            │
│   round: 0,                                                      │
│   error: null                                                    │
│ }                                                                │
└──────────────────────────────────────────────────────────────────┘
                            ↓
                After Node 1 (Query Gen)
┌──────────────────────────────────────────────────────────────────┐
│ {                                                                │
│   ...,                                                           │
│   queries: ["query1", "query2", "query3", "query4"],           │
│   ...                                                            │
│ }                                                                │
└──────────────────────────────────────────────────────────────────┘
                            ↓
                After Node 2 (Search)
┌──────────────────────────────────────────────────────────────────┐
│ {                                                                │
│   ...,                                                           │
│   search_results: [{url, title, snippet, source_query}, ...],  │
│   ...                                                            │
│ }                                                                │
└──────────────────────────────────────────────────────────────────┘
                            ↓
                After Node 3 (Validate)
┌──────────────────────────────────────────────────────────────────┐
│ {                                                                │
│   ...,                                                           │
│   validated_urls: ["url1", "url2", ...],                        │
│   ...                                                            │
│ }                                                                │
└──────────────────────────────────────────────────────────────────┘
                            ↓
                After Node 4 (Scrape)
┌──────────────────────────────────────────────────────────────────┐
│ {                                                                │
│   ...,                                                           │
│   scraped_content: [{url, html, text, timestamp}, ...],        │
│   ...                                                            │
│ }                                                                │
└──────────────────────────────────────────────────────────────────┘
                            ↓
                After Node 5 (Extract)
┌──────────────────────────────────────────────────────────────────┐
│ {                                                                │
│   ...,                                                           │
│   extracted_items: [{Name, Address, Website, source_url}, ...],│
│   ...                                                            │
│ }                                                                │
└──────────────────────────────────────────────────────────────────┘
                            ↓
                After Node 6 (Deduplicate)
┌──────────────────────────────────────────────────────────────────┐
│ {                                                                │
│   ...,                                                           │
│   final_dataset: [{Name, Address, Website}, ...],              │
│   ...                                                            │
│ }                                                                │
└──────────────────────────────────────────────────────────────────┘
```

## Component Interactions

```
┌────────────────┐
│  HTML Client   │
│  (Browser)     │
└────────┬───────┘
         │ WebSocket
         │ emit("start_research", {prompt})
         │
         ▼
┌────────────────────────────┐
│  Flask Server              │
│  handle_start_research()   │
└────────┬───────────────────┘
         │
         ├─ Create ResearchState
         ├─ build_research_algorithm(emit_fn)
         ├─ Store session[session_id]
         │
         ▼
┌────────────────────────────┐
│  Background Task           │
│  _run_algorithm_task()     │
└────────┬───────────────────┘
         │
         ├─ algorithm_graph.ainvoke(state)
         │
         ├─┐ Each Node Runs:
         │ ├─ Node calls progress.update()
         │ ├─ progress.update() calls emit_fn()
         │ ├─ emit_fn() does: socketio.emit("progress", data)
         │ └─ Browser receives and updates UI
         │
         ▼
    Final State
         │
         └─ socketio.emit("research_complete", {...})
            │
            ▼
         ┌────────────────┐
         │  Browser UI    │
         │  Shows Results │
         └────────────────┘
```

This architecture ensures:
- **Real-time feedback** via WebSocket events
- **Clean separation** between server logic and UI
- **Scalable design** - can run multiple sessions
- **Comprehensive logging** - Terminal + Frontend
