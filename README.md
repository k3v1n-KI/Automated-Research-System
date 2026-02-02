# Automated-Research-System

## How Plan & Aggregate Work (Without aggregate.py and plan_research_task.py)

### Architecture Overview

The pipeline works entirely through the **Services class** in task_dispatcher.py. Standalone files like aggregate.py and plan_research_task.py are **legacy/unused**.

```
main.py
  ↓
TaskDispatcher.run() → Services class
  ↓
build_graph(services) → lang_graph.py
  ↓
[plan_node → search_node → validate_node → scrape_node → extract_node → aggregate_node → ...]
```

Each node calls methods on the Services object, not standalone modules.

### Plan Node Flow
- **Services.plan()** calls LLM via DSPyHelpers to create `{"goal": ..., "queries": [...]}`
- **plan_node** (lang_graph.py:66) validates the plan
- **plan_research_task.py is unused** - planning is integrated into Services

### Aggregate Node Flow
- **aggregate_node** (lang_graph.py:393) calls `services.execute_aggregate(before, extracted)`
- **Services.execute_aggregate()** (task_dispatcher.py:739):
  - Tries `aggregate.consolidate_and_dedupe()` (always fails, module not available)
  - **Falls back to built-in dedupe**: combines items, dedupes by (name, address)
- **aggregate.py is unused** - built-in fallback handles aggregation

### Current Issue: Extract Getting 0 Items

The problem is in the extraction chain:

```
scrape_node → extract_node → 0 items → aggregate_node has nothing
```

**Debug steps:**
1. Check Firestore: `research_plans/{plan_id}/artifacts/scrapes_round_*` - are URLs scraped with text?
2. Check logs: `[extract_input] usable_for_extract: X` - is scrape content reaching extract?
3. Check logs: `[extract] extracted: Y` - is extraction finding hospitals?

If scrapes have content but extract returns 0, the issue is in extract.py (LLM or regex failing).