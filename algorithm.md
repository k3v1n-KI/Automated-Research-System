## 1. System Components

1. **Orchestrator**

   * Coordinates the entire loop
   * Maintains the conversation context and overall state
2. **Planner (GPT-4)**

   * Inputs: Current state, goal, past findings
   * Outputs: JSON “plan” of ordered subtasks
3. **Task Dispatcher & Executor (MCP Servers)**

   * Pulls subtasks from a queue
   * Runs code modules to perform searches, API calls, scraping, validation
4. **Result Store**

   * A database (e.g. PostgreSQL, MongoDB) holding raw and structured outputs
5. **Validator**

   * Cross-references new findings with ground truth (e.g. your hospital list)
   * Applies heuristics (dedupe, confidence scoring)
6. **Aggregator & Reporter**

   * Summarizes validated results and feeds them back to the Planner

---

## 2. Data Flow & Iteration Loop

```text
User Goal
   ↓
Orchestrator → Planner (GPT-4) → Plan JSON
   ↓
Dispatcher enqueues subtasks → MCP Workers pick up subtasks
   ↓
Each Worker executes:
   • Search modules (Google CSE, Bing, Brave…)
   • API modules (Maps, Twitter, Reddit…)
   • Scrape modules (Playwright/Selenium)
   • Validation modules (cross-check + heuristics)
   ↓
Store raw + cleaned results in DB
   ↓
Aggregator builds summary (e.g. “Found 42 unique hospitals; 30 matched ground truth; 12 new”)
   ↓
Orchestrator sends summary back to Planner (GPT-4)
   ↓
Planner decides next subtasks or STOP
   ↺ Repeat until complete
```

---

## 3. Example “Plan” JSON Schema

When you call GPT-4 to plan, have it return something like:

```jsonc
{
  "goal": "List of hospitals in Ontario",
  "subtasks": [
    {
      "id": "q1",
      "type": "search",
      "description": "Search 'site:ontario.ca hospitals in Ontario list'",
      "params": {"query":"site:ontario.ca hospitals in Ontario list"}
    },
    {
      "id": "q2",
      "type": "api",
      "description": "Call Google Maps API for 'hospitals in Ontario'",
      "params": {"location":"Ontario","type":"hospital","radius":50000}
    },
    {
      "id": "q3",
      "type": "validate",
      "description": "Cross-reference q1+q2 results against existing spreadsheet",
      "params": {"source_tables":["db.raw_q1","db.raw_q2"],"ground_truth":"db.hosp_list"}
    }
    // …
  ]
}
```

---

## 4. MCP Server Task Execution

* **Containerize** each module (e.g. `search`, `api`, `scrape`, `validate`)
* **Message Queue** (RabbitMQ/Kafka) holds subtask messages
* **Worker Pool**: MCP servers subscribe to queue, pick tasks, execute, push results to DB
* Use **Kubernetes** or a simple **supervisor** to auto-scale workers based on queue length

---

## 5. Validation Heuristics

For each new candidate entry:

1. **Exact match** vs. ground truth → score 0/1
2. **Fuzzy match** (Levenshtein) → score 0–1
3. **Source trust** (.gov/.edu/.org vs. .com blogs) → weight 0.5–1
4. **Recency** (meta date within last 3 years) → weight 0.5–1
5. **Completeness** (name + city + address present) → weight 0–1
   → Combine into a confidence score; filter out below threshold

---

## 6. Orchestration Pseudocode

```python
while not planner.signals_stop():
    plan = planner.make_plan(current_state)      # GPT-4 JSON plan
    for task in plan["subtasks"]:
        dispatcher.enqueue(task)
    # wait for all tasks to finish or timeout
    new_results = db.fetch_recent_results(plan_id=plan["id"])
    validated = validator.run(new_results)
    summary = aggregator.summarize(validated)
    current_state.update(summary)                # include summaries and metrics
```

---

### Why This Works

* **Separation of concerns**: GPT focuses on “what” and “why,” MCP servers do “how.”
* **Iterative feedback**: Every loop refines the plan based on validated data.
* **Scalable**: New modules or data sources plug in as new task types.
* **Human-like**: Small, verifiable steps with explicit validation before moving on.

---

**Next Steps**

1. Prototype the **Planner → Task Dispatcher** interface.
2. Containerize one module (e.g., the Google CSE search worker) on your MCP servers.
3. Build the **Validator** against your hospital spreadsheet.
4. Wire it all together and run a single iteration.

