```
                              AUTOMATED RESEARCH SYSTEM - FLOWCHART
                          (Using Standard Flowchart Symbols)


                                    ┌────────┐
                                    │ START  │
                                    └───┬────┘
                                        │
                                        ▼
                              ╱──────────────────╲
                             ╱  User navigates   ╲
                            │   to localhost     │
                             ╲  /index.html      ╱
                              ╲──────────────────╱
                                        │
                                        ▼
                         ┌──────────────────────────┐
                         │ Load Setup Page (HTML)   │
                         └────────┬─────────────────┘
                                  │
                                  ▼
                    ╔═════════════════════════════╗
                    ║ Data Input/Output (I/O)     ║
                    ╠═════════════════════════════╣
                    ║ • Enter Research Prompt    ║
                    ║ • Define Columns           ║
                    ║ • Mark Priority Columns    ║
                    ╚════════┬══════════════════╝
                             │
                             ▼
                      ◇─────────────────◇
                     ╱  Columns have    ╲
                    │  Priority marked?  │
                     ╲                  ╱
                      ◇────────┬────────◇
                               │
                    ┌──────────┘ YES
                    │
                    ▼
     ┌──────────────────────────────────────┐
     │ Process: Save setup to SessionStorage │
     │ {prompt, columns, isPriority}        │
     └────────────────┬─────────────────────┘
                      │
                      ▼
     ┌──────────────────────────────────────┐
     │ Navigate to /chat (chat_interface)   │
     └────────────────┬─────────────────────┘
                      │
                      ▼
     ┌──────────────────────────────────────┐
     │ Process: Load Setup from SessionSt.  │
     │ Display Columns & Priority Status    │
     └────────────────┬─────────────────────┘
                      │
                      ▼
                ◇─────────────────◇
               ╱  Socket.IO       ╲
              │  Connected?       │
               ╲                  ╱
                ◇────────┬────────◇
                         │
               ┌─────────┘ YES
               │
               ▼
  ┌─────────────────────────────────────────────┐
  │ Process: Auto-trigger sendMessage()         │
  │ (Inject prompt into chat input)             │
  │ Emit: start_research event                  │
  └──────────┬────────────────────────────────┘
             │
             │ WebSocket Event
             │
             ▼
┌────────────────────────────────────────────────┐
│ BACKEND: server.py - Event Handler            │
│ handle_start_research(data)                    │
└────────┬───────────────────────────────────────┘
         │
         ▼
╔════════════════════════════════════════════╗
║ Data I/O: Extract from WebSocket Event    ║
╠════════════════════════════════════════════╣
║ • prompt                                   ║
║ • columns: [{name, isPriority}, ...]      ║
║ • priority_columns: ["Name", "Website"]   ║
╚════════┬═════════════════════════════════╝
         │
         ▼
┌────────────────────────────────────────┐
│ Process: Create ResearchState          │
│ Initialize algorithm graph             │
└────────┬───────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────┐
│ Process: Execute Algorithm Nodes       │
└────────┬───────────────────────────────┘
         │
         ▼
    ┌─────────────────┐
    │ Node 1: Query   │
    │ Generation      │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Node 2: Search  │
    │ (SearXNG)       │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Node 3: Validate│
    │ URLs            │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Node 4: Scrape  │
    │ Content         │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────────────────────┐
    │ Node 5: Extract Data (★)        │
    │ LLM extracts ONLY specified     │
    │ columns per system prompt       │
    └────────┬────────────────────────┘
             │
             ▼
    ┌─────────────────┐
    │ Node 6:         │
    │ Deduplicate     │
    └────────┬────────┘
             │
             ▼
        ◇──────────────◇
       ╱  Algorithm    ╲
      │  Complete?     │
       ╲               ╱
        ◇──────┬───────◇
               │
               │ YES
               ▼
┌──────────────────────────────────────────┐
│ Process: Post-Processing                 │
│ _run_algorithm_task()                    │
└────────┬─────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│ Process: Filter Dataset to Specified     │
│ Columns Only (★)                         │
│ Remove extra fields added by LLM         │
└────────┬─────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│ Process: Generate Diversity Report       │
│ Call: diversity_analyzer.py              │
│ • Calculate Gini-Simpson Index           │
│ • Count unique values per column         │
│ • Calculate percentages                  │
└────────┬─────────────────────────────────┘
         │
         ▼
╭════════════════════════════════════════╮
║ ⌗ Database: Save to CSV                ║
║ datasets/dataset_[session_id].csv      ║
║ (With ONLY specified columns)          ║
╰════════┬═════════════════════════════╯
         │
         ▼
┌──────────────────────────────────────┐
│ Process: Emit research_complete      │
│ Event with all results & metrics     │
└────────┬───────────────────────────┘
         │
         │ WebSocket Event
         │
         ▼
    FRONTEND: chat_interface.html
         │
         ▼
┌──────────────────────────────┐
│ Process: Receive Results     │
│ completeResearch(data)       │
└────────┬─────────────────────┘
         │
         ▼
      ◇──────────────────◇
     ╱  Diversity Data   ╲
    │  Available?        │
     ╲                  ╱
      ◇────────┬────────◇
               │
        ┌──────┘ YES
        │
        ▼
┌───────────────────────────────────────┐
│ Process: Render Diversity Metrics     │
├───────────────────────────────────────┤
│ • Display Diversity Index (0.625)     │
│ • Per-Column Analysis:                │
│   - Unique counts                     │
│   - All unique values with %          │
│   - Diversity index per column        │
└────────┬────────────────────────────┘
         │
         ▼
╭═══════════════════════════════════╮
║ ⌗ Document: Display Results Panel ║
║ (Right column, bottom section)    ║
╰═══════════┬═══════════════════════╯
            │
            ▼
┌──────────────────────────────────┐
│ User views:                       │
│ • Query statistics                │
│ • URL metrics                     │
│ • Final record count              │
│ • Diversity Metrics               │
│ • Value distributions             │
└────────┬─────────────────────────┘
         │
         ▼
      ◇──────────────────◇
     ╱  Continue or      ╲
    │  Start New          │
    │  Research?         │
     ╲                  ╱
      ◇────────┬────────◇
               │
         ┌─────┴─────┐
         │           │
    NO  │           │  YES
         │           │
         ▼           ▼
      ┌────┐    Go back to
      │END │    setup page
      └────┘    or chat
                
```

---

## FLOWCHART LEGEND

### Symbols Used:

| Symbol | Name | Meaning |
|--------|------|---------|
| ⭕ (Oval) | Terminator | Start or End point |
| ▭ (Rectangle) | Process | A step or operation |
| ◇ (Diamond) | Decision | A yes/no question or condition |
| ╱╲ (Parallelogram) | Data I/O | Input or Output data |
| ⌗ (Cylinder) | Database | Data storage/file |
| ▭ (Document) | Document | Written or displayed document |
| ↓ (Arrow) | Flow | Direction of process flow |

---

## KEY DECISION POINTS

1. **Columns Priority Marked?**
   - If NO: Return error, require at least one priority column
   - If YES: Proceed to save and navigate

2. **Socket Connected?**
   - If NO: Retry after delay
   - If YES: Auto-trigger research

3. **Algorithm Complete?**
   - If NO: Continue processing
   - If YES: Move to post-processing

4. **Diversity Data Available?**
   - If YES: Render metrics section
   - If NO: Show basic statistics only

5. **Continue Research?**
   - If YES: Return to setup page
   - If NO: End session

---

## CRITICAL PROCESSES (★)

### Node 5: Extract Data (★)
- LLM constrained to extract ONLY specified columns
- System prompt specifies: "Extract ONLY: Name, Address, Website"

### Filter Dataset (★)
- Server-side filtering ensures clean output
- Removes any extra fields not in column specification
- Guarantees final dataset has exactly specified columns

---

## DATA FLOW SUMMARY

```
Setup Page Input
    ↓
SessionStorage (columns + priority)
    ↓
Chat Interface Auto-start
    ↓
WebSocket: start_research event
    ↓
Algorithm: 6-node pipeline
    ↓
Post-Processing: Filter + Diversity Calc
    ↓
Database: Save filtered CSV
    ↓
WebSocket: research_complete event
    ↓
Frontend: Display Results + Metrics
    ↓
User views Diversity Analysis
```
