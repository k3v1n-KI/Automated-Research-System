# Revamped System Overview

## What Changed

### ✅ Removed
- **Firebase dependency** - No longer storing to Firestore
- **Postgres checkpointer** - No database persistence needed
- **main.py** - Replaced by chat interface
- **Complex task_dispatcher.py** - Simplified into algorithm.py
- **lang_graph.py** - Redesigned as algorithm.py

### ✅ Added
- **algorithm.py** - Clean 6-node LangGraph pipeline
- **Updated server.py** - WebSocket-based algorithm runner
- **Modern chat_interface.html** - Beautiful dataset builder UI
- **Real-time progress tracking** - Terminal + frontend updates

---

## Core System (algorithm.py)

### 6-Node Pipeline

```
START
  ↓
1️⃣  QUERY GENERATION (LLM generates 4 diverse queries)
  ↓
2️⃣  SEARCH (SEARXNG + Google API → 60 URLs)
  ↓
3️⃣  VALIDATE (Semantic similarity, threshold 0.5)
  ↓
4️⃣  SCRAPE (Playwright → raw HTML)
  ↓
5️⃣  EXTRACT (LLM extracts structured data from 3000 char chunks)
  ↓
6️⃣  DEDUPLICATE (LLM removes duplicates → final dataset)
  ↓
END
```

### Key Classes

**ResearchState** (Dict-based)
```python
{
    'initial_prompt': str,
    'queries': List[str],
    'search_results': List[Dict],
    'validated_urls': List[str],
    'scraped_content': List[Dict],
    'extracted_items': List[Dict],
    'final_dataset': List[Dict],
    'session_id': str,
    'round': int,
    'error': Optional[str]
}
```

**ProgressTracker**
- Logs to terminal with detailed formatting
- Emits progress events to frontend via WebSocket
- Tracks current step and detail

---

## Server Architecture (server.py)

### Flask + SocketIO
- **Route: /** → Serves chat_interface.html
- **Route: /health** → Health check
- **WebSocket: connect/disconnect** → Connection events
- **WebSocket: start_research** → Launches algorithm
- **WebSocket: export_dataset** → Downloads results

### Session Management
```python
sessions[session_id] = {
    'client_id': socket_id,
    'state': ResearchState,
    'algorithm_graph': compiled_langgraph,
    'started_at': datetime
}
```

### Algorithm Execution
```
User sends prompt via WebSocket
    ↓
Server creates ResearchState
    ↓
Server builds LangGraph with progress emitter
    ↓
Background task runs: algorithm_graph.ainvoke(state)
    ↓
Progress events emitted in real-time to frontend
    ↓
Final results sent as "research_complete" event
```

---

## Frontend (chat_interface.html)

### Two-Panel Layout

**Left Panel (Input)**
- Textarea for dataset description
- Start/Cancel buttons
- Error messages

**Right Panel (Progress)**
- Real-time progress updates
- Spinner for active step
- Complete checkmark for finished steps

**Bottom (Results)**
- Statistics dashboard (queries, URLs, records)
- Export JSON/CSV buttons

### WebSocket Events

```javascript
// Client → Server
socket.emit('start_research', { prompt: "..." })
socket.emit('export_dataset', { session_id: "...", format: "json" })

// Server → Client
socket.on('research_start', (data) => { ... })
socket.on('progress', (data) => { ... })
socket.on('research_complete', (data) => { ... })
socket.on('research_error', (data) => { ... })
```

---

## Terminal Output Example

```
======================================================================
📊 Starting Research Session
   Client: abc123xyz
   Prompt: Find all hospitals in Ontario...
======================================================================

======================================================================
📍 🔍 Query Generation
   Analyzing prompt and generating diverse search queries...
   Data: {"queries": ["...4 queries..."]}
======================================================================

======================================================================
📍 🔎 Searching
   Query 1/4: hospitals in Ontario public health
======================================================================

[... continues for each node ...]

======================================================================
✓ Research completed for session abc123
  Final dataset: 95 records
======================================================================
```

---

## Data Flow Example

### Input Prompt
```
"Find all hospitals in Ontario with Name, Address, and Website URL"
```

### Node 1: Query Generation
```
Generated 4 diverse queries:
1. "hospitals in Ontario"
2. "Ontario medical centers and facilities"
3. "healthcare institutions Ontario Canada"
4. "Ontario hospital directory list"
```

### Node 2: Search (4 queries × 15 results = 60 URLs)
```
hospital.com
ontario.ca/health/hospitals
canadianhealthcare.org
... (57 more)
```

### Node 3: Validate (threshold 0.5)
```
Input: 60 URLs
Output: 45 validated URLs (15 filtered out as irrelevant)
```

### Node 4: Scrape
```
Input: 45 URLs
Output: 40 HTML documents (5 failed/timeout)
```

### Node 5: Extract
```
Input: 40 HTML documents (3000 char chunks)
Output: 120 extracted records
{
  "Name": "Toronto General Hospital",
  "Address": "123 Main St, Toronto",
  "Website": "torontogeneral.ca"
}
```

### Node 6: Deduplicate
```
Input: 120 records
Output: 95 unique records
(25 duplicates removed)
```

---

## Environment Setup

### Required
```bash
OPENAI_API_KEY=sk-...
SEARXNG_URL=http://localhost:8888
PORT=8000
```

### Optional
```bash
GOOGLE_API_KEY=...
GOOGLE_CX=...
DEBUG=false
```

---

## Running the System

### Step 1: Start SEARXNG (if using)
```bash
cd searxng-docker
docker-compose up -d
```

### Step 2: Start Research Server
```bash
conda activate auto_research
python server.py
```

### Step 3: Open Browser
```
http://localhost:8000
```

### Step 4: Build Dataset
1. Enter prompt (e.g., "Find hospitals in Ontario...")
2. Click "Start Research"
3. Watch progress in real-time
4. Export results as JSON or CSV

---

## File Changes Summary

| File | Action | Reason |
|------|--------|--------|
| algorithm.py | **CREATED** | New 6-node pipeline |
| server.py | **REPLACED** | Removed Firebase/Postgres |
| templates/chat_interface.html | **UPDATED** | New dataset builder UI |
| main.py | **DEPRECATED** | Replaced by chat interface |
| lang_graph.py | **ARCHIVED** | Replaced by algorithm.py |
| task_dispatcher.py | **ARCHIVED** | Functionality merged |
| firebase.py | **ARCHIVED** | No longer needed |
| ALGORITHM_README.md | **CREATED** | Documentation |

---

## Key Improvements

✅ **No Firebase** - Pure in-memory state management
✅ **No Database** - Session data stays in memory
✅ **Real-time Updates** - WebSocket progress tracking
✅ **Clean Code** - Simple 6-node architecture
✅ **User-Friendly** - Modern web interface
✅ **Detailed Logging** - Terminal + Frontend progress
✅ **Error Handling** - Graceful fallbacks
✅ **Exportable** - JSON/CSV download support

---

## Next: Follow-up Queries

Once a dataset is built, users can request modifications:
- "Find more hospitals in rural Ontario"
- "Add phone number and hours columns"
- "Filter for hospitals with emergency services"

This would re-run the algorithm with updated prompts while leveraging the chat interface.
