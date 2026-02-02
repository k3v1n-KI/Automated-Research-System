# Context Persistence & Dataset Tweaking Guide

## Overview

The Automated Research System now supports **persistent context across sessions** using PostgreSQL vector storage. This allows users to:

1. **Run initial research** to collect data
2. **Save the context** (queries, results, extracted items)
3. **Load previous sessions** and continue where they left off
4. **Tweak with new instructions** to collect additional targeted data
5. **Maximize data collection** by iteratively refining queries based on previous results

## Architecture

### New Components

#### 1. **context_persistence.py**
Manages saving and loading research sessions to/from vector store.

```python
# Save a research session
await context_manager.save_research_session(session_data)

# Load previous session context
previous_context = await context_manager.load_session_context(session_id)

# Get previous queries
previous_queries = await context_manager.get_previous_queries(session_id)

# Get previously extracted items
previous_items = await context_manager.get_previous_extracted_items(session_id)
```

#### 2. **Updated ResearchState** (algorithm.py)
Added fields to track multi-session workflows:

```python
class ResearchState(TypedDict):
    # ... existing fields ...
    previous_session_id: Optional[str]      # Link to previous session
    tweak_instructions: Optional[str]       # User's refinement instructions
    previous_queries: List[str]             # Queries from previous session
    previous_items: List[Dict]              # Extracted items from previous session
```

#### 3. **Enhanced Query Generation Node** (nodes/query_generation.py)
Now builds on previous queries:

- Receives previous queries from prior session
- Generates NEW diverse queries based on tweak instructions
- Appends new queries to previous ones instead of replacing them
- Focuses new queries on data gaps identified in tweaks

#### 4. **Frontend Updates** (templates/chat_interface.html)
New UI sections:

- **Previous Sessions Dropdown**: Select which session to continue from
- **Tweak Instructions Textarea**: Specify what data gaps to fill or areas to focus on
- **Session Context Display**: Shows when previous context is loaded

### Data Flow

```
User Session 1
  ↓
[Initial Prompt] → Query Gen → Search → Scrape → Extract → Final Dataset
  ↓
Vector Store (saves queries, items, dataset)
  ↓

User Session 2 (Continue)
  ↓
[Load from Vector Store] + [New Tweak Instructions]
  ↓
[New Prompt] → Query Gen (uses previous + new) → Search → Scrape → Extract → Enhanced Dataset
  ↓
Vector Store (saves new context, appends to existing)
```

## Usage Workflow

### First Session: Initial Research

1. **Write initial prompt**: "Find all hospitals in Ontario with Name, Address, and Website"
2. **Send message**: System generates 4 queries, searches, scrapes, and extracts data
3. **Results saved**: 
   - CSV file: `datasets/dataset_xxxxx.csv`
   - Vector store: Context saved with session ID

### Second Session: Refinement (Tweaking)

1. **Server starts**: Previous sessions appear in dropdown
2. **Select session**: Choose "Session 1" from dropdown
3. **Add tweaks**: "Also include phone numbers, operating hours, and emergency departments"
4. **Write new prompt**: Optional - can be the same or different
5. **Send message**: 
   - System loads previous 4 queries
   - Generates 4 NEW queries targeting the tweaked requirements
   - Searches with all 8 queries (previous + new)
   - Extracts with extended schema (includes phone, hours, departments)
   - Final dataset has ALL records with ALL fields (preserves previous data)

## Server API Changes

### New Socket Event: `get_previous_sessions`

**Client sends:**
```javascript
socket.emit('get_previous_sessions');
```

**Server responds:**
```javascript
socket.on('previous_sessions', (data) => {
    // data.sessions = [{session_id, created_at, metadata}, ...]
});
```

### Updated Socket Event: `start_research`

**Old format:**
```javascript
socket.emit('start_research', { 
    prompt: "Find hospitals..." 
});
```

**New format:**
```javascript
socket.emit('start_research', { 
    prompt: "Find hospitals...",
    previous_session_id: "abc123...",  // Optional
    tweak_instructions: "Also get phone numbers..."  // Optional
});
```

### New Socket Event: `context_loaded`

Emitted when previous context is successfully loaded:

```javascript
socket.on('context_loaded', (data) => {
    // data.previous_queries_count = 4
    // data.previous_items_count = 87
});
```

## Backend Implementation Details

### Session Initialization (server.py: `handle_start_research`)

```python
# If previous_session_id provided:
if previous_session_id and context_manager:
    # Load previous queries and items async
    previous_queries = await context_manager.get_previous_queries(previous_session_id)
    previous_items = await context_manager.get_previous_extracted_items(previous_session_id)
    
    # Add to state
    state['previous_queries'] = previous_queries
    state['previous_items'] = previous_items
    state['previous_session_id'] = previous_session_id
    state['tweak_instructions'] = tweak_instructions
```

### Query Generation (nodes/query_generation.py)

```python
# If previous queries exist:
if previous_queries:
    # Build context for LLM
    previous_context = f"""
Previous queries from earlier research:
{previous_queries}

User instructions for improvement: {tweak_instructions}
"""
    
    # Send to LLM with context
    # LLM generates NEW queries targeting gaps
    
    # Result: new_queries + previous_queries
    state['queries'] = previous_queries + new_queries
```

### Session Persistence (server.py: `_run_algorithm_task`)

After research completes:

```python
# Save to vector store
if context_manager:
    await context_manager.save_research_session({
        "session_id": session_id,
        "initial_prompt": final_state['initial_prompt'],
        "queries": final_state['queries'],
        "extracted_items": final_state['extracted_items'],
        "final_dataset": final_state['final_dataset']
    })
```

## CSV Data Preservation

The system ensures **all records are preserved** across sessions:

1. **Field Normalization**: All records have all fields (empty strings for missing)
2. **Incremental Growth**: New sessions add to existing data, not replace it
3. **Version Tracking**: Each session saved with timestamp in vector store

Example:

**Session 1 Result:**
```csv
Hospital_Name, Address, Website
General Hospital, 123 Main St, hospital.ca
...
```

**Session 2 with Tweaks (for phone + hours):**
```csv
Hospital_Name, Address, Website, Phone, Operating_Hours
General Hospital, 123 Main St, hospital.ca, 555-1234, 24/7
Clinic A, 456 Oak Ave, clinic.ca, 555-5678, 9-5
...
```

All previous records preserved with new fields populated.

## Vector Store Tables

Data saved in PostgreSQL with pgvector:

### knowledge_base table
```sql
- id (serial primary key)
- embedding (vector 1536 dims)
- data (jsonb with queries, items, metadata)
- content_hash (deduplication)
- session_id (UUID)
- created_at, last_updated_at
```

### thoughts table
```sql
- id (serial primary key)
- content (text)
- session_id (UUID)
- step_number (integer)
- type (varchar)
- embedding (vector 1536 dims)
- created_at
```

## Error Handling

### Vector Store Connection Failure
If PostgreSQL is unavailable:
- Server continues without persistence
- Sessions work normally but aren't saved for next time
- Warning logged: "Vector store failed to initialize"

### Previous Context Load Failure
If loading previous context fails:
- Current session starts fresh
- Error logged but research continues
- User can still refine with new queries

## Performance Considerations

1. **Async Loading**: Previous context loaded in background thread
2. **Query Batching**: All queries (previous + new) executed together
3. **Vector Search**: Fast semantic similarity for context retrieval (GPU-accelerated if available)
4. **CSV Writing**: Field normalization done in-memory, single write to disk

## Future Enhancements

1. **Semantic Query Clustering**: Group similar queries to avoid redundancy
2. **Data Gap Analysis**: Automatically identify missing fields/records
3. **Multi-turn Conversation**: Remember context across multiple tweaks
4. **Export History**: Download all versions of dataset at different points
5. **Collaborative Sessions**: Multiple users working on same dataset

## Testing Checklist

- [ ] Start new research session
- [ ] Verify CSV saved to `datasets/` folder
- [ ] Start server again
- [ ] Previous session appears in dropdown
- [ ] Select previous session
- [ ] Add tweak instructions
- [ ] Send new prompt
- [ ] Verify new queries are generated (different from previous)
- [ ] Verify final dataset includes all records (new + previous)
- [ ] Check vector store has context saved (PostgreSQL side)
