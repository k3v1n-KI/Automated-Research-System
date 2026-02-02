# Dataset Builder - Research Algorithm

A complete dataset generation system using LangGraph with 6 specialized nodes for automated web research.

## Architecture

### 6-Node Pipeline

1. **Query Generation** 🔍
   - Takes initial prompt specifying dataset requirements
   - Generates 4 diverse search queries using LLM
   - Example: "Find hospitals in Ontario" → 4 varied queries

2. **Search** 🔎
   - Executes 4 queries across SEARXNG and Google APIs
   - Returns ~15 URLs per query (60 total)
   - Deduplicates results by URL

3. **Validation** ✅
   - Uses semantic similarity (cosine similarity threshold: 0.5)
   - Compares URL metadata against source queries
   - Filters out irrelevant URLs

4. **Scraping** 📄
   - Uses Playwright to scrape validated URLs
   - Extracts raw HTML and text content
   - Handles timeouts and errors gracefully

5. **Extraction** 🎯
   - Processes scraped content (limited to 3000 chars per document)
   - Uses LLM to extract structured data based on original prompt
   - Outputs JSON matching specified columns

6. **Deduplication** 🗑️
   - Uses LLM to identify duplicate entries
   - Removes duplicates while preserving unique records
   - Final dataset output

## Quick Start

### 1. Install Dependencies

```bash
conda activate auto_research
pip install playwright
python -m playwright install chromium
```

### 2. Set Environment Variables

```bash
export OPENAI_API_KEY="your-key"
export SEARXNG_URL="http://localhost:8888"
export GOOGLE_API_KEY="your-key"  # Optional
export GOOGLE_CX="your-cx"         # Optional
```

### 3. Start Server

```bash
python server.py
```

### 4. Access Interface

Open browser to: `http://localhost:8000`

## File Structure

### Core Files

- **algorithm.py** - LangGraph pipeline with 6 nodes
- **server.py** - Flask/SocketIO server with WebSocket support
- **templates/chat_interface.html** - Modern web interface

### Deprecated Files (Removed)

- `main.py` - No longer needed
- Firebase-related code - Replaced with in-memory state
- `lang_graph.py` - Replaced with new `algorithm.py`
- `task_dispatcher.py` - Functionality integrated into `algorithm.py`

## API Reference

### WebSocket Events

#### Client → Server

**start_research**
```javascript
socket.emit('start_research', {
  prompt: "Find all hospitals in Ontario with Name, Address, Website"
})
```

**export_dataset**
```javascript
socket.emit('export_dataset', {
  session_id: "...",
  format: "json" // or "csv"
})
```

#### Server → Client

**research_start**
```javascript
{
  session_id: "...",
  timestamp: "..."
}
```

**progress**
```javascript
{
  step: "🔍 Query Generation",
  detail: "Analyzing prompt and generating diverse queries...",
  data: {...}
}
```

**research_complete**
```javascript
{
  session_id: "...",
  final_dataset: [...],
  statistics: {
    queries: 4,
    urls_found: 60,
    urls_validated: 45,
    urls_scraped: 40,
    records_extracted: 120,
    final_count: 95
  }
}
```

**research_error**
```javascript
{
  session_id: "...",
  error: "Error message"
}
```

## Configuration

### algorithm.py Settings

```python
# Query validation threshold (0.0 - 1.0)
threshold = 0.5

# Text length limit for LLM processing
EXTRACT_TEXT_LIMIT = 3000

# Search results per query
RESULTS_PER_QUERY = 15
```

### SEARXNG Configuration

Ensure SEARXNG is running:
```bash
docker-compose -f searxng-docker/docker-compose.yaml up -d
```

## Terminal Logging

Each node provides detailed terminal output:

```
======================================================================
📍 🔍 Query Generation
   Analyzing prompt and generating diverse search queries...
   Data: {"queries": ["query1", "query2", "query3", "query4"]}
======================================================================

======================================================================
📍 🔎 Search
   Query 1/4: hospitals in Ontario
   [searching...]
======================================================================
```

## Frontend Progress Tracking

The web interface shows:
- Current step with spinner
- Detailed progress messages
- Real-time status updates
- Final statistics dashboard
- Export options (JSON/CSV)

## Error Handling

- Graceful fallback from SEARXNG to Google API
- Skips failed URL scrapes (continues with others)
- LLM extraction failures log warnings, continue
- All errors emitted to frontend with details

## Performance Tips

1. **Parallel Processing**: Each node handles multiple items efficiently
2. **Caching**: Search results deduplicated before validation
3. **Memory**: Large HTML stripped before extraction (3000 chars limit)
4. **Timeouts**: 12-15 second timeouts prevent hanging

## Customization

### Add Custom Node

```python
async def custom_node(state: ResearchState, progress: ProgressTracker):
    progress.update("Custom Step", "Processing...")
    state['custom_field'] = "result"
    return state

# In build_research_algorithm():
graph.add_node("custom", make_node(custom_node))
graph.add_edge("previous_node", "custom")
graph.add_edge("custom", "next_node")
```

### Adjust LLM Parameters

In respective nodes, modify `temperature`, `model`, or `messages`:

```python
response = await openai_client.chat.completions.create(
    model="gpt-4",  # Change model
    temperature=0.3,  # Adjust creativity (0-1)
    messages=[...]
)
```

## Troubleshooting

**"Playwright not installed"**
```bash
pip install playwright
python -m playwright install chromium
```

**"SEARXNG connection failed"**
- Check SEARXNG is running: `docker ps`
- Verify `SEARXNG_URL` environment variable

**"OpenAI API error"**
- Check `OPENAI_API_KEY` is set
- Verify API key is valid
- Check account has credits

## Next Steps

1. Test with sample dataset request
2. Monitor terminal logs for issues
3. Adjust validation threshold if too strict/loose
4. Export results and validate quality
5. Fine-tune prompts for better extraction

---

**Built with:** LangGraph, OpenAI, SEARXNG, Playwright, Flask-SocketIO
