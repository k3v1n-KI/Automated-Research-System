# Automated Research System Algorithm

This document explains the automated research system's algorithm, detailing each node's functionality and execution flow based on real system execution data from November 6, 2025.

## System Overview

The system implements an iterative research process through a directed graph of specialized nodes, each handling a specific aspect of the research task. The system uses LangGraph for workflow orchestration and maintains state across iterations.

## Node Execution Flow

### 1. Plan Node
**Purpose**: Initializes the research process and establishes the initial search strategy.

**Implementation Details**:
- Creates a structured plan from the research goal
- Generates initial search queries using multiple strategies:
  - Direct goal query
  - List-focused query
  - Directory-focused query
  - Website-focused query
- Initializes session tracking and control parameters

**Example from Latest Run**:
```python
plan = {
    "goal": "Find hospitals in Ontario",
    "steps": ["search","validate","refine","scrape","extract","aggregate"],
    "queries": [
        "Find hospitals in Ontario",
        "Find hospitals in Ontario list",
        "Find hospitals in Ontario directory",
        "Find hospitals in Ontario website"
    ]
}
```

### 2. Search Node
**Purpose**: Executes search queries and collects results from multiple sources.

**Implementation Details**:
- Processes each query through SearXNG search engine
- Tracks hits per query
- Deduplicates results across queries
- Maintains search history to avoid duplicates in future rounds

**Metrics from Latest Run**:
```
[13:23:02] SearXNG hits for 'Find hospitals in Ontario': 15
[13:23:03] SearXNG hits for 'Find hospitals in Ontario list': 10
[13:23:04] SearXNG hits for 'Find hospitals in Ontario directory': 10
[13:23:05] SearXNG hits for 'Find hospitals in Ontario website': 10
Total raw results: 45
Deduplicated results: 25
```

### 3. Validate Node
**Purpose**: Evaluates and ranks search results for relevance and quality.

**Implementation Details**:
- Uses embedding-based similarity scoring
- Applies configurable threshold filtering
- Ranks results by relevance to research goal
- Maintains quality control through parameterized filtering

**Performance Metrics**:
```
[13:23:07] Processing 25 raw results
[13:23:08] Kept 17/25 (threshold=0.6)
Parameters: top_k=50, threshold=0.6
```

### 4. Scrape Node
**Purpose**: Retrieves and processes content from validated URLs.

**Implementation Details**:
- Parallel content retrieval from validated URLs
- Error handling and domain failure tracking
- Content cleaning and normalization
- HTML to text conversion when needed

**Latest Run Statistics**:
```
[13:23:08] Scraping 17 validated URLs
[13:23:31] Retrieved 17 responses (16 with usable text)
Success rate: 94%
```

### 5. Extract Node
**Purpose**: Converts raw scraped content into structured data.

**Implementation Details**:
- Processes text in configurable chunks (default 3000 chars)
- Uses LLM-based extraction with regex fallback
- Handles long documents through smart chunking
- Validates and normalizes extracted entities

**Processing Statistics**:
```
Input: 16 scraped documents
Chunk size: 3000 characters
Overlap: 250 characters
Extracted entities: 33
```

### 6. Aggregate Node
**Purpose**: Combines and deduplicates extracted information.

**Implementation Details**:
- Merges new items with existing dataset
- Resolves conflicts and duplicates
- Tracks item count changes
- Monitors information gain

**Latest Run Metrics**:
```
Previous total: 0
New items added: 35
Gain metric tracked for stopping decisions
```

### 7. Profile Node
**Purpose**: Analyzes result quality and coverage.

**Implementation Details**:
- Computes field coverage metrics
- Identifies information gaps
- Analyzes source diversity
- Tracks geographical distribution

**Key Metrics Tracked**:
```json
{
    "item_count": 35,
    "field_coverage": {
        "name": 1.0,
        "address": 0.85,
        "phone": 0.72,
        "website": 0.68
    }
}
```

### 8. Critic Node
**Purpose**: Evaluates results and guides refinement strategy.

**Implementation Details**:
- Analyzes coverage gaps
- Suggests focused queries for missing information
- Adapts search strategy based on results
- Generates refined search templates

**Example Strategy**:
```python
actions = [
    {
        "query_templates": [
            "{goal} hospital contact",
            "{goal} hospital website",
            "{goal} hospital address"
        ],
        "slots": {"goal": goal}
    }
]
```

## Control Flow

The system uses a `StateGraph` to manage node execution with conditional transitions:

1. START → plan → search → validate → scrape → extract → aggregate → profile → critic
2. critic → conditional branch:
   - If refinement needed: bump_round → refine → search (new round)
   - If stopping conditions met: stop_check → END

## Stopping Conditions

The system monitors several metrics to determine when to stop:
- Maximum rounds reached (set to 3 in latest run)
- Zero hits in consecutive rounds
- No information gain in consecutive rounds
- Coverage thresholds met

## Real-world Performance

From the November 6, 2025 run:
- Completed 3 rounds
- Processed 17 unique sources
- Extracted 33-35 unique entities
- Achieved good coverage across key fields
- Demonstrated efficient stopping when additional rounds would yield diminishing returns