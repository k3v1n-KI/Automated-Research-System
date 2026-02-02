# Diversity Tracking Feature - Complete Implementation

## Overview
The research system now includes a comprehensive diversity analysis feature using the Gini-Simpson diversity index to track dataset quality across priority columns.

## Feature Components

### 1. Setup Page (`templates/index.html`)
- **Purpose**: Initial research configuration page
- **Features**:
  - Research prompt textarea
  - Dynamic column definition interface
  - Priority column checkboxes (for diversity tracking)
  - Add/Remove column buttons
  - Form validation
  - SessionStorage persistence

- **User Flow**:
  1. User enters research prompt (e.g., "Find hospitals in Ontario")
  2. User defines data columns they expect (Name, Address, Website)
  3. User marks which columns are "Priority" for diversity tracking
  4. Sends to chat interface with setup data

### 2. Diversity Analyzer (`diversity_analyzer.py`)
- **Functions**:
  - `calculate_gini_simpson_index(values)` → float
    - Formula: 1 - Σ(n_i/N)²
    - Returns: 0.0 (no diversity) to 1.0 (maximum diversity)
  
  - `get_value_distribution(values)` → Dict
    - Counts unique values and calculates percentages
    - Includes diversity index per column
  
  - `analyze_dataset_diversity(records, priority_columns)` → Dict
    - Analyzes all priority columns
    - Returns per-column and overall diversity
  
  - `generate_diversity_report(session_id, records, priority_columns, round)` → Dict
    - Creates frontend-ready report with structure:
      ```json
      {
        "overall_diversity_index": 0.625,
        "column_analysis": [
          {
            "column": "Name",
            "unique_count": 5,
            "total_count": 10,
            "diversity_index": 0.64,
            "value_breakdown": [...]
          }
        ]
      }
      ```
  
  - `compare_diversity_across_rounds(reports)` → Dict
    - Compares trends across multiple research rounds

### 3. Chat Interface (`templates/chat_interface.html`)
- **Updates**:
  - Display selected columns and priority status at top
  - Load setup data from SessionStorage
  - Send columns and priority_columns with start_research event
  - Display diversity metrics in results section
  - Show per-column analysis with unique values and percentages

- **Results Display**:
  - Overall Diversity Index (0-100%)
  - Per-column breakdown with diversity scores
  - Unique value counts
  - Diversity assessment (Low/Moderate/High)

### 4. Server Integration (`server.py`)
- **Routes**:
  - `GET /` → Serve `templates/index.html` (setup page)
  - `GET /chat` → Serve `templates/chat_interface.html` (research interface)

- **Socket Events**:
  - Modified `start_research` to accept `columns` and `priority_columns`
  - Updated `research_complete` to include `diversity_report` in response

- **Integration Points**:
  - Imports `generate_diversity_report` from diversity_analyzer
  - Calls diversity analysis when research completes
  - Passes results to frontend

### 5. Algorithm State (`algorithm.py`)
- **New Fields in ResearchState**:
  - `columns: List[Dict]` - [{name: str, isPriority: bool}, ...]
  - `priority_columns: List[str]` - ["Name", "Address", ...]

## Gini-Simpson Diversity Index Explained

The Gini-Simpson index measures how evenly distributed values are in a column:

- **Formula**: 1 - Σ(n_i/N)²
  - n_i = count of each unique value
  - N = total number of records
  - Σ = sum of all squared proportions

- **Interpretation**:
  - 0.0 = All values identical (no diversity)
  - 0.33 = Low diversity (few unique values)
  - 0.67 = Moderate diversity
  - 1.0 = Maximum diversity (all values unique)

- **Example**:
  - Column with 10 records, all same value: 0.0
  - Column with 10 records, 5 unique values (2 each): 0.8
  - Column with 10 records, 10 unique values (1 each): 0.9

## Setup Data Flow

```
Setup Page (index.html)
  ↓
  User fills: Prompt + Columns with Priority checkboxes
  ↓
  SessionStorage saves setup data
  ↓
  Redirect to /chat
  ↓
Chat Interface (chat_interface.html)
  ↓
  Load setup from SessionStorage
  ↓
  Display columns and priority status
  ↓
  User clicks Send
  ↓
  Extract columns/priority_columns from setup
  ↓
  Emit start_research with all data
  ↓
Server receives start_research
  ↓
  Extract columns and priority_columns from event
  ↓
  Pass to algorithm in state
  ↓
Algorithm processes data
  ↓
  Returns extracted dataset
  ↓
Server computes diversity metrics
  ↓
  generate_diversity_report(session_id, dataset, priority_columns, round)
  ↓
  Returns report with overall_diversity_index + column_analysis
  ↓
Frontend receives research_complete
  ↓
  Display diversity metrics in results
  ↓
  Show overall index + per-column breakdown
```

## Testing the Feature

### Test 1: Page Load and Navigation
```bash
1. Start server: python3 server.py
2. Open browser to http://localhost:8000/
3. Verify index.html loads (Research Setup page)
4. Enter prompt: "Test research"
5. Keep default columns (Name, Address, Website)
6. Click "Start Research"
7. Verify redirects to /chat
8. Verify column info displays at top
```

### Test 2: Diversity Metrics Display
```bash
1. Setup page complete as above
2. In chat interface, submit a message
3. Wait for research to complete
4. In Results section, verify:
   - "📊 Diversity Metrics" header appears
   - Overall index shows as percentage
   - Each priority column listed with:
     - Unique count / total count
     - Diversity % value
   - Diversity assessment (Low/Moderate/High)
```

### Test 3: Multiple Rounds
```bash
1. Complete first research round
2. Note the diversity index for a column
3. Click "Continue from previous session" (if available)
4. Run second research round
5. Verify diversity metrics update
6. Verify combined dataset includes both rounds
```

## File Structure
```
/home/k3v1n/projects/Automated-Research-System/
├── templates/
│   ├── index.html (NEW - Setup page)
│   └── chat_interface.html (UPDATED - Shows diversity metrics)
├── diversity_analyzer.py (NEW - Diversity calculations)
├── algorithm.py (UPDATED - Added columns & priority_columns fields)
├── server.py (UPDATED - Routes & diversity report generation)
└── ... (other files)
```

## Key Implementation Details

### SessionStorage Usage
- Setup data persists in browser session
- Format: `{prompt: string, columns: [{name: string, isPriority: bool}], startTime: string}`
- Cleared when user closes browser tab

### Column Definition Format
```javascript
// In setup data
{
  name: "Hospital Name",      // Column name
  isPriority: true            // Whether to track diversity
}
```

### Diversity Report Format
```javascript
{
  session_id: "uuid",
  round: 0,
  total_records: 150,
  priority_columns: ["Name", "Address"],
  overall_diversity_index: 0.625,  // 0.0 to 1.0
  column_analysis: [
    {
      column: "Name",
      unique_count: 47,
      total_count: 150,
      diversity_index: 0.625,
      value_breakdown: [...]
    },
    // ... more columns
  ],
  summary: "Round 0: 150 records | Diversity Index: 62.50% (Moderate) | ..."
}
```

## Future Enhancements

1. **Dataset Merging**: Implement deduplication when combining rounds
2. **Trend Analysis**: Show diversity progression across rounds
3. **Column Updates**: Allow adding/removing columns mid-research
4. **Export with Metadata**: Include diversity metrics in CSV/JSON export
5. **Diversity Goals**: Let users set target diversity metrics
6. **Comparison View**: Compare diversity between different research sessions

## Troubleshooting

### Issue: Setup data not showing in chat
- **Cause**: SessionStorage cleared or wrong URL
- **Fix**: Start from http://localhost:8000/ (setup page)

### Issue: No diversity metrics in results
- **Cause**: No priority columns selected
- **Fix**: Ensure at least one column has Priority checked

### Issue: Incorrect diversity percentages
- **Cause**: Empty values in column
- **Fix**: Check data quality - empty strings are counted as values

### Issue: Server shows vector store error
- **Cause**: PostgreSQL not running
- **Fix**: Not critical - core functionality works without it
