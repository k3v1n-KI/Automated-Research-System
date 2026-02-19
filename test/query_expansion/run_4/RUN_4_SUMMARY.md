# Run 4 - Full Algorithm Comparison Summary

## Execution Details
- **Timestamp**: 2026-02-03T16:12:18.861285
- **Goal**: hospitals in ontario
- **Queries per method**: 6

## Output Files

### Simple LLM Method
- **Directory**: `simple_llm/`
- **Dataset CSV**: `dataset_*.csv` - Contains 529 rows
- **Metrics**:
  - Rows: 529
  - Unique rows: 529
  - Columns: 95
  - Completeness: 22.73%
  - Sources found: 38
  - Execution time: 1849.14s

### Query Expansion Matrix (QEM) Method
- **Directory**: `qem/`
- **Dataset CSV**: `dataset_*.csv` - Contains 125 rows
- **Metrics**:
  - Rows: 125
  - Unique rows: 125
  - Columns: 81
  - Completeness: 33.18%
  - Sources found: 4
  - Execution time: 1527.56s

### Comparison Data
- **Full comparison JSON**: `full_comparison_*.json`
- **Queries used CSV**: `queries_used_*.csv`

## Key Findings

| Metric | Simple LLM | QEM | Difference |
|--------|-----------|-----|-----------|
| Rows | 529 | 125 | -404 (-76.4%) |
| Columns | 95 | 81 | -14 |
| Completeness | 22.7% | 33.2% | +10.4pp |
| Sources | 38 | 4 | -34 (-89.5%) |
| Time | 1849.14s | 1527.56s | - |

## How to Verify

1. **Check dataset CSVs** in `simple_llm/` and `qem/` directories
2. **Review queries used** in `queries_used_*.csv`
3. **See full metrics** in `full_comparison_*.json`
4. **Compare row counts** between the two dataset CSVs
5. **Analyze columns** to see which method discovered more/better data
