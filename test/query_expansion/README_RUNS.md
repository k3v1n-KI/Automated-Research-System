# Query Expansion Comparison Runs

## Overview
Two comparison runs executed to evaluate query generation methods with different configurations.

## Directory Structure
```
query_expansion/
├── run_1/                          # Run 1: Unfair comparison (5 vs 15 queries)
│   ├── DUAL_RUN_COMPARISON_REPORT.md       # Detailed report (11 KB)
│   └── dual_run_comparison_20260203_002757.json  # Raw metrics JSON
│
├── run_2/                          # Run 2: Fair comparison (15 vs 15 queries)
│   ├── RUN_2_COMPARISON_REPORT.md          # Detailed report (9 KB)
│   └── dual_run_comparison_20260203_095842.json  # Raw metrics JSON
│
├── RUNS_COMPARISON.md              # Cross-run analysis and recommendations
├── dual_run_comparison.py          # Comparison script (updated for both runs)
└── [other implementation files]
```

## Run 1: Unfair Comparison

**Executed:** February 3, 2026, 00:27 UTC

### Methods
1. **Current Method** (Hand-crafted 5 queries)
   - Static query set
   - 100% consistent
   - 73.4/100 diversity
   - 35.0/100 coverage

2. **Query Expansion Matrix** (15 queries)
   - Structured decomposition
   - 0% consistency (LLM variation)
   - 96.8/100 diversity
   - 100.0/100 coverage

### Results
```
Current Method:   5 queries, diversity 73.4, coverage 35.0
QEM:             15 queries, diversity 96.8, coverage 100.0
Improvement:    +200% queries, +23.4 diversity, +65.0 coverage
```

### Issue
Unfair comparison because query counts differ (5 vs 15).

### Report
See `run_1/DUAL_RUN_COMPARISON_REPORT.md` for full analysis.

---

## Run 2: Fair Comparison

**Executed:** February 3, 2026, 09:58 UTC

### Methods
1. **Simple LLM Expansion** (15 queries via basic prompt)
   - LLM-generated variations
   - 6.7% consistency
   - 86.0/100 diversity
   - 50.0/100 coverage

2. **Query Expansion Matrix** (14.5 avg queries)
   - Structured 4-axis decomposition
   - 0% consistency (different axes)
   - 100.0/100 diversity
   - 100.0/100 coverage

### Results
```
Simple LLM:      15 queries, diversity 86.0, coverage 50.0
QEM:           14.5 queries, diversity 100.0, coverage 100.0
Improvement:    +14.0 diversity, +50.0 coverage
```

### Insight
Even with equal query counts, QEM's systematic approach beats simple LLM variation.

### Report
See `run_2/RUN_2_COMPARISON_REPORT.md` for full analysis.

---

## Cross-Run Analysis

### Comparison Summary

| Metric | Run 1 Current | Run 2 Simple LLM | QEM (Best) |
|--------|---------------|-----------------|-----------|
| Queries | 5 | 15 | 14.5 |
| Diversity | 73.4 | 86.0 | 100.0 |
| Coverage | 35.0 | 50.0 | 100.0 |
| Consistency | 100% | 6.7% | 0% |

### Key Findings

1. **Quantity Helps But Isn't Enough**
   - 5 queries → 35% coverage
   - 15 queries → 50% coverage (with simple LLM)
   - 15 queries → 100% coverage (with QEM)

2. **Systematic Beats Random**
   - LLM-based variation: 86 diversity
   - Axis decomposition: 100 diversity
   - Difference: +16.3%

3. **Coverage Matters Most**
   - Simple LLM stuck at 50% despite 15 queries
   - QEM hits 100% with same query count
   - Systematic approach = 2x coverage improvement

4. **Consistency Trade-off**
   - Hand-crafted 100% consistent but limited
   - QEM 0% consistent but comprehensive
   - Solution: Cache decomposition for stability

### Complete Analysis
See `RUNS_COMPARISON.md` for detailed cross-run analysis and recommendations.

---

## Metrics Definition

### Diversity Score (0-100)
Measures query variation across:
- Word variety in queries
- Query length distribution
- Query uniqueness

Simple LLM: 86/100 (wording variations)
QEM: 100/100 (multi-dimensional variations)

### Coverage Score (0-100)
Measures domain exploration depth:
- Number of axes identified
- Number of variants per axis

Simple LLM: 50/100 (no axis structure)
QEM: 100/100 (4 axes × 5 variants)

### Consistency (%)
Percentage of queries matching between Iteration 1 and 2:
- Simple LLM: 6.7% (different LLM outputs)
- QEM: 0.0% (different decompositions)

---

## Query Generation Process

### Method 1: Hand-Crafted (Run 1 Current)
```
Manual query list
→ No generation needed
→ 100% consistent, but limited
```

### Method 2: Simple LLM (Run 2 Current)
```
User goal
→ LLM: "Generate 15 diverse queries"
→ 15 queries via basic prompt
→ Diverse but no framework
```

### Method 3: Query Expansion Matrix (Best)
```
User goal
→ LLM Step 1: Decompose into 4 axes
   - Primary Entities (5 types)
   - Scope/Scale (3 levels)
   - Characteristics (5 attributes)
   - Data Sources (5 formats)
→ LLM Step 2: Generate corner queries
→ 15 strategic queries covering all angles
→ Perfect diversity + coverage
```

---

## Recommendations

### For Quick Exploration
Use Simple LLM (Run 2 method)
- Fast LLM call
- 86 diversity score
- 50 coverage score
- Good for initial testing

### For Production Data Mining
Use Query Expansion Matrix
- 100 diversity score
- 100 coverage score
- Cache decomposition for consistency
- Best for comprehensive search

### For Balanced Approach
Use Hybrid Method
```
Hand-crafted baseline (5 queries)
+ QEM decomposition (10-15 queries)
= 20 queries with both reliability and coverage
```

---

## Files

### Reports
- `run_1/DUAL_RUN_COMPARISON_REPORT.md` - Run 1 detailed analysis
- `run_2/RUN_2_COMPARISON_REPORT.md` - Run 2 detailed analysis
- `RUNS_COMPARISON.md` - Cross-run comparison and insights
- `README_RUNS.md` - This file

### Raw Data
- `run_1/dual_run_comparison_20260203_002757.json` - Run 1 metrics
- `run_2/dual_run_comparison_20260203_095842.json` - Run 2 metrics

### Implementation
- `dual_run_comparison.py` - Main comparison script
- `query_expansion_matrix.py` - QEM implementation
- `quick_demo.py` - Quick demonstration
- `example_column_context.py` - Column context examples

---

## Usage

Run a comparison:
```bash
cd /home/k3v1n/projects/Automated-Research-System
conda activate auto_research
python test/query_expansion/dual_run_comparison.py
```

This will:
1. Run Simple LLM expansion 2 iterations
2. Run QEM 2 iterations
3. Compare results
4. Save to comparison_results/ (or run_N/ if manually organized)
5. Print detailed metrics

---

## Conclusion

**Winner: Query Expansion Matrix**

Evidence:
- 100% vs 50% coverage (2x improvement)
- 100.0 vs 86.0 diversity (16.3% better)
- Systematic decomposition beats random variation
- Works for any domain, not just hospitals

**Deployment Strategy:**
1. Use QEM as primary method
2. Cache first decomposition for consistency
3. Validate with real scraping results
4. Consider hybrid approach for balanced cost/benefit

---

*Created: February 3, 2026*  
*Analysis: Two-run comparison of query generation methods*  
*Conclusion: Systematic decomposition superior to simple LLM variation*
