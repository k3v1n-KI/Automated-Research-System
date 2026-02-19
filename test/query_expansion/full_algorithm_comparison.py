#!/usr/bin/env python3
"""
Full Algorithm Comparison: Run complete research pipeline with both query methods
and compare the resulting datasets.

This script:
1. Generates 6 queries with Simple LLM
2. Generates 6 queries with QEM
3. Runs full research algorithm on each query set
4. Compares the datasets produced (completeness, coverage, quality)
"""

import asyncio
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass, asdict
import sys

from dotenv import find_dotenv, load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from query_expansion_matrix import QueryExpansionMatrix
from algorithm import build_research_algorithm


@dataclass
class DatasetMetrics:
    """Metrics from research algorithm execution"""
    method: str
    queries_used: int
    total_rows: int
    unique_rows: int
    columns_count: int
    completeness_score: float  # % of non-null values
    execution_time_seconds: float
    sources_found: int


SIMPLE_QUERY_EXPANSION_PROMPT = """
You are a search query generation expert. Generate diverse search queries to find information about: {goal}

Create 6 natural, varied search queries that would find different relevant sources.
Vary the wording, terminology, and specificity to explore different angles.

Return ONLY a JSON array of strings, like:
["query 1", "query 2", "query 3", ...]

Make the queries realistic and diverse - not just synonym lists.
"""


async def generate_simple_llm_queries(goal: str, num_queries: int = 6) -> List[str]:
    """Generate queries using simple LLM prompt"""
    print(f"   ⏳ Generating {num_queries} queries with simple LLM...")
    
    from nodes.deduplicate import get_openai_client
    client, model = get_openai_client()
    
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": SIMPLE_QUERY_EXPANSION_PROMPT.format(goal=goal)}],
    )
    
    content = response.choices[0].message.content
    
    import re
    json_match = re.search(r"\[.*\]", content, re.DOTALL)
    if json_match:
        queries = json.loads(json_match.group())[:num_queries]
    else:
        queries = []
    
    print(f"   ✅ Generated {len(queries)} queries")
    return queries


async def generate_qem_queries(goal: str, num_queries: int = 6) -> List[str]:
    """Generate queries using QEM"""
    print(f"   ⏳ Generating {num_queries} queries with QEM...")
    
    qem = QueryExpansionMatrix()
    
    # Use corner_only strategy for 6 queries
    result = await qem.execute(goal, strategy="corner_only")
    
    # Extract queries from result object
    if hasattr(result, "corner_queries"):
        queries = list(result.corner_queries)
    elif isinstance(result, dict) and "corner_queries" in result:
        queries = list(result["corner_queries"])
    else:
        queries = []
    
    # Limit to requested number
    queries = queries[:num_queries]
    
    print(f"   ✅ Generated {len(queries)} queries")
    return queries


async def run_research_algorithm(queries: List[str], goal: str, output_dir: Path, method_name: str, column_specs: List[Dict] = None) -> Tuple[DatasetMetrics, Path]:
    """Run full research algorithm with given queries"""
    print(f"   🔬 Running full research algorithm on {len(queries)} queries...")
    print(f"      This may take several minutes...")
    
    import time
    from langgraph.graph import StateGraph, START, END
    from nodes.search import SearchNode
    from nodes.validate import ValidateNode
    from nodes.scrape import ScrapeNode
    from nodes.extract import ExtractNode
    from nodes.deduplicate import DeduplicateNode
    from algorithm import ResearchState, ProgressTracker
    
    start_time = time.time()
    
    # Build custom algorithm that SKIPS query generation (we provide queries)
    progress = ProgressTracker(None)
    
    search_node = SearchNode()
    validate_node = ValidateNode(threshold=0.5)
    scrape_node = ScrapeNode(timeout_ms=15000)
    extract_node = ExtractNode(char_limit=3000)
    deduplicate_node = DeduplicateNode()
    
    # Create graph starting from search (skip query generation)
    graph = StateGraph(ResearchState)
    
    def make_node(node_instance):
        async def wrapped(state):
            return await node_instance.execute(state, progress)
        return wrapped
    
    graph.add_node("search", make_node(search_node))
    graph.add_node("validate", make_node(validate_node))
    graph.add_node("scrape", make_node(scrape_node))
    graph.add_node("extract", make_node(extract_node))
    graph.add_node("deduplicate", make_node(deduplicate_node))
    
    # Start from search (queries already provided)
    graph.add_edge(START, "search")
    graph.add_edge("search", "validate")
    graph.add_edge("validate", "scrape")
    graph.add_edge("scrape", "extract")
    graph.add_edge("extract", "deduplicate")
    graph.add_edge("deduplicate", END)
    
    algorithm = graph.compile()
    
    # Create initial state with pre-generated queries
    session_id = f"run4_{method_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    state = {
        "initial_prompt": goal,
        "column_specs": column_specs if column_specs else [],  # Use specified columns or let algorithm infer
        "queries": queries,  # Pre-generated queries from our methods
        "search_results": [],
        "validated_urls": [],
        "scraped_content": [],
        "extracted_items": [],
        "final_dataset": [],
        "session_id": session_id,
        "round": 1,
        "error": None,
        "previous_session_id": None,
        "tweak_instructions": None,
        "previous_queries": [],
        "previous_items": [],
        "columns": [],
        "priority_columns": []
    }
    
    # Run the algorithm (starting from search node)
    print(f"      Starting algorithm execution with {len(queries)} queries...")
    result = await algorithm.ainvoke(state)
    
    execution_time = time.time() - start_time
    
    # Get dataset from result
    dataset = result.get("final_dataset", [])
    
    if not dataset:
        print(f"   ⚠️  No dataset generated")
        return DatasetMetrics(
            method=method_name,
            queries_used=len(queries),
            total_rows=0,
            unique_rows=0,
            columns_count=0,
            completeness_score=0.0,
            execution_time_seconds=execution_time,
            sources_found=0
        ), None
    
    # Save dataset to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_file = output_dir / f"dataset_{timestamp}.csv"
    
    # Convert to DataFrame and save
    df = pd.DataFrame(dataset)
    df.to_csv(dataset_file, index=False)
    print(f"      📊 Saved dataset CSV: {dataset_file.name} ({len(df)} rows)")
    
    # Analyze the dataset
    try:
        # Convert any dict columns to strings for analysis
        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, (dict, list))).any():
                df[col] = df[col].astype(str)
        
        total_rows = len(df)
        unique_rows = df.drop_duplicates().shape[0]
        columns_count = len(df.columns)
        
        # Calculate completeness (% of non-null values)
        total_cells = df.size
        non_null_cells = df.count().sum()
        completeness_score = (non_null_cells / total_cells * 100) if total_cells > 0 else 0.0
        
        # Count unique sources if there's a source column
        sources_found = 0
        source_cols = [col for col in df.columns if 'source' in col.lower() or 'url' in col.lower()]
        if source_cols:
            sources_found = df[source_cols[0]].nunique()
        
        print(f"   ✅ Algorithm complete: {total_rows} rows, {columns_count} columns, {completeness_score:.1f}% complete")
        
        return DatasetMetrics(
            method=method_name,
            queries_used=len(queries),
            total_rows=total_rows,
            unique_rows=unique_rows,
            columns_count=columns_count,
            completeness_score=round(completeness_score, 2),
            execution_time_seconds=round(execution_time, 2),
            sources_found=sources_found
        ), dataset_file
        
    except Exception as e:
        print(f"   ⚠️  Error analyzing dataset: {e}")
        return DatasetMetrics(
            method=method_name,
            queries_used=len(queries),
            total_rows=0,
            unique_rows=0,
            columns_count=0,
            completeness_score=0.0,
            execution_time_seconds=round(execution_time, 2),
            sources_found=0
        ), dataset_file


async def run_full_comparison(goal: str = "hospitals in ontario", num_queries: int = 6) -> None:
    """Run both methods with full algorithm and compare datasets"""
    load_dotenv(find_dotenv())
    
    # Generate timestamp early for use throughout
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "█" * 80)
    print("FULL ALGORITHM COMPARISON: END-TO-END DATASET EVALUATION (RUN 4)")
    print("█" * 80)
    print(f"\nGoal: {goal}")
    print(f"Queries per method: {num_queries}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Setup directories
    results_dir = Path(__file__).parent / "run_4"
    results_dir.mkdir(exist_ok=True)
    
    simple_dir = results_dir / "simple_llm"
    simple_dir.mkdir(exist_ok=True)
    
    qem_dir = results_dir / "qem"
    qem_dir.mkdir(exist_ok=True)
    
    print(f"📁 Results will be saved to: {results_dir}\n")
    
    # Define desired columns for extraction
    column_specs = [
        {"name": "name", "description": "Hospital name"},
        {"name": "address", "description": "Street address"},
        {"name": "city", "description": "City name"},
        {"name": "phone", "description": "Phone number"}
    ]
    
    # =========================================================================
    # METHOD 1: Simple LLM Query Generation + Full Algorithm
    # =========================================================================
    print("📊 METHOD 1: Simple LLM Query Expansion")
    print("-" * 80)
    
    simple_queries = await generate_simple_llm_queries(goal, num_queries)
    
    print(f"\n   Generated queries:")
    for i, q in enumerate(simple_queries, 1):
        print(f"      {i}. {q}")
    
    print(f"\n   🚀 Running full research algorithm...")
    simple_metrics, simple_dataset = await run_research_algorithm(simple_queries, goal, simple_dir, "Simple LLM", column_specs)
    
    # =========================================================================
    # METHOD 2: QEM Query Generation + Full Algorithm
    # =========================================================================
    print("\n\n📊 METHOD 2: Query Expansion Matrix")
    print("-" * 80)
    
    qem_queries = await generate_qem_queries(goal, num_queries)
    
    print(f"\n   Generated queries:")
    for i, q in enumerate(qem_queries, 1):
        print(f"      {i}. {q}")
    
    print(f"\n   🚀 Running full research algorithm...")
    qem_metrics, qem_dataset = await run_research_algorithm(qem_queries, goal, qem_dir, "Query Expansion Matrix", column_specs)
    
    # =========================================================================
    # DATASET COMPARISON
    # =========================================================================
    print("\n\n" + "█" * 80)
    print("DATASET COMPARISON")
    print("█" * 80)
    print("\n🔍 Analyzing results...\n")
    
    print("\n1️⃣ DATASET METRICS")
    print("-" * 80)
    print(f"{'Method':<30} {'Rows':<10} {'Unique':<10} {'Columns':<10} {'Complete%':<12} {'Sources':<10}")
    print("-" * 80)
    print(f"{simple_metrics.method:<30} {simple_metrics.total_rows:<10} {simple_metrics.unique_rows:<10} "
          f"{simple_metrics.columns_count:<10} {simple_metrics.completeness_score:<12.1f} {simple_metrics.sources_found:<10}")
    print(f"{qem_metrics.method:<30} {qem_metrics.total_rows:<10} {qem_metrics.unique_rows:<10} "
          f"{qem_metrics.columns_count:<10} {qem_metrics.completeness_score:<12.1f} {qem_metrics.sources_found:<10}")
    
    print("\n\n2️⃣ EXECUTION TIME")
    print("-" * 80)
    print(f"Simple LLM:             {simple_metrics.execution_time_seconds:>8.1f}s")
    print(f"Query Expansion Matrix: {qem_metrics.execution_time_seconds:>8.1f}s")
    
    print("\n\n3️⃣ COMPARATIVE ANALYSIS")
    print("-" * 80)
    
    row_diff = qem_metrics.total_rows - simple_metrics.total_rows
    row_pct = (row_diff / simple_metrics.total_rows * 100) if simple_metrics.total_rows > 0 else 0
    
    col_diff = qem_metrics.columns_count - simple_metrics.columns_count
    
    complete_diff = qem_metrics.completeness_score - simple_metrics.completeness_score
    
    source_diff = qem_metrics.sources_found - simple_metrics.sources_found
    source_pct = (source_diff / simple_metrics.sources_found * 100) if simple_metrics.sources_found > 0 else 0
    
    print(f"  Row Count Difference:       {row_diff:>+6} rows  ({row_pct:>+6.1f}%)")
    print(f"  Column Count Difference:    {col_diff:>+6} columns")
    print(f"  Completeness Difference:    {complete_diff:>+6.1f} percentage points")
    print(f"  Source Diversity:           {source_diff:>+6} sources  ({source_pct:>+6.1f}%)")
    
    # Save comprehensive results
    print("\n\n💾 Saving results...")
    
    # Save queries to CSV for easy reference
    queries_df = pd.DataFrame({
        'simple_llm': simple_queries + [''] * (max(len(simple_queries), len(qem_queries)) - len(simple_queries)),
        'qem': qem_queries + [''] * (max(len(simple_queries), len(qem_queries)) - len(qem_queries))
    })
    queries_file = results_dir / f"queries_used_{run_timestamp}.csv"
    queries_df.to_csv(queries_file, index=False)
    print(f"   ✅ Saved queries comparison: {queries_file.name}")
    
    # Save metrics
    results = {
        "timestamp": datetime.now().isoformat(),
        "goal": goal,
        "num_queries": num_queries,
        "simple_llm": {
            "queries": simple_queries,
            "metrics": asdict(simple_metrics),
            "dataset_file": str(simple_dataset.name) if simple_dataset else None
        },
        "qem": {
            "queries": qem_queries,
            "metrics": asdict(qem_metrics),
            "dataset_file": str(qem_dataset.name) if qem_dataset else None
        },
        "comparison": {
            "row_difference": row_diff,
            "row_difference_pct": round(row_pct, 2),
            "column_difference": col_diff,
            "completeness_difference": round(complete_diff, 2),
            "source_difference": source_diff,
            "source_difference_pct": round(source_pct, 2)
        }
    }
    
    output_file = results_dir / f"full_comparison_{run_timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"   ✅ Results JSON: {output_file.name}")
    print(f"   ✅ Simple LLM dataset: {simple_dir.name}/{simple_dataset.name if simple_dataset else 'N/A'}")
    print(f"   ✅ QEM dataset: {qem_dir.name}/{qem_dataset.name if qem_dataset else 'N/A'}")
    
    # Create a summary file documenting all outputs
    summary_md = f"""# Run 4 - Full Algorithm Comparison Summary

## Execution Details
- **Timestamp**: {datetime.now().isoformat()}
- **Goal**: {goal}
- **Queries per method**: {num_queries}

## Output Files

### Simple LLM Method
- **Directory**: `simple_llm/`
- **Dataset CSV**: `dataset_*.csv` - Contains {simple_metrics.total_rows} rows
- **Metrics**:
  - Rows: {simple_metrics.total_rows}
  - Unique rows: {simple_metrics.unique_rows}
  - Columns: {simple_metrics.columns_count}
  - Completeness: {simple_metrics.completeness_score}%
  - Sources found: {simple_metrics.sources_found}
  - Execution time: {simple_metrics.execution_time_seconds}s

### Query Expansion Matrix (QEM) Method
- **Directory**: `qem/`
- **Dataset CSV**: `dataset_*.csv` - Contains {qem_metrics.total_rows} rows
- **Metrics**:
  - Rows: {qem_metrics.total_rows}
  - Unique rows: {qem_metrics.unique_rows}
  - Columns: {qem_metrics.columns_count}
  - Completeness: {qem_metrics.completeness_score}%
  - Sources found: {qem_metrics.sources_found}
  - Execution time: {qem_metrics.execution_time_seconds}s

### Comparison Data
- **Full comparison JSON**: `full_comparison_*.json`
- **Queries used CSV**: `queries_used_*.csv`

## Key Findings

| Metric | Simple LLM | QEM | Difference |
|--------|-----------|-----|-----------|
| Rows | {simple_metrics.total_rows} | {qem_metrics.total_rows} | {row_diff:+d} ({row_pct:+.1f}%) |
| Columns | {simple_metrics.columns_count} | {qem_metrics.columns_count} | {col_diff:+d} |
| Completeness | {simple_metrics.completeness_score:.1f}% | {qem_metrics.completeness_score:.1f}% | {complete_diff:+.1f}pp |
| Sources | {simple_metrics.sources_found} | {qem_metrics.sources_found} | {source_diff:+d} ({source_pct:+.1f}%) |
| Time | {simple_metrics.execution_time_seconds}s | {qem_metrics.execution_time_seconds}s | - |

## How to Verify

1. **Check dataset CSVs** in `simple_llm/` and `qem/` directories
2. **Review queries used** in `queries_used_*.csv`
3. **See full metrics** in `full_comparison_*.json`
4. **Compare row counts** between the two dataset CSVs
5. **Analyze columns** to see which method discovered more/better data
"""
    
    summary_file = results_dir / "RUN_4_SUMMARY.md"
    with open(summary_file, "w") as f:
        f.write(summary_md)
    print(f"   ✅ Summary markdown: {summary_file.name}")
    
    print("\n" + "█" * 80)
    print("✨ RUN 4 COMPLETE")
    print("█" * 80)
    print(f"\n📊 All results saved to: {results_dir}/")
    print(f"\n📋 Files created:")
    print(f"   - simple_llm/dataset_*.csv ({simple_metrics.total_rows} rows)")
    print(f"   - qem/dataset_*.csv ({qem_metrics.total_rows} rows)")
    print(f"   - queries_used_*.csv")
    print(f"   - full_comparison_*.json")
    print(f"   - RUN_4_SUMMARY.md\n")


if __name__ == "__main__":
    asyncio.run(run_full_comparison())
