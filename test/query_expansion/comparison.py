#!/usr/bin/env python3
"""
Comparison framework for Query Expansion methods
Metrics: Coverage, Completeness, Missing Values, Sample Count
"""

import asyncio
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple

import sys
from dotenv import find_dotenv, load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithm import build_research_algorithm, ProgressTracker
from query_expansion_matrix import QueryExpansionMatrix


@dataclass
class QueryMetrics:
    """Metrics for a single query or method"""
    method_name: str
    query_count: int
    total_urls: int
    validated_urls: int
    successfully_scraped: int
    extraction_count: int
    avg_fields_per_record: float
    missing_values_total: int
    missing_values_pct: float
    unique_records: int
    processing_time: float


def _build_initial_state(prompt: str) -> Dict[str, Any]:
    """Build initial state for algorithm"""
    return {
        "initial_prompt": prompt,
        "column_specs": [],
        "queries": [],
        "search_results": [],
        "validated_urls": [],
        "scraped_content": [],
        "extracted_items": [],
        "final_dataset": [],
        "session_id": f"query-test-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "round": 0,
        "error": None,
        "previous_session_id": None,
        "tweak_instructions": None,
        "previous_queries": [],
        "previous_items": [],
        "columns": [],
        "priority_columns": [],
    }


async def run_with_queries(
    goal: str,
    queries: List[str],
    method_name: str
) -> Tuple[QueryMetrics, List[Dict]]:
    """
    Run algorithm with specific queries injected
    """
    print(f"\n{'='*70}")
    print(f"Running: {method_name}")
    print(f"Queries: {len(queries)}")
    print(f"{'='*70}\n")
    
    algorithm_graph = build_research_algorithm(emit_fn=None)
    state = _build_initial_state(goal)
    
    # Inject custom queries instead of generating them
    state["queries"] = queries
    
    import time
    start_time = time.time()
    
    final_state = await algorithm_graph.ainvoke(state)
    
    processing_time = time.time() - start_time
    
    # Extract metrics
    search_results = final_state.get("search_results", [])
    validated_urls = final_state.get("validated_urls", [])
    scraped_content = final_state.get("scraped_content", [])
    extracted_items = final_state.get("extracted_items", [])
    
    # Calculate missing values
    missing_values = 0
    total_fields = 0
    
    if extracted_items:
        for item in extracted_items:
            for key, value in item.items():
                total_fields += 1
                if not value or (isinstance(value, list) and not value):
                    missing_values += 1
        
        avg_fields = total_fields / len(extracted_items) if extracted_items else 0
        missing_pct = (missing_values / total_fields * 100) if total_fields > 0 else 0
    else:
        avg_fields = 0
        missing_pct = 0
    
    metrics = QueryMetrics(
        method_name=method_name,
        query_count=len(queries),
        total_urls=len(search_results),
        validated_urls=len(validated_urls),
        successfully_scraped=len(scraped_content),
        extraction_count=len(extracted_items),
        avg_fields_per_record=avg_fields,
        missing_values_total=missing_values,
        missing_values_pct=missing_pct,
        unique_records=len(set(json.dumps(item, sort_keys=True) for item in extracted_items)),
        processing_time=processing_time
    )
    
    return metrics, extracted_items


async def run_comparison(goal: str = "hospitals in ontario") -> None:
    """
    Compare Query Expansion Matrix vs Current Method
    """
    load_dotenv(find_dotenv())
    
    print("\n" + "="*70)
    print("QUERY GENERATION METHOD COMPARISON")
    print("="*70)
    print(f"\nGoal: {goal}\n")
    
    results_dir = Path(__file__).parent / "comparison_results"
    results_dir.mkdir(exist_ok=True)
    
    # METHOD 1: Current (simple expansion)
    print("📊 METHOD 1: Current Query Generation")
    print("   Strategy: Simple synonym expansion\n")
    
    current_queries = [
        "hospitals in ontario",
        "hospital list ontario",
        "ontario hospital directory",
        "healthcare facilities ontario",
        "medical centers in ontario"
    ]
    
    current_metrics, current_items = await run_with_queries(
        goal, current_queries, "Current Method"
    )
    
    # METHOD 2: Query Expansion Matrix
    print("📊 METHOD 2: Query Expansion Matrix")
    print("   Strategy: Structured axes + corner check\n")
    
    qem = QueryExpansionMatrix()
    result = await qem.execute(goal, strategy="full_matrix")
    matrix_queries = result.corner_queries
    
    matrix_metrics, matrix_items = await run_with_queries(
        goal, matrix_queries, "Query Expansion Matrix"
    )
    
    # COMPARISON
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    
    comparison_data = {
        "timestamp": datetime.now().isoformat(),
        "goal": goal,
        "comparison": {
            "current_method": {
                "method_name": current_metrics.method_name,
                "query_count": current_metrics.query_count,
                "total_urls": current_metrics.total_urls,
                "validated_urls": current_metrics.validated_urls,
                "successfully_scraped": current_metrics.successfully_scraped,
                "extraction_count": current_metrics.extraction_count,
                "avg_fields_per_record": round(current_metrics.avg_fields_per_record, 2),
                "missing_values_total": current_metrics.missing_values_total,
                "missing_values_pct": round(current_metrics.missing_values_pct, 2),
                "unique_records": current_metrics.unique_records,
                "processing_time_seconds": round(current_metrics.processing_time, 2)
            },
            "query_expansion_matrix": {
                "method_name": matrix_metrics.method_name,
                "query_count": matrix_metrics.query_count,
                "total_urls": matrix_metrics.total_urls,
                "validated_urls": matrix_metrics.validated_urls,
                "successfully_scraped": matrix_metrics.successfully_scraped,
                "extraction_count": matrix_metrics.extraction_count,
                "avg_fields_per_record": round(matrix_metrics.avg_fields_per_record, 2),
                "missing_values_total": matrix_metrics.missing_values_total,
                "missing_values_pct": round(matrix_metrics.missing_values_pct, 2),
                "unique_records": matrix_metrics.unique_records,
                "processing_time_seconds": round(matrix_metrics.processing_time, 2)
            }
        }
    }
    
    # Print comparison table
    print(f"\n{'Metric':<30} {'Current':<20} {'Matrix':<20} {'Difference':<15}")
    print("-" * 85)
    
    metrics_list = [
        ("Query Count", current_metrics.query_count, matrix_metrics.query_count, True),
        ("Total URLs Found", current_metrics.total_urls, matrix_metrics.total_urls, True),
        ("Validated URLs", current_metrics.validated_urls, matrix_metrics.validated_urls, True),
        ("Successfully Scraped", current_metrics.successfully_scraped, matrix_metrics.successfully_scraped, True),
        ("Extracted Records", current_metrics.extraction_count, matrix_metrics.extraction_count, True),
        ("Avg Fields/Record", round(current_metrics.avg_fields_per_record, 2), round(matrix_metrics.avg_fields_per_record, 2), False),
        ("Missing Values", current_metrics.missing_values_total, matrix_metrics.missing_values_total, False),
        ("Missing Values %", round(current_metrics.missing_values_pct, 1), round(matrix_metrics.missing_values_pct, 1), False),
        ("Unique Records", current_metrics.unique_records, matrix_metrics.unique_records, True),
    ]
    
    for metric_name, current_val, matrix_val, higher_is_better in metrics_list:
        diff = matrix_val - current_val
        sign = "+" if diff >= 0 else ""
        print(f"{metric_name:<30} {str(current_val):<20} {str(matrix_val):<20} {sign}{diff:<14}")
    
    # Coverage & Completeness
    print("\n" + "="*70)
    print("KEY METRICS")
    print("="*70)
    
    coverage_current = (current_metrics.extraction_count / current_metrics.validated_urls * 100) if current_metrics.validated_urls > 0 else 0
    coverage_matrix = (matrix_metrics.extraction_count / matrix_metrics.validated_urls * 100) if matrix_metrics.validated_urls > 0 else 0
    
    completeness_current = 100 - current_metrics.missing_values_pct
    completeness_matrix = 100 - matrix_metrics.missing_values_pct
    
    print(f"\nCoverage (extraction rate):")
    print(f"  Current:  {coverage_current:.1f}%")
    print(f"  Matrix:   {coverage_matrix:.1f}%")
    print(f"  Winner:   {'Matrix' if coverage_matrix > coverage_current else 'Current' if coverage_current > coverage_matrix else 'Tie'} ({abs(coverage_matrix - coverage_current):.1f}% difference)")
    
    print(f"\nCompleteness (data quality):")
    print(f"  Current:  {completeness_current:.1f}%")
    print(f"  Matrix:   {completeness_matrix:.1f}%")
    print(f"  Winner:   {'Matrix' if completeness_matrix > completeness_current else 'Current' if completeness_current > completeness_matrix else 'Tie'} ({abs(completeness_matrix - completeness_current):.1f}% difference)")
    
    print(f"\nSampling Efficiency:")
    print(f"  Current:  {current_metrics.extraction_count / current_metrics.query_count:.1f} records/query")
    print(f"  Matrix:   {matrix_metrics.extraction_count / matrix_metrics.query_count:.1f} records/query")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = results_dir / f"comparison_{timestamp}.json"
    
    with open(json_path, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"\n✅ Results saved to: {json_path.name}\n")


if __name__ == "__main__":
    asyncio.run(run_comparison())
