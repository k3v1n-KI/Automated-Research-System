#!/usr/bin/env python3
"""
Dual Run Comparison: Execute both query generation methods twice
and compare completeness and coverage metrics.

This script:
1. Runs Current Method (Iteration 1 & 2)
2. Runs Query Expansion Matrix (Iteration 1 & 2)
3. Compares query diversity, coverage, and consistency
4. Measures decomposition stability across iterations
"""

import asyncio
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass, asdict
import sys

from dotenv import find_dotenv, load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from query_expansion_matrix import QueryExpansionMatrix


@dataclass
class QueryMetrics:
    """Metrics from query generation run"""
    method: str
    iteration: int
    query_count: int
    unique_queries: int
    avg_query_length: float
    coverage_score: float  # How well queries cover the domain
    diversity_score: float  # How diverse the queries are
    processing_time_seconds: float


# LLM Prompts for query generation

SIMPLE_QUERY_EXPANSION_PROMPT = """
You are a search query generation expert. Generate diverse search queries to find information about: {goal}

Create 15 natural, varied search queries that would find different relevant sources.
Vary the wording, terminology, and specificity to explore different angles.

Return ONLY a JSON array of strings, like:
["query 1", "query 2", "query 3", ...]

Make the queries realistic and diverse - not just synonym lists.
"""


def _calculate_query_diversity(queries: List[str]) -> float:
    """Calculate diversity score of queries (0-100)
    
    Measures:
    - Word variety
    - Query length variation
    - Semantic distinction
    """
    if not queries:
        return 0.0
    
    # Tokenize
    all_words = set()
    for q in queries:
        all_words.update(q.lower().split())
    
    # Word variety (0-50 points)
    unique_word_ratio = min(len(all_words) / (len(queries) * 2), 1.0) * 50
    
    # Length variation (0-30 points)
    lengths = [len(q) for q in queries]
    avg_length = sum(lengths) / len(lengths)
    length_variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
    length_score = min(length_variance / 100, 1.0) * 30
    
    # Query uniqueness (0-20 points)
    unique_count = len(set(queries))
    uniqueness_score = (unique_count / len(queries)) * 20
    
    return round(unique_word_ratio + length_score + uniqueness_score, 2)


def _calculate_coverage_score(axes: List) -> float:
    """Calculate coverage score based on decomposition axes
    
    More axes and variants = better coverage
    """
    if not axes:
        return 0.0
    
    total_variants = 0
    for axis in axes:
        if hasattr(axis, 'values'):
            total_variants += len(axis.values)
        elif isinstance(axis, dict) and 'values' in axis:
            total_variants += len(axis['values'])
    
    axis_count = len(axes)
    
    # Base score from number of axes
    axis_score = min(axis_count / 4, 1.0) * 50  # 0-50 points
    
    # Variant coverage
    variant_score = min(total_variants / 15, 1.0) * 50  # 0-50 points
    
    return round(axis_score + variant_score, 2)


def save_queries_to_csv(
    all_queries: Dict[str, List[str]],
    output_dir: Path,
    run_name: str
) -> None:
    """Save all queries from both methods to CSV files"""
    
    # Create CSVs for each method
    # Simple LLM CSV
    simple_csv = output_dir / f"simple_llm_queries_{run_name}.csv"
    simple_queries = []
    for iter_num, key in enumerate(["simple_iter1", "simple_iter2"], 1):
        for query in all_queries.get(key, []):
            simple_queries.append({
                "iteration": iter_num,
                "query": query
            })
    
    with open(simple_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["iteration", "query"])
        writer.writeheader()
        writer.writerows(simple_queries)
    
    print(f"   ✅ Saved {len(simple_queries)} Simple LLM queries to: {simple_csv.name}")
    
    # QEM CSV
    qem_csv = output_dir / f"qem_queries_{run_name}.csv"
    qem_queries = []
    for iter_num, key in enumerate(["qem_iter1", "qem_iter2"], 1):
        for query in all_queries.get(key, []):
            qem_queries.append({
                "iteration": iter_num,
                "query": query
            })
    
    with open(qem_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["iteration", "query"])
        writer.writeheader()
        writer.writerows(qem_queries)
    
    print(f"   ✅ Saved {len(qem_queries)} QEM queries to: {qem_csv.name}")


async def run_qem_decomposition(
    goal: str,
    iteration: int
) -> Tuple[QueryMetrics, List[str], List]:
    """Run Query Expansion Matrix decomposition"""
    print(f"      ⏳ Running QEM decomposition...")
    
    import time
    start_time = time.time()
    
    qem = QueryExpansionMatrix()
    result = await qem.execute(goal, strategy="full_matrix")
    
    processing_time = time.time() - start_time
    
    # Extract from QueryMatrix result
    queries = list(result.corner_queries) if hasattr(result, 'corner_queries') else []
    axes = list(result.axes) if hasattr(result, 'axes') else []
    
    diversity = _calculate_query_diversity(queries)
    coverage = _calculate_coverage_score(axes)
    
    metrics = QueryMetrics(
        method="Query Expansion Matrix",
        iteration=iteration,
        query_count=len(queries),
        unique_queries=len(set(queries)),
        avg_query_length=round(sum(len(q) for q in queries) / len(queries), 2) if queries else 0,
        coverage_score=coverage,
        diversity_score=diversity,
        processing_time_seconds=round(processing_time, 2)
    )
    
    return metrics, queries, axes


async def run_simple_llm_generation(
    goal: str,
    iteration: int
) -> Tuple[QueryMetrics, List[str]]:
    """Generate 15 queries using simple LLM prompt (Current Method replacement)"""
    print(f"      ⏳ Generating queries with simple LLM prompt...")
    
    import time
    start_time = time.time()
    
    from nodes.deduplicate import get_openai_client
    client, model = get_openai_client()
    
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": SIMPLE_QUERY_EXPANSION_PROMPT.format(goal=goal)}
        ]
    )
    
    processing_time = time.time() - start_time
    
    content = response.choices[0].message.content
    
    # Extract JSON array
    import json
    import re
    json_match = re.search(r'\[.*\]', content, re.DOTALL)
    if json_match:
        queries = json.loads(json_match.group())
    else:
        queries = []
    
    diversity = _calculate_query_diversity(queries)
    coverage = 50.0  # Simple baseline score
    
    metrics = QueryMetrics(
        method="LLM Simple Expansion",
        iteration=iteration,
        query_count=len(queries),
        unique_queries=len(set(queries)),
        avg_query_length=round(sum(len(q) for q in queries) / len(queries), 2) if queries else 0,
        coverage_score=coverage,
        diversity_score=diversity,
        processing_time_seconds=round(processing_time, 2)
    )
    
    return metrics, queries


async def run_dual_comparison(goal: str = "hospitals in ontario") -> None:
    """Run both methods twice and compare results"""
    load_dotenv(find_dotenv())
    
    print("\n" + "█" * 80)
    print("DUAL RUN COMPARISON: QUERY GENERATION QUALITY ANALYSIS (RUN 3)")
    print("█" * 80)
    print(f"\nGoal: {goal}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Save to run_3 folder
    results_dir = Path(__file__).parent / "run_3"
    results_dir.mkdir(exist_ok=True)
    print(f"📁 Results will be saved to: {results_dir}\n")
    
    all_metrics = []
    all_queries = {}
    all_axes = {}
    
    # =========================================================================
    # METHOD 1: Simple LLM Query Expansion (15 queries)
    # =========================================================================
    print("📊 METHOD 1: Simple LLM Query Expansion (15 queries via basic prompt)")
    print("-" * 80)
    
    # Run simple LLM method twice
    for iteration in [1, 2]:
        print(f"   🔄 Iteration {iteration}:")
        metrics, queries = await run_simple_llm_generation(goal, iteration)
        all_metrics.append(metrics)
        all_queries[f"simple_iter{iteration}"] = queries
        print(f"   ✅ Complete: {len(queries)} queries, diversity={metrics.diversity_score:.1f}, coverage={metrics.coverage_score:.1f}\n")
    
    # =========================================================================
    # METHOD 2: Query Expansion Matrix (Axes + Corner Check)
    # =========================================================================
    print("\n📊 METHOD 2: Query Expansion Matrix (Axes + Corner Check)")
    print("-" * 80)
    
    # Run QEM twice to test consistency
    for iteration in [1, 2]:
        print(f"   🔄 Iteration {iteration}:")
        metrics, queries, axes = await run_qem_decomposition(goal, iteration)
        all_metrics.append(metrics)
        all_queries[f"qem_iter{iteration}"] = queries
        all_axes[f"qem_iter{iteration}"] = axes
        print(f"   ✅ Complete: {len(queries)} queries, diversity={metrics.diversity_score:.1f}, coverage={metrics.coverage_score:.1f}\n")
    
    # =========================================================================
    # ANALYSIS & COMPARISON
    # =========================================================================
    print("\n" + "█" * 80)
    print("DETAILED METRICS COMPARISON")
    print("█" * 80)
    print("\n🔍 Analyzing results...\n")
    
    # Table 1: Individual Run Metrics
    print("\n1️⃣ INDIVIDUAL RUN METRICS")
    print("-" * 80)
    print(f"{'Method':<30} {'Iter':<5} {'Queries':<8} {'Unique':<8} {'Diversity':<10} {'Coverage':<10}")
    print("-" * 80)
    
    for m in all_metrics:
        print(f"{m.method:<30} {m.iteration:<5} {m.query_count:<8} {m.unique_queries:<8} {m.diversity_score:>8.1f} {m.coverage_score:>8.1f}")
    
    # Table 2: Consistency Analysis
    print("\n\n2️⃣ CONSISTENCY ANALYSIS (Iteration 1 vs 2)")
    print("-" * 80)
    
    current_metrics_list = [m for m in all_metrics if m.method == "LLM Simple Expansion"]
    matrix_metrics_list = [m for m in all_metrics if m.method == "Query Expansion Matrix"]
    
    # Simple LLM consistency
    simple_iter1 = current_metrics_list[0]
    simple_iter2 = current_metrics_list[1]
    simple_queries_iter1 = set(all_queries.get("simple_iter1", []))
    simple_queries_iter2 = set(all_queries.get("simple_iter2", []))
    
    if simple_queries_iter1 and simple_queries_iter2:
        simple_consistency = (len(simple_queries_iter1 & simple_queries_iter2) / max(len(simple_queries_iter1), len(simple_queries_iter2), 1)) * 100
    else:
        simple_consistency = 0.0
    
    # QEM consistency
    qem_iter1 = matrix_metrics_list[0]
    qem_iter2 = matrix_metrics_list[1]
    qem_queries_iter1 = set(all_queries.get("qem_iter1", []))
    qem_queries_iter2 = set(all_queries.get("qem_iter2", []))
    
    if qem_queries_iter1 and qem_queries_iter2:
        qem_consistency = (len(qem_queries_iter1 & qem_queries_iter2) / max(len(qem_queries_iter1), len(qem_queries_iter2), 1)) * 100
    else:
        qem_consistency = 0.0
    
    print(f"\n{'Simple LLM Expansion':<30} Query Consistency: {simple_consistency:>6.1f}%")
    print(f"{'Query Expansion Matrix':<30} Query Consistency: {qem_consistency:>6.1f}%")
    
    print(f"\n{'Simple LLM Expansion':<30}")
    print(f"  Iter 1 Diversity: {simple_iter1.diversity_score:>6.1f}  Coverage: {simple_iter1.coverage_score:>6.1f}")
    print(f"  Iter 2 Diversity: {simple_iter2.diversity_score:>6.1f}  Coverage: {simple_iter2.coverage_score:>6.1f}")
    
    print(f"\n{'Query Expansion Matrix':<30}")
    print(f"  Iter 1 Diversity: {qem_iter1.diversity_score:>6.1f}  Coverage: {qem_iter1.coverage_score:>6.1f}")
    print(f"  Iter 2 Diversity: {qem_iter2.diversity_score:>6.1f}  Coverage: {qem_iter2.coverage_score:>6.1f}")
    
    # Table 3: Summary Statistics
    print("\n\n3️⃣ AVERAGE PERFORMANCE BY METHOD")
    print("-" * 80)
    
    simple_avg_diversity = sum(m.diversity_score for m in current_metrics_list) / len(current_metrics_list)
    simple_avg_coverage = sum(m.coverage_score for m in current_metrics_list) / len(current_metrics_list)
    simple_avg_queries = sum(m.query_count for m in current_metrics_list) / len(current_metrics_list)
    
    qem_avg_diversity = sum(m.diversity_score for m in matrix_metrics_list) / len(matrix_metrics_list)
    qem_avg_coverage = sum(m.coverage_score for m in matrix_metrics_list) / len(matrix_metrics_list)
    qem_avg_queries = sum(m.query_count for m in matrix_metrics_list) / len(matrix_metrics_list)
    
    print(f"\n{'SIMPLE LLM EXPANSION':<40} (Iterations 1-2 Average)")
    print("-" * 80)
    print(f"  Average Queries:       {simple_avg_queries:>6.1f}")
    print(f"  Average Diversity:     {simple_avg_diversity:>6.1f}/100")
    print(f"  Average Coverage:      {simple_avg_coverage:>6.1f}/100")
    print(f"  Consistency:           {simple_consistency:>6.1f}%")
    
    print(f"\n{'QUERY EXPANSION MATRIX':<40} (Iterations 1-2 Average)")
    print("-" * 80)
    print(f"  Average Queries:       {qem_avg_queries:>6.1f}")
    print(f"  Average Diversity:     {qem_avg_diversity:>6.1f}/100")
    print(f"  Average Coverage:      {qem_avg_coverage:>6.1f}/100")
    print(f"  Consistency:           {qem_consistency:>6.1f}%")
    
    # Table 4: Comparative Gains
    print("\n\n4️⃣ COMPARATIVE ANALYSIS: QEM vs Simple LLM Expansion")
    print("-" * 80)
    
    diversity_gain = qem_avg_diversity - simple_avg_diversity
    coverage_gain = qem_avg_coverage - simple_avg_coverage
    query_increase = ((qem_avg_queries - simple_avg_queries) / simple_avg_queries * 100) if simple_avg_queries > 0 else 0
    
    print(f"  Diversity Improvement: {diversity_gain:>+6.1f} points  ({qem_avg_diversity:.1f} vs {simple_avg_diversity:.1f})")
    print(f"  Coverage Improvement:  {coverage_gain:>+6.1f} points  ({qem_avg_coverage:.1f} vs {simple_avg_coverage:.1f})")
    print(f"  Query Generation:      {query_increase:>+6.1f}%  ({qem_avg_queries:.1f} vs {simple_avg_queries:.1f})")
    
    # Save detailed results
    print("\n💾 Saving results to disk...")
    results = {
        "timestamp": datetime.now().isoformat(),
        "goal": goal,
        "metrics": [asdict(m) for m in all_metrics],
        "query_samples": {
            "simple_llm_iter1": all_queries.get("simple_iter1", [])[:5],
            "simple_llm_iter2": all_queries.get("simple_iter2", [])[:5],
            "qem_iter1": all_queries.get("qem_iter1", [])[:5],
            "qem_iter2": all_queries.get("qem_iter2", [])[:5]
        },
        "summary": {
            "simple_llm_expansion": {
                "avg_diversity_score": round(simple_avg_diversity, 2),
                "avg_coverage_score": round(simple_avg_coverage, 2),
                "avg_query_count": round(simple_avg_queries, 2),
                "consistency_pct": round(simple_consistency, 2)
            },
            "query_expansion_matrix": {
                "avg_diversity_score": round(qem_avg_diversity, 2),
                "avg_coverage_score": round(qem_avg_coverage, 2),
                "avg_query_count": round(qem_avg_queries, 2),
                "consistency_pct": round(qem_consistency, 2)
            },
            "comparative_gains": {
                "diversity_improvement": round(diversity_gain, 2),
                "coverage_improvement": round(coverage_gain, 2),
                "query_increase_pct": round(query_increase, 2)
            }
        }
    }
    
    # Save to file
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = results_dir / f"dual_run_comparison_{run_timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"   ✅ JSON results saved: {output_file.name}")
    
    # Save queries to CSV files
    print("   📄 Saving query datasets to CSV...")
    save_queries_to_csv(all_queries, results_dir, run_timestamp)
    
    print("\n" + "█" * 80)
    print("✨ RUN 3 COMPLETE")
    print("█" * 80)
    print(f"\n📊 All results saved to: {results_dir}/\n")


if __name__ == "__main__":
    asyncio.run(run_dual_comparison())
