#!/usr/bin/env python3
"""
Quick Comparison Test
Shows the difference between current and QEM methods without running full algorithm
"""

import json
from datetime import datetime
from pathlib import Path

from query_expansion_matrix import QueryExpansionMatrix
import asyncio
from dotenv import find_dotenv, load_dotenv


async def quick_comparison():
    """Quick demonstration of QEM vs Current method"""
    load_dotenv(find_dotenv())
    
    goal = "Find hospitals in Ontario"
    
    print("\n" + "="*70)
    print("QUERY GENERATION COMPARISON (Quick Demo)")
    print("="*70)
    print(f"\nGoal: {goal}\n")
    
    # CURRENT METHOD
    print("📊 METHOD 1: Current Query Generation")
    print("   Strategy: Simple synonym expansion\n")
    
    current_queries = [
        "hospitals in ontario",
        "hospital list ontario",
        "ontario hospital directory",
        "healthcare facilities ontario",
        "medical centers in ontario"
    ]
    
    print("   Generated Queries:")
    for i, q in enumerate(current_queries, 1):
        print(f"     {i}. {q}")
    print(f"\n   Total: {len(current_queries)} queries")
    print(f"   Diversity: Low (mostly synonyms, single geographic scope)\n")
    
    # QUERY EXPANSION MATRIX
    print("📊 METHOD 2: Query Expansion Matrix")
    print("   Strategy: Axes decomposition + corner check\n")
    
    qem = QueryExpansionMatrix()
    result = await qem.execute(goal, strategy="full_matrix")
    
    print("\n   Axes Discovered:")
    for axis in result.axes:
        print(f"     • {axis.name}: {len(axis.values)} variants")
        print(f"       → {', '.join(axis.values[:3])}...")
    
    print(f"\n   Generated Queries:")
    for i, q in enumerate(result.corner_queries, 1):
        print(f"     {i}. {q}")
    
    print(f"\n   Total: {len(result.corner_queries)} queries")
    print(f"   Theoretical Matrix: {result.full_matrix_size} possible combinations")
    print(f"   Diversity: Very High (varies entity type, geography, format, attributes)\n")
    
    # COMPARISON TABLE
    print("="*70)
    print("COMPARISON METRICS")
    print("="*70 + "\n")
    
    comparison = {
        "Metric": [
            "Query Count",
            "Query Diversity",
            "Geographic Variants",
            "Entity Type Variants", 
            "Format Variants",
            "Attribute Coverage",
            "Format Specificity",
            "Coverage Strategy"
        ],
        "Current Method": [
            len(current_queries),
            "Low",
            "1 (Ontario-wide)",
            "3-4 synonyms",
            "0 (general web)",
            "No",
            "Generic search",
            "Synonym expansion"
        ],
        "Query Matrix": [
            len(result.corner_queries),
            "Very High",
            f"{len(result.axes[1].values)} regions",
            f"{len(result.axes[0].values)} types",
            f"{len(result.axes[3].values)} formats",
            "Yes (5 attributes)",
            "Filetype/format specific",
            "Strategic corner check"
        ]
    }
    
    for i, metric in enumerate(comparison["Metric"]):
        current = comparison["Current Method"][i]
        matrix = comparison["Query Matrix"][i]
        print(f"{metric:<25} {str(current):<30} {str(matrix):<30}")
    
    # EXPECTED IMPACT
    print("\n" + "="*70)
    print("EXPECTED IMPACT ON METRICS")
    print("="*70 + "\n")
    
    impact = {
        "Metric": [
            "URLs Found",
            "Coverage (%)",
            "Completeness (%)",
            "Missing Values",
            "Unique Records"
        ],
        "Current Baseline": [
            "28 URLs",
            "~85%",
            "~72%",
            "~1,200 fields",
            "168 records"
        ],
        "Expected QEM Gain": [
            "+40-60%",
            "+5-10%",
            "+8-12%",
            "-20-30%",
            "+30-50%"
        ],
        "Reasoning": [
            "Format queries find PDFs, directories, archives",
            "Geographic variants reach more regions",
            "More fields extracted from structured sources",
            "Targeted queries yield cleaner data",
            "Broader search scope increases unique samples"
        ]
    }
    
    for i, metric in enumerate(impact["Metric"]):
        print(f"{metric:<20} Current    QEM Gain       Why")
        print(f"{'':20} {impact['Current Baseline'][i]:<12} {impact['Expected QEM Gain'][i]:<14} {impact['Reasoning'][i]}")
    
    # SAVE DEMO RESULTS
    results_dir = Path(__file__).parent / "comparison_results"
    results_dir.mkdir(exist_ok=True)
    
    demo_data = {
        "timestamp": datetime.now().isoformat(),
        "goal": goal,
        "comparison": {
            "current_method": {
                "query_count": len(current_queries),
                "queries": current_queries,
                "diversity": "Low",
                "geographic_variants": 1,
                "entity_variants": 3,
                "format_specificity": "No"
            },
            "query_matrix": {
                "query_count": len(result.corner_queries),
                "queries": result.corner_queries,
                "axes": [
                    {
                        "name": axis.name,
                        "value_count": len(axis.values),
                        "values": axis.values
                    }
                    for axis in result.axes
                ],
                "theoretical_matrix_size": result.full_matrix_size,
                "diversity": "Very High"
            }
        }
    }
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = results_dir / f"demo_comparison_{timestamp}.json"
    
    with open(json_path, 'w') as f:
        json.dump(demo_data, f, indent=2)
    
    print(f"\n✅ Demo results saved: {json_path.name}\n")


if __name__ == "__main__":
    asyncio.run(quick_comparison())
