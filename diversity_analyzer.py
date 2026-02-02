"""
Diversity analysis module - Calculates Gini-Simpson diversity index for datasets
Uses priority columns to track dataset diversity across research iterations
"""

from typing import List, Dict, Any, Tuple
from collections import Counter
import json


def calculate_gini_simpson_index(values: List[Any]) -> float:
    """
    Calculate Gini-Simpson Diversity Index (1 - sum((n/N)^2))
    
    Formula: 1 - Σ(n_i/N)²
    where:
        n_i = count of each unique value
        N = total number of items
        
    Results:
        0.0 = Zero diversity (all values identical)
        1.0 = Maximum diversity (all values unique)
        
    Args:
        values: List of values to analyze
        
    Returns:
        float: Diversity index between 0.0 and 1.0
    """
    if not values or len(values) == 0:
        return 0.0
    
    # Count occurrences
    counts = Counter(values)
    total = len(values)
    
    # Calculate sum of squared proportions
    sum_squared_proportions = sum((count / total) ** 2 for count in counts.values())
    
    # Diversity index = 1 - sum
    diversity_index = 1 - sum_squared_proportions
    
    return round(diversity_index, 4)


def get_value_distribution(values: List[Any]) -> Dict[str, Any]:
    """
    Get distribution of values with counts and percentages
    
    Args:
        values: List of values to analyze
        
    Returns:
        Dict with:
            - unique_count: Number of unique values
            - total_count: Total number of items
            - distribution: List of {value, count, percentage}
            - sorted by count (descending)
    """
    if not values or len(values) == 0:
        return {
            "unique_count": 0,
            "total_count": 0,
            "distribution": [],
            "diversity_index": 0.0
        }
    
    counts = Counter(values)
    total = len(values)
    
    distribution = [
        {
            "value": str(value) if value is not None else "None",
            "count": count,
            "percentage": round((count / total) * 100, 2)
        }
        for value, count in counts.most_common()
    ]
    
    return {
        "unique_count": len(counts),
        "total_count": total,
        "distribution": distribution,
        "diversity_index": calculate_gini_simpson_index(values)
    }


def analyze_dataset_diversity(
    records: List[Dict],
    priority_columns: List[str]
) -> Dict[str, Any]:
    """
    Analyze diversity of a dataset across priority columns
    
    Args:
        records: List of record dictionaries
        priority_columns: List of column names to analyze for diversity
        
    Returns:
        Dict with diversity analysis for each priority column:
        {
            "column_name": {
                "unique_count": int,
                "total_count": int,
                "diversity_index": float,
                "distribution": [...]
            },
            "overall_diversity": float (average across all priority columns),
            "analysis_timestamp": string
        }
    """
    if not records or not priority_columns:
        return {
            "column_analyses": {},
            "overall_diversity": 0.0,
            "analysis_timestamp": None
        }
    
    column_analyses = {}
    diversity_scores = []
    
    for column in priority_columns:
        # Extract values from all records
        values = [
            record.get(column, "")
            for record in records
            if column in record
        ]
        
        if values:
            analysis = get_value_distribution(values)
            column_analyses[column] = analysis
            diversity_scores.append(analysis["diversity_index"])
    
    # Calculate overall diversity (average of all priority columns)
    overall_diversity = (
        sum(diversity_scores) / len(diversity_scores)
        if diversity_scores
        else 0.0
    )
    
    return {
        "column_analyses": column_analyses,
        "overall_diversity": round(overall_diversity, 4),
        "priority_columns": priority_columns,
        "total_records": len(records),
        "unique_columns": len(set(col for col in priority_columns if col in [c for r in records for c in r]))
    }


def generate_diversity_report(
    session_id: str,
    records: List[Dict],
    priority_columns: List[str],
    round: int = 0
) -> Dict[str, Any]:
    """
    Generate comprehensive diversity report for a research session
    
    Args:
        session_id: Research session ID
        records: Dataset records
        priority_columns: Columns to analyze
        round: Which iteration of research
        
    Returns:
        Formatted report suitable for display
    """
    diversity_analysis = analyze_dataset_diversity(records, priority_columns)
    
    # Build column analysis list for frontend
    column_analysis = []
    for column, analysis in diversity_analysis["column_analyses"].items():
        column_analysis.append({
            "column": column,
            "unique_count": analysis["unique_count"],
            "total_count": analysis["total_count"],
            "diversity_index": analysis["diversity_index"],
            "value_breakdown": analysis["distribution"]
        })
    
    report = {
        "session_id": session_id,
        "round": round,
        "total_records": len(records),
        "priority_columns": priority_columns,
        "overall_diversity_index": diversity_analysis["overall_diversity"],
        "column_analysis": column_analysis,
        "summary": ""
    }
    
    # Generate summary text
    diversity = diversity_analysis["overall_diversity"]
    if diversity < 0.3:
        diversity_level = "Low"
    elif diversity < 0.6:
        diversity_level = "Moderate"
    else:
        diversity_level = "High"
    
    report["summary"] = (
        f"Round {round}: {len(records)} records | "
        f"Diversity Index: {diversity:.2%} ({diversity_level}) | "
        f"Unique values in priority columns: {sum(d['unique_count'] for d in column_analysis)}"
    )
    
    return report


def compare_diversity_across_rounds(
    reports: List[Dict]
) -> Dict[str, Any]:
    """
    Compare diversity metrics across multiple research rounds
    
    Args:
        reports: List of diversity reports from different rounds
        
    Returns:
        Comparison showing trends
    """
    if not reports:
        return {}
    
    return {
        "total_rounds": len(reports),
        "records_progression": [r["total_records"] for r in reports],
        "diversity_progression": [r["overall_diversity_index"] for r in reports],
        "trend": (
            "Improving" if len(reports) > 1 and reports[-1]["overall_diversity_index"] > reports[-2]["overall_diversity_index"]
            else "Declining" if len(reports) > 1 and reports[-1]["overall_diversity_index"] < reports[-2]["overall_diversity_index"]
            else "Stable"
        ),
        "final_diversity": reports[-1]["overall_diversity_index"],
        "final_record_count": reports[-1]["total_records"]
    }
