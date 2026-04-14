#!/usr/bin/env python3
"""
Query Expansion Matrix Strategy
Implements a structured approach to generate diverse queries for improved coverage.

Architecture:
    Step 1: Zero-Shot Decomposition (identify axes of variance)
    Step 2: Corner Check Strategy (initial permutation for broad coverage)
    
The idea: Instead of generating many similar queries, we strategically sample
the "corners" of the query space to ensure we explore all dimensions early.
"""

import json
import re
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import find_dotenv, load_dotenv


@dataclass
class AxisOfVariance:
    """Represents one dimension of the query matrix"""
    name: str  # "Geographic Levels", "Entity Types", etc.
    values: List[str]  # ["Ontario", "Greater Toronto Area", "Rural Ontario"]
    importance: int = 1  # Weight in query generation


@dataclass
class QueryMatrix:
    """The decomposed axes and initial queries"""
    goal: str
    axes: List[AxisOfVariance]
    corner_queries: List[str]
    full_matrix_size: int  # Potential total queries


class ZeroShotDecomposer:
    """Step 1: Decompose research goal into axes of variance"""
    
    DECOMPOSITION_PROMPT_BASE = """
You are a research query strategist specializing in comprehensive data discovery.
Decompose the research goal into structured axes of variance to maximize coverage.

For each axis, provide 5 representative values that would yield different results.
Think about what a user might vary when searching for this information.

Return ONLY valid JSON with this structure:
{{
    "primary_entities": {{"name": "Primary Entity Types", "values": ["variant1", "variant2", "variant3"]}},
    "scope": {{"name": "Scope/Scale", "values": ["broad", "medium", "narrow"]}},
    "characteristics": {{"name": "Key Characteristics", "values": ["attr1", "attr2"]}},
    "sources": {{"name": "Data Sources", "values": ["format1", "format2"]}}
}}

Principles:
- Entities: What types or categories of the main subject exist?
- Scope: What geographic, temporal, or organizational levels matter?
- Characteristics: What distinguishing attributes or qualities vary?
- Sources: What data formats or source types would be relevant?

Focus on real-world variance that would affect search results.
"""

    DECOMPOSITION_PROMPT_WITH_COLUMNS = """
You are a research query strategist specializing in comprehensive data discovery.
Decompose the research goal into structured axes of variance to maximize coverage.

The user wants to extract these specific fields:
{columns}

Use these columns to guide your axis identification. Your decomposition should 
help find diverse sources that might contain or inform these fields.

For each axis, provide 5 representative values that would yield different results.
Think about what variations would help capture all these data points.

Return ONLY valid JSON with this structure:
{{
    "primary_entities": {{"name": "Primary Entity Types", "values": ["variant1", "variant2", "variant3"]}},
    "scope": {{"name": "Scope/Scale", "values": ["broad", "medium", "narrow"]}},
    "characteristics": {{"name": "Key Characteristics", "values": ["attr1", "attr2"]}},
    "sources": {{"name": "Data Sources", "values": ["format1", "format2"]}}
}}

Principles:
- Entities: What types or categories of the main subject exist?
- Scope: What geographic, temporal, or organizational levels matter?
- Characteristics: What distinguishing attributes or qualities vary?
- Sources: What data formats or source types would have these fields?

Focus on variants that would improve coverage of the requested columns.
"""
    
    def __init__(self):
        from nodes.deduplicate import get_openai_client
        self.client, self.model = get_openai_client()
    
    async def decompose(self, goal: str, column_specs: List[Dict] = None) -> Dict[str, AxisOfVariance]:
        """Break down the goal into axes of variance
        
        Args:
            goal: Research goal (e.g., "Find hospitals in Ontario")
            column_specs: Optional list of column definitions with names/descriptions
                         to use as context for better axis identification
        """
        print(f"📐 Step 1: Zero-Shot Decomposition")
        print(f"   Goal: {goal}")
        if column_specs:
            print(f"   Columns: {len(column_specs)} fields to extract\n")
        else:
            print()

        
        try:
            # Select prompt based on whether columns were provided
            if column_specs:
                columns_text = "\n".join([
                    f"- {col.get('name', f'Field {i+1}')}: {col.get('description', 'N/A')}"
                    for i, col in enumerate(column_specs)
                ])
                system_prompt = self.DECOMPOSITION_PROMPT_WITH_COLUMNS.format(columns=columns_text)
            else:
                system_prompt = self.DECOMPOSITION_PROMPT_BASE
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Goal: {goal}"}
                ]
            )
            
            content = response.choices[0].message.content
            
            # Extract JSON
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in response")
            
            decomposition = json.loads(json_match.group())
            
            # Convert to AxisOfVariance objects
            axes = {}
            for key, axis_data in decomposition.items():
                axes[key] = AxisOfVariance(
                    name=axis_data.get("name", key.title()),
                    values=axis_data.get("values", []),
                    importance=axis_data.get("importance", 1)
                )
            
            # Log results
            print("   ✓ Axes identified:")
            for key, axis in axes.items():
                print(f"      • {axis.name}: {axis.values}")
            print()
            
            return axes
        
        except Exception as e:
            print(f"   ❌ Decomposition failed: {e}")
            return {}


class FormulaQueryStrategy:
    """Step 2: Generate queries using simple formula strategies"""
    
    def __init__(self):
        from nodes.deduplicate import get_openai_client
        self.client, self.model = get_openai_client()
    
    async def generate_formula_queries(
        self,
        goal: str,
        limit_per_formula: int = 10
    ) -> List[str]:
        """
        Generate queries using three formulas:
        - Broad Net: Entity + Scope
        - Deep Dive: Entity + Attribute + Scope
        - Artifact Hunter: Entity + Scope + Source
        """
        print("🔍 Step 2: Formula Query Strategy")
        print(f"   Generating {limit_per_formula} queries per formula...\n")
        
        prompt = f"""
You are a query generation assistant.

Goal: {goal}

Generate 10 items for each list below. Keep each item short and search-friendly.

Return ONLY valid JSON with this structure:
{{
    "entities": ["...", "...", "...", "...", "...", "...", "...", "...", "...", "..."],
    "scopes": ["...", "...", "...", "...", "...", "...", "...", "...", "...", "..."],
    "attributes": ["...", "...", "...", "...", "...", "...", "...", "...", "...", "..."],
    "sources": ["...", "...", "...", "...", "...", "...", "...", "...", "...", "..."]
}}

Guidance:
- Entities: types/synonyms of the main subject
- Scopes: geographic/organizational scopes
- Attributes: specific data fields or qualifiers
- Sources: file types or source hints (e.g., PDF, CSV, directory)
"""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.choices[0].message.content
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in response")
            
            data = json.loads(json_match.group())
            entities = data.get("entities", [])[:limit_per_formula]
            scopes = data.get("scopes", [])[:limit_per_formula]
            attributes = data.get("attributes", [])[:limit_per_formula]
            sources = data.get("sources", [])[:limit_per_formula]
            
            queries = []
            
            # Broad Net: Entity + Scope
            for entity, scope in zip(entities, scopes):
                queries.append(f"{entity} in {scope}")
            
            # Deep Dive: Entity + Attribute + Scope
            for entity, attribute, scope in zip(entities, attributes, scopes):
                queries.append(f"{entity} {attribute} in {scope}")
            
            # Artifact Hunter: Entity + Scope + Source
            for entity, scope, source in zip(entities, scopes, sources):
                queries.append(f"{entity} in {scope} {source}")
            
            print(f"   ✓ Generated {len(queries)} queries\n")
            return queries
        
        except Exception as e:
            print(f"   ❌ Error: {e}\n")
            return []


class QueryExpansionMatrix:
    """Main orchestrator for query expansion"""
    
    def __init__(self):
        self.decomposer = ZeroShotDecomposer()
        self.formula_strategy = FormulaQueryStrategy()
    
    async def execute(
        self,
        goal: str,
        strategy: str = "formula",
        column_specs: List[Dict] = None
    ) -> QueryMatrix:
        """
        Execute the query expansion strategy
        
        Args:
            goal: Research goal (e.g., "Find hospitals in Ontario")
            strategy: "formula" (30 queries total: 10 per formula)
            column_specs: Optional list of column definitions to use for axis guidance
        
        Returns:
            QueryMatrix with decomposed axes and generated queries
        """
        # Step 1: Decompose (used only for reporting)
        axes = await self.decomposer.decompose(goal, column_specs=column_specs)
        if not axes:
            axes = {
                "entities": AxisOfVariance("Entities", [goal.split()[1]]),
                "geographic": AxisOfVariance("Geographic", [goal.split()[-1]])
            }
        
        # Step 2: Generate queries using formula strategy
        queries = await self.formula_strategy.generate_formula_queries(goal, limit_per_formula=10)
        
        # Calculate theoretical matrix size
        total_axes = 1
        for axis in axes.values():
            total_axes *= len(axis.values)
        
        return QueryMatrix(
            goal=goal,
            axes=list(axes.values()),
            corner_queries=queries,
            full_matrix_size=total_axes
        )


async def main():
    """Demo the query expansion matrix"""
    load_dotenv(find_dotenv())
    
    goal = "Find hospitals in Ontario"
    
    print("\n" + "="*70)
    print("QUERY EXPANSION MATRIX DEMO")
    print("="*70 + "\n")
    
    qem = QueryExpansionMatrix()
    result = await qem.execute(goal, strategy="formula")
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\nGoal: {result.goal}")
    print(f"Axes identified: {len(result.axes)}")
    print(f"Theoretical matrix size: {result.full_matrix_size} (full permutations)")
    print(f"\nGenerated queries ({len(result.corner_queries)}):")
    for i, q in enumerate(result.corner_queries, 1):
        print(f"  {i}. {q}")
    print()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
