"""
Node 1: Query Generation
Generates 4 diverse search queries from initial prompt using LLM.
"""

import json
import re
import os
from typing import TYPE_CHECKING

from openai import AsyncOpenAI
from nodes.base import BaseNode

if TYPE_CHECKING:
    from algorithm import ResearchState, ProgressTracker

# Initialize OpenAI client lazily
_openai_client = None
_openai_model = None

def get_openai_client():
    global _openai_client, _openai_model
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")
        _openai_client = AsyncOpenAI(api_key=api_key)
        _openai_model = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    return _openai_client, _openai_model


class QueryGenerationNode(BaseNode):
    """
    Generates 4 diverse search queries from the initial dataset prompt.
    
    Input State Keys:
        - initial_prompt: User's dataset description
    
    Output State Keys:
        - queries: List of 4 search queries
    """
    
    async def execute(self, state: "ResearchState", progress: "ProgressTracker") -> "ResearchState":
        """Generate diverse queries from initial prompt, potentially building on previous queries"""
        
        progress.update(
            "Query Generation",
            "Analyzing prompt and generating search queries..."
        )
        
        prompt = state['initial_prompt']
        previous_queries = state.get('previous_queries', [])
        tweak_instructions = state.get('tweak_instructions', '')
        openai_client, openai_model = get_openai_client()
        
        # Build context about previous queries
        previous_context = ""
        if previous_queries:
            previous_context = f"""
Previous queries from earlier research:
{json.dumps(previous_queries, indent=2)}

User instructions for improvement: {tweak_instructions if tweak_instructions else 'None - generate different queries to find new data'}
"""
        
        try:
            response = await openai_client.chat.completions.create(
                model=openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a research assistant helping to construct datasets.
Your task is to generate search queries that will find all relevant results.
The queries should:
1. Use different phrasings and keywords
2. Cover different aspects and locations
3. Be specific enough to find relevant results
4. Be broad enough to catch variations
5. If previous queries exist, generate NEW queries that target different angles or untapped areas

Return ONLY a JSON object with a "queries" array. No other text."""
                    },
                    {
                        "role": "user",
                        "content": f"""Generate search queries for this dataset task:

{prompt}
{previous_context}

Return as: {{"queries": ["query1", "query2", "query3", "query4"]}}"""
                    }
                ],
            )
            
            # Parse response
            content = response.choices[0].message.content
            json_match = re.search(r'\{.*"queries".*\}', content, re.DOTALL)
            
            if json_match:
                data = json.loads(json_match.group())
                queries = data.get("queries", [])[:4]
            else:
                queries = []
            
            if not queries:
                raise ValueError("Failed to parse queries from LLM response")
            
            # If we have previous queries, append new ones to build dataset
            if previous_queries:
                queries = previous_queries + queries
                progress.update(
                    "Query Generation Complete",
                    f"Generated {len(queries) - len(previous_queries)} new queries (plus {len(previous_queries)} previous)",
                    {"new_queries": queries[len(previous_queries):], "total_queries": queries}
                )
            else:
                progress.update(
                    "Query Generation Complete",
                    f"Generated {len(queries)} diverse queries",
                    {"queries": queries}
                )
            
            state['queries'] = queries
            return state
            
        except Exception as e:
            state['error'] = f"Query generation failed: {str(e)}"
            progress.update("Query Generation Failed", str(e))
            raise
