"""
Node 1: Query Generation
Generates 4 diverse search queries from initial prompt using LLM.
"""

import json
import re
import os
from typing import TYPE_CHECKING

from openai import AsyncOpenAI
# GEMINI - Temporarily commented out
# import google.generativeai as genai
from nodes.base import BaseNode

if TYPE_CHECKING:
    from algorithm import ResearchState, ProgressTracker

_openai_client = None
_openai_model = None

# GEMINI - Temporarily commented out
# _gemini_model = None

# GEMINI - Temporarily commented out
# async def _call_gemini_with_retry(model, prompt: str, max_retries: int = 3) -> str:
#     """Call Gemini API with automatic retry on 429 rate limit errors"""
#     for attempt in range(max_retries):
#         try:
#             response = await model.generate_content_async(prompt)
#             return response.text
#         except Exception as e:
#             error_code = getattr(e, 'status_code', None)
#             if error_code == 429 or "429" in str(e) or "quota" in str(e).lower():
#                 if attempt < max_retries - 1:
#                     retry_delay = 15
#                     try:
#                         if hasattr(e, 'retry_delay') and hasattr(e.retry_delay, 'seconds'):
#                             retry_delay = e.retry_delay.seconds + 2
#                         elif "Please retry in" in str(e):
#                             import re as re_module
#                             match = re_module.search(r'Please retry in ([0-9.]+)s', str(e))
#                             if match:
#                                 retry_delay = float(match.group(1)) + 2
#                     except:
#                         retry_delay = (2 ** attempt) + 15
#                     
#                     print(f"⏳ Rate limit hit (429). Retrying in {retry_delay:.1f}s... (Attempt {attempt + 1}/{max_retries})")
#                     await asyncio.sleep(retry_delay)
#                     continue
#                 else:
#                     print(f"✗ Rate limit exceeded. Max retries ({max_retries}) reached.")
#                     raise
#             raise

def get_openai_client():
    global _openai_client, _openai_model
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")
        _openai_client = AsyncOpenAI(api_key=api_key)
        _openai_model = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    return _openai_client, _openai_model

# GEMINI - Temporarily commented out
# def get_gemini_model():
#     global _gemini_model
#     if _gemini_model is None:
#         api_key = os.getenv("GEMINI_API_KEY")
#         if not api_key:
#             raise ValueError("GEMINI_API_KEY not set in environment")
#         genai.configure(api_key=api_key)
#         model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
#         _gemini_model = genai.GenerativeModel(model_name)
#     return _gemini_model


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
        # GEMINI - Temporarily commented out
        # gemini_model = get_gemini_model()
        
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
            content = response.choices[0].message.content

            # GEMINI - Temporarily commented out
            # full_prompt = f"""You are a research assistant helping to construct datasets.
            # Your task is to generate search queries that will find all relevant results.
            # The queries should:
            # 1. Use different phrasings and keywords
            # 2. Cover different aspects and locations
            # 3. Be specific enough to find relevant results
            # 4. Be broad enough to catch variations
            # 5. If previous queries exist, generate NEW queries that target different angles or untapped areas
            # 
            # Return ONLY a JSON object with a "queries" array. No other text.
            # 
            # Generate search queries for this dataset task:
            # 
            # {prompt}
            # {previous_context}
            # 
            # Return as: {{"queries": ["query1", "query2", "query3", "query4"]}}"""
            # 
            # content = await _call_gemini_with_retry(gemini_model, full_prompt)
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
