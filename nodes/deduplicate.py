"""
Node 6: Deduplicate
Removes duplicates from extracted dataset using LLM.
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
        _openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return _openai_client, _openai_model


class DeduplicateNode(BaseNode):
    """
    Removes duplicate records from extracted dataset using LLM.
    
    Input State Keys:
        - extracted_items: List of extracted records
    
    Output State Keys:
        - final_dataset: Deduplicated list of records
    """
    
    async def execute(self, state: "ResearchState", progress: "ProgressTracker") -> "ResearchState":
        """Remove duplicates from extracted dataset"""
        
        items = state['extracted_items']
        openai_client, openai_model = get_openai_client()
        
        progress.update(
            "🗑️  Deduplication Starting",
            f"Removing duplicates from {len(items)} records..."
        )
        
        if not items:
            state['final_dataset'] = []
            progress.update("🗑️  Deduplication Complete", "No items to deduplicate")
            return state
        
        try:
            # Use LLM to identify and remove duplicates
            response = await openai_client.chat.completions.create(
                model=openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a data deduplication expert. Identify and remove duplicate entries.
Two entries are duplicates if they represent the same entity (same name, location, etc.).
Return a JSON array of deduplicated records. Keep all unique entries."""
                    },
                    {
                        "role": "user",
                        "content": f"Remove duplicates from this dataset:\n\n{json.dumps(items)}"
                    }
                ],
            )
            
            content = response.choices[0].message.content
            
            # Parse deduplicated data
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                try:
                    deduped = json.loads(json_match.group())
                    state['final_dataset'] = deduped if isinstance(deduped, list) else items
                except json.JSONDecodeError:
                    state['final_dataset'] = items
            else:
                state['final_dataset'] = items
        
        except Exception as e:
            print(f"⚠️  Deduplication error: {e}")
            state['final_dataset'] = items
        
        final_count = len(state['final_dataset'])
        
        progress.update(
            "🗑️  Deduplication Complete",
            f"Final dataset: {final_count} unique records (removed {len(items) - final_count} duplicates)",
            {
                "original_count": len(items),
                "final_count": final_count,
                "duplicates_removed": len(items) - final_count
            }
        )
        
        return state
