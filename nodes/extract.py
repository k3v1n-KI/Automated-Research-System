"""
Node 5: Extract
Extracts structured data from scraped content using LLM.
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


class ExtractNode(BaseNode):
    """
    Extracts structured data from scraped content using LLM.
    
    Input State Keys:
        - scraped_content: List of documents to extract from
        - initial_prompt: Task description for context
    
    Output State Keys:
        - extracted_items: List of extracted records with source_url
    """
    
    def __init__(self, char_limit: int = 6000):
        super().__init__()
        self.char_limit = char_limit
    
    async def execute(self, state: "ResearchState", progress: "ProgressTracker") -> "ResearchState":
        """Extract structured data from scraped content"""
        
        scraped_items = state['scraped_content']
        initial_prompt = state['initial_prompt']
        columns = state.get('columns', [])
        openai_client, openai_model = get_openai_client()
        
        # Get column names for constraint
        column_names = [col['name'] for col in columns if isinstance(col, dict)] if columns else []
        column_constraint = ""
        if column_names:
            column_constraint = f"\n\nIMPORTANT: Extract ONLY these columns in your JSON output: {', '.join(column_names)}. Do not include any other fields."
        
        progress.update(
            "🎯 Extraction Starting",
            f"Extracting data from {len(scraped_items)} documents..."
        )
        
        extracted_items = []
        
        for idx, item in enumerate(scraped_items):
            if (idx + 1) % 5 == 0:
                progress.update(
                    "🎯 Extracting",
                    f"Extracted from {idx + 1}/{len(scraped_items)} documents"
                )
            
            url = item['url']
            text = item.get('text', '')
            
            if not text.strip():
                continue
            
            try:
                total_length = len(text)
                total_chunks = max(1, (total_length + self.char_limit - 1) // self.char_limit)

                for chunk_index, start in enumerate(range(0, total_length, self.char_limit), 1):
                    chunk = text[start:start + self.char_limit]
                    if not chunk.strip():
                        continue

                    # Use LLM to extract structured data from each chunk
                    response = await openai_client.chat.completions.create(
                        model=openai_model,
                        messages=[
                            {
                                "role": "system",
                                "content": f"""You are a High-Precision Data Extraction Engine. 
Your goal is to filter and extract data that **strictly aligns** with the Following constraints:

Task: {initial_prompt}
Data extraction constraints: {column_constraint}

Extract only valid, complete entries. Return a JSON array of objects."""
                            },
                            {
                                "role": "user",
                                "content": (
                                    f"Extract data from this content chunk ({chunk_index}/{total_chunks}):\n\n"
                                    f"{chunk}\n\nURL: {url}"
                                )
                            }
                        ],
                        max_completion_tokens=6000
                    )

                    content = response.choices[0].message.content

                    # Parse extracted data
                    json_match = re.search(r'\[.*\]', content, re.DOTALL)
                    if json_match:
                        try:
                            data = json.loads(json_match.group())
                            if isinstance(data, list):
                                for record in data:
                                    record['source_url'] = url
                                    extracted_items.append(record)
                        except json.JSONDecodeError:
                            pass

            except Exception as e:
                print(f"⚠️  Extraction error for {url}: {e}")
        
        state['extracted_items'] = extracted_items
        
        progress.update(
            "🎯 Extraction Complete",
            f"Extracted {len(extracted_items)} records from {len(scraped_items)} documents",
            {"extracted_count": len(extracted_items), "source_documents": len(scraped_items)}
        )
        
        return state
