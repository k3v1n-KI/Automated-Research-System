"""
Node 5: Extract
Extracts structured data from scraped content using LLM.
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
        # GEMINI - Temporarily commented out
        # gemini_model = get_gemini_model()
        
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

                    # GEMINI - Temporarily commented out
                    # prompt = f"""You are a High-Precision Data Extraction Engine. 
                    # Your goal is to filter and extract data that **strictly aligns** with the Following constraints:
                    # 
                    # Task: {initial_prompt}
                    # Data extraction constraints: {column_constraint}
                    # 
                    # Extract only valid, complete entries. Return a JSON array of objects.
                    # 
                    # Extract data from this content chunk ({chunk_index}/{total_chunks}):
                    # 
                    # {chunk}
                    # 
                    # URL: {url}"""
                    # 
                    # content = await _call_gemini_with_retry(gemini_model, prompt)

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
