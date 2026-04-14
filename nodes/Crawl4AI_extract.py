"""
Node 5 (Temp): Crawl4AI Extract
Extracts structured data from markdown content using LLM.
Chunking strategy respects header hierarchy.
"""

import json
import re
import os
from typing import TYPE_CHECKING, List, Dict

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


class Crawl4AIExtractNode(BaseNode):
    """
    Extracts structured data from markdown using LLM.

    Input State Keys:
        - scraped_content: List of {url, text}
        - initial_prompt: Task description for context
        - columns or column_specs: Optional column definitions

    Output State Keys:
        - extracted_items: List of extracted records with source_url
    """

    def __init__(self, char_limit: int = 6000):
        super().__init__()
        self.char_limit = char_limit

    async def execute(self, state: "ResearchState", progress: "ProgressTracker") -> "ResearchState":
        scraped_items = state.get("scraped_content", [])
        initial_prompt = state.get("initial_prompt", "")
        columns = state.get("columns") or state.get("column_specs") or []
        openai_client, openai_model = get_openai_client()
        # GEMINI - Temporarily commented out
        # gemini_model = get_gemini_model()

        column_names = [col["name"] for col in columns if isinstance(col, dict) and col.get("name")] if columns else []
        column_constraint = ""
        if column_names:
            column_constraint = (
                f"\n\nIMPORTANT: Extract ONLY these columns in your JSON output: {', '.join(column_names)}."
                " Do not include any other fields."
            )

        progress.update(
            "🎯 Extraction Starting",
            f"Extracting data from {len(scraped_items)} documents..."
        )

        extracted_items: List[Dict] = []

        for idx, item in enumerate(scraped_items):
            if (idx + 1) % 5 == 0:
                progress.update(
                    "🎯 Extracting",
                    f"Extracted from {idx + 1}/{len(scraped_items)} documents"
                )

            url = item.get("url", "")
            markdown_text = item.get("text", "")
            source_query = str(item.get("source_query", "") or "")
            query_technique = str(item.get("query_technique", "unspecified") or "unspecified")

            if not markdown_text.strip():
                continue

            chunks = self._chunk_markdown(markdown_text)

            for chunk_index, chunk in enumerate(chunks, 1):
                if not chunk.strip():
                    continue

                try:
                    response = await openai_client.chat.completions.create(
                        model=openai_model,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a High-Precision Data Extraction Engine. "
                                    "Your goal is to filter and extract data that **strictly aligns** with the Following constraints:\n\n"
                                    f"Task: {initial_prompt}\n"
                                    f"Data extraction constraints: {column_constraint}\n\n"
                                    "Extract only valid, complete entries. Return a JSON array of objects."
                                ),
                            },
                            {
                                "role": "user",
                                "content": (
                                    f"Extract data from this content chunk ({chunk_index}/{len(chunks)}):\n\n"
                                    f"{chunk}\n\nURL: {url}"
                                ),
                            },
                        ],
                        max_completion_tokens=6000,
                    )
                    content = response.choices[0].message.content

                    # GEMINI - Temporarily commented out
                    # prompt = (
                    #     "You are a High-Precision Data Extraction Engine. "
                    #     "Your goal is to filter and extract data that **strictly aligns** with the Following constraints:\n\n"
                    #     f"Task: {initial_prompt}\n"
                    #     f"Data extraction constraints: {column_constraint}\n\n"
                    #     "Extract only valid, complete entries. Return a JSON array of objects.\n\n"
                    #     f"Extract data from this content chunk ({chunk_index}/{len(chunks)}):\n\n"
                    #     f"{chunk}\n\nURL: {url}"
                    # )
                    # 
                    # content = await _call_gemini_with_retry(gemini_model, prompt)
                    json_match = re.search(r"\[.*\]", content, re.DOTALL)
                    if json_match:
                        data = json.loads(json_match.group())
                        if isinstance(data, list):
                            for record in data:
                                record["source_url"] = url
                                if source_query:
                                    record["source_query"] = source_query
                                record["query_technique"] = query_technique
                                extracted_items.append(record)

                except Exception as e:
                    print(f"⚠️  Extraction error for {url}: {e}")

        state["extracted_items"] = extracted_items

        progress.update(
            "🎯 Extraction Complete",
            f"Extracted {len(extracted_items)} records from {len(scraped_items)} documents",
            {"extracted_count": len(extracted_items), "source_documents": len(scraped_items)}
        )

        return state

    def _chunk_markdown(self, text: str) -> List[str]:
        blocks = self._split_by_headers(text)
        if not blocks:
            return self._fixed_chunks(text)

        chunks: List[str] = []
        bucket = []
        bucket_len = 0

        for block in blocks:
            block_len = len(block)
            if block_len > self.char_limit:
                if bucket:
                    chunks.append("\n".join(bucket))
                    bucket = []
                    bucket_len = 0
                chunks.extend(self._fixed_chunks(block))
                continue

            if bucket_len + block_len > self.char_limit and bucket:
                chunks.append("\n".join(bucket))
                bucket = [block]
                bucket_len = block_len
            else:
                bucket.append(block)
                bucket_len += block_len

        if bucket:
            chunks.append("\n".join(bucket))

        return chunks

    def _split_by_headers(self, text: str) -> List[str]:
        lines = text.splitlines()
        blocks: List[str] = []
        current: List[str] = []
        header_pattern = re.compile(r"^#{1,3}\s+.+")

        for line in lines:
            if header_pattern.match(line):
                if current:
                    blocks.append("\n".join(current).strip())
                    current = []
            current.append(line)

        if current:
            blocks.append("\n".join(current).strip())

        return [b for b in blocks if b]

    def _fixed_chunks(self, text: str) -> List[str]:
        return [text[i:i + self.char_limit] for i in range(0, len(text), self.char_limit)]
