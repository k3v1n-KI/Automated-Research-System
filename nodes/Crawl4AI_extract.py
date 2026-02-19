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
from nodes.base import BaseNode

if TYPE_CHECKING:
    from algorithm import ResearchState, ProgressTracker


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
                    json_match = re.search(r"\[.*\]", content, re.DOTALL)
                    if json_match:
                        data = json.loads(json_match.group())
                        if isinstance(data, list):
                            for record in data:
                                record["source_url"] = url
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
