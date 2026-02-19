"""
Node 1 (Alt): Query Expansion Matrix
Generates 15 queries using three formulas (5 each):
- Broad Net: Entity + Scope
- Deep Dive: Entity + Attribute + Scope
- Artifact Hunter: Entity + Scope + Source
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


class QueryExpansionMatrixNode(BaseNode):
    """
    Generates 15 queries using formula strategy.

    Input State Keys:
        - initial_prompt: User's dataset description
        - column_specs: Optional list of column definitions
        - previous_queries: Optional list of previous queries
        - tweak_instructions: Optional user instructions

    Output State Keys:
        - queries: List of 15 search queries
    """

    async def execute(self, state: "ResearchState", progress: "ProgressTracker") -> "ResearchState":
        progress.update(
            "Query Generation",
            "Generating formula-based queries (Broad Net / Deep Dive / Artifact Hunter)..."
        )

        prompt = state["initial_prompt"]
        column_specs = state.get("column_specs", [])
        previous_queries = state.get("previous_queries", [])
        tweak_instructions = state.get("tweak_instructions", "")
        openai_client, openai_model = get_openai_client()

        columns_text = ""
        if column_specs:
            if isinstance(column_specs, list) and column_specs and isinstance(column_specs[0], dict):
                columns_text = "\n".join(
                    f"- {col.get('name', 'field')}: {col.get('description', 'N/A')}"
                    for col in column_specs
                )
            elif isinstance(column_specs, list):
                columns_text = "\n".join(f"- {str(col)}" for col in column_specs)

        previous_context = ""
        if previous_queries:
            previous_context = (
                "Avoid repeating these previous queries; generate different angles:\n"
                f"{json.dumps(previous_queries, indent=2)}\n"
            )

        if tweak_instructions:
            previous_context += f"User instructions: {tweak_instructions}\n"

        schema_prompt = f"""
You are a query generation assistant.

Goal: {prompt}

Generate exactly 5 items for each list below. Keep each item short and search-friendly.
{('Desired fields:\n' + columns_text) if columns_text else ''}

Return ONLY valid JSON with this structure:
{{
  "entities": ["...", "...", "...", "...", "..."],
  "scopes": ["...", "...", "...", "...", "..."],
  "attributes": ["...", "...", "...", "...", "..."],
  "sources": ["...", "...", "...", "...", "..."]
}}

Guidance:
- Entities: types/synonyms of the main subject
- Scopes: geographic/organizational scopes
- Attributes: specific data fields or qualifiers
- Sources: file types or source hints (e.g., PDF, CSV, directory)

{previous_context}
"""

        try:
            response = await openai_client.chat.completions.create(
                model=openai_model,
                messages=[{"role": "user", "content": schema_prompt}],
            )
            content = response.choices[0].message.content
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in LLM response")

            data = json.loads(json_match.group())
            entities = self._pad_list(data.get("entities", []), "entity", prompt)
            scopes = self._pad_list(data.get("scopes", []), "scope", prompt)
            attributes = self._pad_list(data.get("attributes", []), "attribute", prompt, column_specs)
            sources = self._pad_list(data.get("sources", []), "source", prompt)

            queries = []
            for entity, scope in zip(entities, scopes):
                queries.append(f"{entity} in {scope}")
            for entity, attribute, scope in zip(entities, attributes, scopes):
                queries.append(f"{entity} {attribute} in {scope}")
            for entity, scope, source in zip(entities, scopes, sources):
                queries.append(f"{entity} in {scope} {source}")

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
                    f"Generated {len(queries)} formula-based queries",
                    {"queries": queries}
                )

            state["queries"] = queries
            return state

        except Exception as e:
            state["error"] = f"Query generation failed: {str(e)}"
            progress.update("Query Generation Failed", str(e))
            raise

    def _pad_list(
        self,
        items: List[str],
        kind: str,
        prompt: str,
        column_specs: List[Dict] = None
    ) -> List[str]:
        cleaned = [str(item).strip() for item in items if str(item).strip()]
        while len(cleaned) < 5:
            cleaned.append(self._fallback_item(kind, len(cleaned), prompt, column_specs))
        return cleaned[:5]

    def _fallback_item(self, kind: str, index: int, prompt: str, column_specs: List[Dict] = None) -> str:
        if kind == "entity":
            base = "hospital" if "hospital" in prompt.lower() else "entity"
            return f"{base} type {index + 1}"
        if kind == "scope":
            base = "Ontario" if "ontario" in prompt.lower() else "region"
            return f"{base} {index + 1}"
        if kind == "attribute" and column_specs:
            for col in column_specs:
                if isinstance(col, dict) and col.get("name"):
                    return col["name"]
        if kind == "attribute":
            return f"attribute {index + 1}"
        if kind == "source":
            defaults = ["filetype:pdf", "filetype:csv", "dataset", "directory", "open data"]
            return defaults[index % len(defaults)]
        return f"{kind} {index + 1}"
