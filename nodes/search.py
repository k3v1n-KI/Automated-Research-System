"""
Node 2: Search
Searches for URLs using SEARXNG, Google CSE, and Tavily fallbacks.
"""

import os
from pathlib import Path
import requests
from dotenv import load_dotenv
from typing import TYPE_CHECKING

from nodes.base import BaseNode

if TYPE_CHECKING:
    from algorithm import ResearchState, ProgressTracker


_dotenv_path = Path(__file__).resolve().parents[1] / "searxng" / ".env"
if _dotenv_path.exists():
    load_dotenv(_dotenv_path)


class SearchNode(BaseNode):
    """
    Searches for URLs using SEARXNG as primary engine and Google API as fallback.
    
    Input State Keys:
        - queries: List of search query strings or query metadata dicts
    
    Output State Keys:
        - search_results: List of {"url", "title", "snippet", "source_query", "query_technique"}
    """
    
    async def execute(self, state: "ResearchState", progress: "ProgressTracker") -> "ResearchState":
        """Search for URLs across configured search providers."""
        
        raw_queries = state['queries']
        searxng_url = os.getenv("SEARXNG_URL", "http://localhost:8080").rstrip("/")
        google_key = os.getenv("GOOGLE_API_KEY")
        google_cx = os.getenv("GOOGLE_CX") or os.getenv("GOOGLE_SEARCH_ENGINE_ID")
        tavily_key = os.getenv("TAVILY_API_KEY")
        prefer_tavily = bool(state.get("pathways_prefer_tavily"))

        limits = state.get("pathways_limits") or {}
        max_queries = int(limits.get("max_queries", len(raw_queries)))
        max_urls = int(limits.get("max_urls", 0))
        query_rows = []
        for q in raw_queries:
            if isinstance(q, dict):
                query_text = str(q.get("query", "")).strip()
                if not query_text:
                    continue
                query_rows.append({
                    "query": query_text,
                    "query_technique": str(q.get("query_technique") or "unspecified"),
                })
            else:
                query_text = str(q).strip()
                if not query_text:
                    continue
                query_rows.append({
                    "query": query_text,
                    "query_technique": "simple_generation",
                })
        query_rows = query_rows[:max_queries]
        
        all_results = []
        
        progress.update(
            "🔎 Search Starting",
            f"Searching {len(query_rows)} queries across search engines..."
        )
        
        for idx, query_row in enumerate(query_rows):
            query = query_row["query"]
            technique = query_row["query_technique"]
            progress.update(
                "🔎 Searching",
                f"Query {idx+1}/{len(query_rows)}: {query}"
            )

            if prefer_tavily and tavily_key:
                try:
                    response = requests.post(
                        "https://api.tavily.com/search",
                        headers={"Content-Type": "application/json"},
                        json={"api_key": tavily_key, "query": query, "search_depth": "basic", "max_results": min(10, max_urls or 10), "include_answer": False, "include_raw_content": False},
                        timeout=20,
                    )
                    if response.status_code == 200:
                        tavily_items = response.json().get("results", [])
                        for item in tavily_items:
                            if item.get("url"):
                                all_results.append({"url": item["url"], "title": item.get("title", ""), "snippet": item.get("content", ""), "source_query": query, "query_technique": "tavily"})
                        if tavily_items:
                            continue
                except Exception as error:
                    print(f"❌ Tavily preferred search error for '{query}': {error}")
            
            # Try SEARXNG first
            try:
                response = requests.get(
                    f"{searxng_url}/search",
                    params={
                        "q": query,
                        "format": "json",
                        "language": "en",
                        "safesearch": 1,
                        "categories": "general"
                    },
                    timeout=12
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])[:25]  # Get top 25 results to allow for some filtering
                    for r in results:
                        if r.get("url"):
                            all_results.append({
                                "url": r.get("url"),
                                "title": r.get("title", ""),
                                "snippet": r.get("content", ""),
                                "source_query": query,
                                "query_technique": technique,
                            })
            except Exception as e:
                print(f"❌ SEARXNG error for '{query}': {e}")
            
            # Fallback to Google
            if len(all_results) < (idx + 1) * 25 and google_key and google_cx:
                try:
                    response = requests.get(
                        "https://www.googleapis.com/customsearch/v1",
                        params={
                            "key": google_key,
                            "cx": google_cx,
                            "q": query,
                            "num": 10
                        },
                        timeout=12
                    )
                    if response.status_code == 200:
                        data = response.json()
                        items = data.get("items", [])[:10]
                        for item in items:
                            if item.get("link"):
                                all_results.append({
                                    "url": item.get("link"),
                                    "title": item.get("title", ""),
                                    "snippet": item.get("snippet", ""),
                                    "source_query": query,
                                    "query_technique": technique,
                                })
                except Exception as e:
                    print(f"❌ Google API error for '{query}': {e}")

            # Tavily fallback for AI-oriented web research.
            if len(all_results) < (idx + 1) * 25 and tavily_key:
                try:
                    response = requests.post(
                        "https://api.tavily.com/search",
                        headers={"Content-Type": "application/json"},
                        json={
                            "api_key": tavily_key,
                            "query": query,
                            "search_depth": "basic",
                            "max_results": min(10, max_urls or 10),
                            "include_answer": False,
                            "include_raw_content": False,
                        },
                        timeout=20,
                    )
                    if response.status_code == 200:
                        for item in response.json().get("results", []):
                            if item.get("url"):
                                all_results.append({
                                    "url": item["url"],
                                    "title": item.get("title", ""),
                                    "snippet": item.get("content", ""),
                                    "source_query": query,
                                    "query_technique": "tavily",
                                })
                    else:
                        print(f"❌ Tavily error for '{query}': HTTP {response.status_code}")
                except Exception as error:
                    print(f"❌ Tavily request error for '{query}': {error}")
        
        # Deduplicate by URL
        seen = set()
        deduped = []
        for r in all_results:
            url = r['url']
            if url not in seen:
                seen.add(url)
                deduped.append(r)
                if max_urls and len(deduped) >= max_urls:
                    break
        
            if max_urls and len(deduped) >= max_urls:
                break

        fallback_queries = state.get("pathways_fallback_queries") or []
        if not deduped and fallback_queries:
            for fallback_query in fallback_queries[:2]:
                try:
                    response = requests.get(
                        f"{searxng_url}/search",
                        params={"q": fallback_query, "format": "json", "language": "en", "safesearch": 1, "categories": "general"},
                        headers={"User-Agent": "Pathways-ARS/1.0"},
                        timeout=12,
                    )
                    for result in response.json().get("results", []):
                        url = result.get("url")
                        if url and url not in seen:
                            seen.add(url)
                            deduped.append({
                                "url": url,
                                "title": result.get("title", ""),
                                "snippet": result.get("content", ""),
                                "source_query": fallback_query,
                                "query_technique": "pathways_fallback",
                            })
                            if max_urls and len(deduped) >= max_urls:
                                break
                except Exception as error:
                    print(f"❌ SearXNG fallback error for '{fallback_query}': {error}")
                if deduped:
                    break
        
        state['search_results'] = deduped
        
        techniques = sorted({r.get("query_technique", "unspecified") for r in deduped})
        progress.update(
            "🔎 Search Complete",
            f"Found {len(deduped)} unique URLs across {len(query_rows)} queries",
            {"total_urls": len(deduped), "queries": len(query_rows), "techniques": techniques}
        )
        
        return state
