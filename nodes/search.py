"""
Node 2: Search
Searches for URLs using SEARXNG and Google API fallback.
"""

import os
import requests
from typing import TYPE_CHECKING

from nodes.base import BaseNode

if TYPE_CHECKING:
    from algorithm import ResearchState, ProgressTracker


class SearchNode(BaseNode):
    """
    Searches for URLs using SEARXNG as primary engine and Google API as fallback.
    
    Input State Keys:
        - queries: List of search query strings or query metadata dicts
    
    Output State Keys:
        - search_results: List of {"url", "title", "snippet", "source_query", "query_technique"}
    """
    
    async def execute(self, state: "ResearchState", progress: "ProgressTracker") -> "ResearchState":
        """Search for URLs across SEARXNG and Google API"""
        
        raw_queries = state['queries']
        searxng_url = os.getenv("SEARXNG_URL", "http://localhost:8888").rstrip("/")
        google_key = os.getenv("GOOGLE_API_KEY")
        google_cx = os.getenv("GOOGLE_CX")

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
        
        # Deduplicate by URL
        seen = set()
        deduped = []
        for r in all_results:
            url = r['url']
            if url not in seen:
                seen.add(url)
                deduped.append(r)
        
        state['search_results'] = deduped
        
        techniques = sorted({r.get("query_technique", "unspecified") for r in deduped})
        progress.update(
            "🔎 Search Complete",
            f"Found {len(deduped)} unique URLs across {len(query_rows)} queries",
            {"total_urls": len(deduped), "queries": len(query_rows), "techniques": techniques}
        )
        
        return state
