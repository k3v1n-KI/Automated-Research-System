"""
Semantic Ranker - Ranks items based on semantic similarity to queries.
Uses embeddings to score relevance of extracted items to the original research prompt.
"""

import os
import json
from typing import List, Dict, Optional
from openai import OpenAI

# Initialize OpenAI client lazily
_openai_client = None
_embedding_model = None

def get_openai_client():
    """Get or create OpenAI client"""
    global _openai_client, _embedding_model
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")
        _openai_client = OpenAI(api_key=api_key)
        _embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    return _openai_client, _embedding_model


def get_embedding(text: str) -> List[float]:
    """Get embedding for text"""
    client, model = get_openai_client()
    
    # Truncate long text
    if len(text) > 8000:
        text = text[:8000]
    
    response = client.embeddings.create(
        model=model,
        input=text
    )
    return response.data[0].embedding


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = sum(a * a for a in vec1) ** 0.5
    mag2 = sum(b * b for b in vec2) ** 0.5
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    return dot_product / (mag1 * mag2)


def rank_items(
    items: List[Dict],
    query: str,
    score_key: str = "semantic_score",
    threshold: float = 0.5
) -> List[Dict]:
    """
    Rank items based on semantic similarity to query.
    
    Args:
        items: List of items to rank (dicts with 'name', 'text', or 'content' fields)
        query: Query/prompt to rank against
        score_key: Key name for the score in returned items
        threshold: Minimum score to keep (0-1, where 1 is perfect match)
    
    Returns:
        List of items sorted by semantic relevance, filtered by threshold
    """
    
    if not items:
        return []
    
    if not query:
        return items
    
    try:
        # Get query embedding
        query_embedding = get_embedding(query)
        
        # Score each item
        scored_items = []
        for item in items:
            # Extract text from item (try multiple common fields)
            text = ""
            if isinstance(item, dict):
                text = item.get('text') or item.get('content') or item.get('name') or str(item)
            else:
                text = str(item)
            
            if not text:
                continue
            
            # Get item embedding and score
            item_embedding = get_embedding(text)
            score = cosine_similarity(query_embedding, item_embedding)
            
            # Add score to item
            scored_item = item.copy() if isinstance(item, dict) else {'value': item}
            scored_item[score_key] = score
            scored_items.append(scored_item)
        
        # Filter by threshold and sort
        filtered = [item for item in scored_items if item[score_key] >= threshold]
        sorted_items = sorted(filtered, key=lambda x: x[score_key], reverse=True)
        
        return sorted_items
    
    except Exception as e:
        print(f"❌ Semantic ranking failed: {e}")
        # Fallback: return items with 0 score
        for item in items:
            if isinstance(item, dict):
                item[score_key] = 0
        return items


def rank_search_results(
    results: List[Dict],
    query: str,
    min_score: float = 0.4
) -> List[Dict]:
    """
    Rank search results by relevance to query.
    
    Args:
        results: List of search results (from search node)
        query: Original search query
        min_score: Minimum score to keep
    
    Returns:
        Ranked list of search results
    """
    
    if not results:
        return []
    
    try:
        query_embedding = get_embedding(query)
        
        scored_results = []
        for result in results:
            # Combine title and snippet for ranking
            text = f"{result.get('title', '')} {result.get('snippet', '')}"
            if not text.strip():
                continue
            
            item_embedding = get_embedding(text)
            score = cosine_similarity(query_embedding, item_embedding)
            
            result_copy = result.copy()
            result_copy['relevance_score'] = score
            scored_results.append(result_copy)
        
        # Sort by relevance
        ranked = sorted(scored_results, key=lambda x: x['relevance_score'], reverse=True)
        
        # Filter by minimum score
        return [r for r in ranked if r['relevance_score'] >= min_score]
    
    except Exception as e:
        print(f"❌ Search result ranking failed: {e}")
        return results


def rerank_extracted_items(
    items: List[Dict],
    queries: List[str],
    min_score: float = 0.5
) -> List[Dict]:
    """
    Rerank extracted items using multiple queries.
    Assigns a composite score based on similarity to all queries.
    
    Args:
        items: Extracted data items
        queries: List of search queries used
        min_score: Minimum composite score to keep
    
    Returns:
        Reranked items sorted by composite relevance
    """
    
    if not items or not queries:
        return items
    
    try:
        # Get embeddings for all queries
        query_embeddings = [get_embedding(q) for q in queries]
        
        scored_items = []
        for item in items:
            # Combine all fields into searchable text
            text = " ".join(str(v) for v in item.values() if v)
            if not text:
                continue
            
            item_embedding = get_embedding(text)
            
            # Calculate average similarity across all queries
            scores = [cosine_similarity(item_embedding, q_emb) for q_emb in query_embeddings]
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            
            item_copy = item.copy()
            item_copy['composite_score'] = avg_score
            item_copy['max_query_score'] = max_score
            item_copy['query_scores'] = scores
            
            scored_items.append(item_copy)
        
        # Filter and sort
        filtered = [item for item in scored_items if item['composite_score'] >= min_score]
        ranked = sorted(filtered, key=lambda x: x['composite_score'], reverse=True)
        
        return ranked
    
    except Exception as e:
        print(f"❌ Item reranking failed: {e}")
        return items


if __name__ == "__main__":
    # Test
    print("Semantic Ranker loaded. Use rank_items(), rank_search_results(), or rerank_extracted_items().")
