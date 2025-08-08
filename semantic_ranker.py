from sentence_transformers import SentenceTransformer
import numpy as np

def load_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Load and return a SentenceTransformer model.
    """
    return SentenceTransformer(model_name)


def score_result_similarity(
    result: dict,
    query: str,
    model: SentenceTransformer
) -> dict:
    """
    Compute cosine similarity between the query and a search result
    (using title + snippet only).

    Args:
        result: {"title": str, "snippet": str, "url": str}
        query: original search query string
        model: loaded SentenceTransformer
    
    Returns:
        {"similarity_score": float}
    """
    # combine title + snippet
    text = (result.get("title", "") + " " + result.get("snippet", "")).strip()
    if not text:
        return {"similarity_score": 0.0}

    # embed both
    query_emb  = model.encode(query, convert_to_numpy=True)
    result_emb = model.encode(text,  convert_to_numpy=True)

    # cosine similarity
    sim = np.dot(query_emb, result_emb) / (
        np.linalg.norm(query_emb) * np.linalg.norm(result_emb)
    )
    return {"similarity_score": float(sim)}


def rank_results_by_similarity(
    results: list[dict],
    query: str,
    model: SentenceTransformer,
    top_k: int = None,
    threshold: float = None
) -> list[dict]:
    """
    Score and sort a list of search results by semantic similarity to the query.

    Args:
        results: list of {"title", "snippet", "url"} dicts
        query: original search query string
        model: loaded SentenceTransformer
        top_k: if set, only return the top_k highest-scoring results
        threshold: if set, only return results with similarity_score >= threshold

    Returns:
        List of results dicts extended with "similarity_score", sorted descending.
    """
    scored = []
    for res in results:
        score = score_result_similarity(res, query, model)["similarity_score"]
        entry = dict(res)
        entry["similarity_score"] = score
        scored.append(entry)

    # apply threshold filter
    if threshold is not None:
        scored = [r for r in scored if r["similarity_score"] >= threshold]

    # sort by descending similarity
    scored.sort(key=lambda x: x["similarity_score"], reverse=True)

    # apply top_k
    if top_k is not None:
        scored = scored[:top_k]

    return scored
