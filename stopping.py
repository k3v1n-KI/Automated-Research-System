from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

def _base_domain(u: str) -> str:
    try:
        return urlparse(u).netloc.lower()
    except Exception:
        return ""

def compute_url_metrics(url_items: List[Dict[str, Any]]) -> Dict[str, int]:
    """Compute unique URL & domain counts."""
    urls    = [x.get("url", "") for x in url_items if x.get("url")]
    url_set = set(urls)
    domains = set(_base_domain(u) for u in url_set if u)
    return {
        "n_urls": len(url_set),
        "n_domains": len(domains),
    }

def compute_entity_metrics(entities: List[Dict[str, Any]]) -> Dict[str, int]:
    """Compute unique entity count using (name,address) tuple."""
    uniq = set()
    for e in entities or []:
        name = (e.get("name", "") or "").strip().lower()
        addr = (e.get("address", "") or "").strip().lower()
        uniq.add((name, addr))
    return {"n_entities": len(uniq)}

def should_stop(
    prev: Dict[str, int],
    curr: Dict[str, int],
    *,
    min_new_urls: int = 25,
    min_new_domains: int = 5,
    min_new_entities: int = 5,
    max_rounds: Optional[int] = None,
    round_idx: Optional[int] = None
) -> bool:
    """Stop on plateau or when max rounds reached."""
    if max_rounds is not None and round_idx is not None and (round_idx + 1) >= max_rounds:
        return True
    if not prev:
        return False
    gain_urls    = curr.get("n_urls", 0)     - prev.get("n_urls", 0)
    gain_domains = curr.get("n_domains", 0)  - prev.get("n_domains", 0)
    gain_entities= curr.get("n_entities", 0) - prev.get("n_entities", 0)
    return (gain_urls < min_new_urls) and (gain_domains < min_new_domains) and (gain_entities < min_new_entities)
