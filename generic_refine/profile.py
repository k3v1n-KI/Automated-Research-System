# generic_refine/profile.py
from __future__ import annotations
from typing import Dict, Any, List
from collections import Counter
import tldextract

Profile = Dict[str, Any]

def _is_year_like(v: Any) -> bool:
    try:
        i = int(str(v))
        return 1900 <= i <= 2100
    except:  # noqa: E722
        return False

def profile_results(items: List[Dict[str, Any]]) -> Profile:
    if not items:
        return {
            "item_count": 0, "fields": {}, "categorical_topk": {},
            "temporal": {}, "source_domains": {}, "dedupe": {}
        }

    fields: Dict[str, Dict[str, Any]] = {}
    cat_topk: Dict[str, List[Dict[str, Any]]] = {}
    src_domains = Counter()

    keys = set().union(*[i.keys() for i in items]) if items else set()
    for k in keys:
        vals = [i.get(k) for i in items]
        miss = sum(1 for v in vals if not v)
        sample = [v for v in vals if v][:200]
        # naive type guess
        uniq = {str(v).strip().lower() for v in sample}
        kind = "categorical" if len(uniq) < 30 else "text"
        if all(_is_year_like(v) for v in sample[:50]) and sample:
            kind = "numeric"
        fields[k] = {"missing_rate": miss / len(items), "type": kind, "examples": sample[:5]}
        if kind == "categorical":
            freq = Counter([str(v).strip() for v in sample if v])
            cat_topk[k] = [{"value": v, "count": c} for v, c in freq.most_common(20)]

    # domains
    for it in items:
        u = it.get("source_url") or it.get("url")
        if not u: continue
        ext = tldextract.extract(u)
        dom = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain
        if dom: src_domains[dom] += 1

    # temporal
    year_field = next((k for k, meta in fields.items() if meta["type"] == "numeric"), None)
    temporal = {}
    if year_field:
        years = [int(i.get(year_field)) for i in items if _is_year_like(i.get(year_field))]
        if years:
            hist = Counter(years)
            temporal = {"year_field": year_field, "min": min(years), "max": max(years), "hist": dict(hist)}

    # dedupe placeholder (hook your real numbers)
    dedupe = {"near_dup_rate": 0.0, "method": "vector-sim"}

    return {
        "item_count": len(items),
        "fields": fields,
        "categorical_topk": cat_topk,
        "temporal": temporal,
        "source_domains": dict(src_domains),
        "dedupe": dedupe
    }
