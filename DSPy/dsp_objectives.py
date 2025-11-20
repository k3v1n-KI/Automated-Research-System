# dsp_objectives.py
import json
import math
from typing import List

def _jaccard(a: set, b: set) -> float:
    if not a and not b: return 1.0
    return len(a & b) / max(1, len(a | b))

def _pairwise_diversity(queries: List[str]) -> float:
    toks = [set(q.lower().split()) for q in queries]
    n = len(toks)
    if n < 2: return 0.0
    s = 0.0; c = 0
    for i in range(n):
        for j in range(i+1, n):
            s += (1.0 - _jaccard(toks[i], toks[j]))
            c += 1
    return s / c

def _deny_penalty(queries: List[str], deny_terms: set) -> float:
    penalty = 0
    for q in queries:
        if any(dt in q.lower() for dt in deny_terms):
            penalty += 1
    return penalty / max(1, len(queries))

def seed_query_objective(example, pred) -> float:
    # example: .goal str (optionally .deny_terms), pred: list[str]
    queries = pred or []
    diversity = _pairwise_diversity(queries)
    deny = _deny_penalty(queries, set(getattr(example, "deny_terms", {"jobs","careers"})))
    # Simple objective: prefer diversity, avoid deny terms
    return max(0.0, 0.6*diversity - 0.4*deny)

def critic_objective(example, pred) -> float:
    # pred: actions_json str
    try:
        obj = json.loads(pred or "{}")
    except Exception:
        return 0.0
    keys = {"expand_sources","enrich_missing","deny_terms","max_queries"}
    has_keys = sum(k in obj for k in keys) / 4.0
    length = len(pred or "")
    compact = 1.0 / (1.0 + math.log10(max(10, length)))  # shorter is a bit better
    return 0.6*has_keys + 0.4*compact

def extract_objective(example, pred) -> float:
    # example.gold_items: list[dict], pred: list[dict]
    gold = getattr(example, "gold_items", []) or []
    pred = pred or []
    # crude F1 on 'name' presence overlap (lightweight offline proxy)
    gold_names = {g.get("name","").strip().lower() for g in gold if g.get("name")}
    pred_names = {p.get("name","").strip().lower() for p in pred if p.get("name")}
    tp = len(gold_names & pred_names)
    precision = tp / max(1, len(pred_names))
    recall = tp / max(1, len(gold_names)) if gold_names else 0.0
    if precision+recall == 0: f1 = 0.0
    else: f1 = 2*precision*recall/(precision+recall)
    # plus basic validity: % rows with required fields
    required = getattr(example, "required_fields", ["name","address"])
    valid = 0
    for p in pred:
        if all(p.get(f) for f in required): valid += 1
    valid_rate = valid / max(1, len(pred))
    return 0.7*f1 + 0.3*valid_rate
