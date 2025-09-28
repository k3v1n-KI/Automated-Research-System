# refine_graph.py
from __future__ import annotations
from typing import Any, Dict, List, Callable
from types import SimpleNamespace
from langgraph.graph import StateGraph, END

def _dedupe_by_url(items: List[dict]) -> List[dict]:
    seen, out = set(), []
    for it in items:
        u = it.get("url")
        if u and u not in seen:
            seen.add(u); out.append(it)
    return out

def _gen_seed_queries(goal: str) -> List[str]:
    base = [goal, f"{goal} site:.gov", f"{goal} site:.org", f"{goal} list", f"{goal} pdf"]
    return list(dict.fromkeys(base))

def build_graph(services: SimpleNamespace):
    """services must provide:
       searx, google, ranker, is_valid_url, save_subtask, get_subtask, vector_add,
       read_fast, read_full, extract, aggregate, compute_url_metrics, should_stop,
         fallback_goal
    """

    # ---------- Nodes (capture services via closure) ----------
    def plan_node(state: Dict[str, Any]) -> Dict[str, Any]:
        goal = (state.get("goal") or "").strip() or services.fallback_goal
        state["goal"] = goal

        if not state.get("queries"):
            state["queries"] = _gen_seed_queries(goal)[: int(state.get("max_queries", 24))]

        state.setdefault("round_idx", 0)
        state.setdefault("validated", [])
        state.setdefault("refinements", [])
        state.setdefault("refined_urls", [])
        state.setdefault("refined_validated", [])
        state.setdefault("prev_url_metrics", {"n_urls": 0, "n_domains": 0})
        state.setdefault("curr_url_metrics", {"n_urls": 0, "n_domains": 0})
        state.setdefault("_validated_text", state.get("_validated_text", ""))
        return state

    def search_node(state: Dict[str, Any]) -> Dict[str, Any]:
        desired = max(int(state.get("max_per_query", 25)), 15)
        all_hits: List[dict] = []
        for q in state.get("queries", []):
            hits = services.searx(q, desired)
            if len(hits) < desired:
                hits += services.google(q, desired - len(hits))
            for h in hits:
                h["q"] = q
            all_hits.extend(hits)
        state["results"] = _dedupe_by_url(all_hits)
        return state

    def validate_node(state: Dict[str, Any]) -> Dict[str, Any]:
        goal = state["goal"]  # guaranteed by plan_node
        results = state.get("results", [])
        ranked = services.ranker(results, goal)
        thr = float(state.get("threshold", 0.6))
        filtered = [r for r in ranked if r.get("score", 0) >= thr and services.is_valid_url(r.get("url", ""))]

        sub_id = f"validate_round_{int(state.get('round_idx', 0))}"
        services.save_subtask(sub_id, {"results": filtered})
        services.vector_add(f"Validate {sub_id} yielded {len(filtered)} URLs.", {"type": "validate", "subtask_id": sub_id})

        prev = state.get("validated", [])
        seen = {x.get("url") for x in prev}
        state["validated"] = prev + [x for x in filtered if x.get("url") not in seen]
        return state

    def gaps_node(state: Dict[str, Any]) -> Dict[str, Any]:
        goal = state["goal"]

        # Generic, engine-friendly refinements (no geography baked in)
        templates = [
            "{g} list",
            "{g} directory",
            "{g} database",
            "{g} registry",
            "{g} index",
            "{g} catalog",
            "{g} filetype:pdf",
            "{g} intitle:list",
            "{g} inurl:directory",
            "{g} site:.gov",
            "{g} site:.org",
            "{g} site:.edu",
        ]

        # Seed from what we already validated to avoid redoing obvious stuff later
        # (optional heuristic): if you prefer, keep it simple and skip this block.
        # validated_titles = " ".join([(x.get("title","")+" "+x.get("snippet","")) for x in state.get("validated", [])])

        proposed = [t.format(g=goal) for t in templates]

        # uniq & cap
        max_new = int(state.get("max_new_queries", 8))
        state["refinements"] = list(dict.fromkeys(proposed))[:max_new]
        return state

    def refine_node(state: Dict[str, Any]) -> Dict[str, Any]:
        proposed = state.get("refinements", [])
        if not proposed:
            return state

        fetched: List[dict] = []
        for q in proposed:
            hits = services.searx(q, 15)
            if len(hits) < 15:
                hits += services.google(q, 15 - len(hits))
            for h in hits:
                h["q"] = q
            fetched.extend(hits)
        fetched = _dedupe_by_url(fetched)

        goal = state["goal"]
        ranked = services.ranker(fetched, goal)
        thr = float(state.get("threshold", 0.6))
        refined_valid = [r for r in ranked if r.get("score", 0) >= thr and services.is_valid_url(r.get("url", ""))]

        sub_id = f"refine_round_{int(state.get('round_idx', 0))}"
        services.save_subtask(sub_id, {"proposed_queries": proposed, "refine_fetch": fetched, "refine_validated": refined_valid})

        # unions
        def _merge(prev, new, key="url"):
            seen = {x.get(key) for x in prev}
            return prev + [x for x in new if x.get(key) not in seen]

        state["refined_urls"]      = _merge(state.get("refined_urls", []), fetched)
        state["refined_validated"] = _merge(state.get("refined_validated", []), refined_valid)
        state["validated"]         = _merge(state.get("validated", []), refined_valid)

        vtext = " ".join([(x.get("title","") + " " + x.get("snippet","")) for x in state["validated"]][:200])
        state["_validated_text"] = vtext
        return state

    def stop_check_node(state):
        before = state.get("prev_url_metrics", {"n_urls": 0, "n_domains": 0})
        after  = services.compute_url_metrics(state.get("validated", []))
        state["curr_url_metrics"] = after

        max_rounds = int(state.get("max_rounds", 2))
        r_idx = int(state.get("round_idx", 0))

        cont = not services.should_stop(
            before, after,
            min_new_urls=25,
            min_new_domains=5,
            min_new_entities=5,
            round_idx=int(state.get("round_idx", 0)),
            max_rounds=int(state.get("max_rounds", 2)),
        )
        state["should_continue"] = cont
        state["prev_url_metrics"] = after
        state["round_idx"] = r_idx + 1
        return state

    def scrape_node(state: Dict[str, Any]) -> Dict[str, Any]:
        urls = [x.get("url") for x in state.get("validated", []) if x.get("url")]
        urls = list(dict.fromkeys(urls))
        goal = state["goal"]

        existing_items: List[dict] = []
        all_extracted: List[dict] = []
        for u in urls:
            text = ""
            try: text = services.read_fast(u)
            except Exception: text = ""
            if not text:
                try: text = services.read_full(u)
                except Exception: continue
            if not text.strip(): continue

            try: extracted = services.extract(goal, text, existing_items)
            except Exception: extracted = []
            if extracted:
                all_extracted.extend(extracted)
                existing_items.extend(extracted)

        # exact-ish dedupe
        seen, uniq = set(), []
        for it in all_extracted:
            key = str(sorted(it.items()))
            if key not in seen:
                seen.add(key); uniq.append(it)

        services.save_subtask("scrape_combined", {"results": uniq})
        return state

    def aggregate_node(state: Dict[str, Any]) -> Dict[str, Any]:
        payload = services.aggregate(state.get("plan_id"), {"id": "aggregate_final", "params": {
            "source_subtasks": ["scrape_combined"], "output_format": "json"
        }})
        if isinstance(payload, list):
            payload = {"results": payload}
        elif not (isinstance(payload, dict) and "results" in payload):
            payload = {"results": []}

        dedup, seen_pairs = [], set()
        for e in payload.get("results", []):
            name = (e.get("name","") or "").strip().lower()
            addr = (e.get("address","") or "").strip().lower()
            key = (name, addr)
            if key not in seen_pairs:
                seen_pairs.add(key); dedup.append(e)
        services.save_subtask("aggregate_final", {"results": dedup})
        return state

    # ---------- Graph ----------
    g = StateGraph(dict)
    g.add_node("plan", plan_node)
    g.add_node("search", search_node)
    g.add_node("validate", validate_node)
    g.add_node("gaps", gaps_node)
    g.add_node("refine", refine_node)
    g.add_node("stop_check", stop_check_node)
    g.add_node("scrape", scrape_node)
    g.add_node("aggregate", aggregate_node)

    g.set_entry_point("plan")
    g.add_edge("plan", "search")
    g.add_edge("search", "validate")
    g.add_edge("validate", "gaps")
    g.add_edge("gaps", "refine")
    g.add_edge("refine", "stop_check")

    def branch_stop(state: Dict[str, Any]):
        return "search" if state.get("should_continue") else "scrape"

    g.add_conditional_edges("stop_check", branch_stop, {"search": "search", "scrape": "scrape"})
    g.add_edge("scrape", "aggregate")
    g.add_edge("aggregate", END)

    return g.compile()
