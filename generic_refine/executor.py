# generic_refine/executor.py
from __future__ import annotations
from typing import Dict, Any, List

def _materialize_queries(action: Dict[str, Any], services) -> List[str]:
    templates = action.get("query_templates", []) or []
    slots = action.get("slots", {}) or {}
    cap = services.controls.get("max_slot_bindings", 100)

    q = []
    for t in templates:
        q.extend(services.bind_template(t, slots, cap=cap))
    # dedupe + domain guardrail + negative terms
    q = services.dedupe_queries(q, seen=services.state_seen_queries())
    q = services.filter_queries(q, deny_terms=action.get("negative_terms", []))
    allow = action.get("allowed_domains", [])
    if allow:
        q = services.filter_domains(q, allowed=allow)
    return q[: services.controls.get("max_queries_per_action", 40)]

def _measure_gain(before: Dict[str, Any], after: Dict[str, Any], services) -> Dict[str, Any]:
    # plug your real metrics here (new entity count, missing rate delta, dup rate delta, domain diversity delta, etc.)
    n_before = len((before or {}).get("items", []))
    n_after  = len((after or {}).get("items", []))
    return {"new_entities": max(0, n_after - n_before)}

def generic_refine_node(state: Dict[str, Any], services) -> Dict[str, Any]:
    plan = state.get("refine_plan", {}) or {}
    actions = plan.get("actions", [])[: services.controls.get("max_actions_per_round", 3)]
    if not actions:
        services.db.write_step("generic_refine", {"note": "No actions to execute"})
        return state

    gains_all = []
    for idx, action in enumerate(actions):
        queries = _materialize_queries(action, services)
        if not queries:
            services.db.write_step("refine_action_skipped", {"action": action, "reason": "no_queries"})
            continue

        # Execute the standard short loop
        results = services.execute_search(queries)
        validated = services.execute_validate(results)
        scraped = services.execute_scrape(validated)
        extracted = services.execute_extract(scraped, goal=state["goal"])
        before = state.get("aggregated")
        after = services.execute_aggregate(previous=before, new_items=extracted)

        delta = _measure_gain(before, after, services)
        gains_all.append({"action": action.get("type"), "queries": len(queries), "delta": delta})

        # Persist step + artifacts
        services.db.write_step("refine_action", {
            "idx": idx, "action": action, "queries": queries,
            "validated_urls": [r.get("url") for r in validated], "delta": delta
        })
        services.context.remember_json("refine_action_delta", gains_all[-1])

        state["aggregated"] = after
        state["queries"] = (state.get("queries") or []) + queries

    services.db.write_artifact("refine_round", {"gains": gains_all})
    state["refine_gains"] = gains_all
    return state
