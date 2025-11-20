# generic_refine/critic.py
from __future__ import annotations
from typing import Dict, Any, List

RefinePlan = Dict[str, Any]  # {"actions": [...], "notes": "..."}

def llm_gap_analyst_node(state: Dict[str, Any], services) -> Dict[str, Any]:
    """
    Reads aggregated items (a small sample) + computed profile; asks LLM for a plan.
    Persists plan in Firestore and memory.
    """
    items = (state.get("aggregated") or {}).get("items", [])
    profile = state.get("profile") or {}

    # compact sample
    sample_items = items[: min(10, len(items))]

    sys = (
        "You are an IR/IE critic. Inspect the profile of current results and propose at most 3 refinement actions. "
        "Return ONLY JSON with keys: actions(list), notes(str). "
        "An action has keys: type(one of ExpandDimensionCoverage, EnrichMissingFields, DiversifySources, FixDuplicates, ExtendTemporalCoverage), "
        "justification(str), target_fields(list), dimension_field(str|null), missing_values(list), query_templates(list), "
        "negative_terms(list), allowed_domains(list), slots(object of lists), batch_size(int)."
    )
    user = {"profile": profile, "sample_items": sample_items}

    # Call your JSON-safe LLM helper (must validate JSON shape; truncate on failure)
    plan: RefinePlan = services.llm_json(system=sys, user=user, schema="RefinePlan")
    if not isinstance(plan, dict) or "actions" not in plan:
        plan = {"actions": [], "notes": "No valid plan returned."}

    # Persist
    services.db.write_step("llm_gap_analyst", {"input_profile": profile, "plan": plan})
    services.context.remember_json("refine_plan", plan)

    state["refine_plan"] = plan
    return state
