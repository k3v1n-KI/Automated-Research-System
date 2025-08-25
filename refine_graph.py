from typing import Dict, Any, List
from langgraph.graph import StateGraph, END

class State(Dict[str, Any]):
    """
    {
      'goal': str,
      'queries': List[str],
      'results': List[dict],
      'ranked': List[dict],
      'refinements': List[str],
      # injected at runtime by dispatcher:
      '_searx': callable(query:str, num:int)->List[dict],
      '_google': callable(query:str, num:int)->List[dict],
      '_ranker': callable(results:list, query:str)->List[dict],
      '_validated_text': str,  # concatenated text from validated items (title+snippet)
      '_max_new': int
    }
    """
    pass

COMMON_CITIES = [
    "toronto","ottawa","mississauga","brampton","london","hamilton",
    "windsor","markham","vaughan","kitchener","waterloo","guelph","kingston","barrie"
]

def plan_node(state: State) -> State:
    goal = state["goal"]
    state["queries"] = list(dict.fromkeys([
        goal,
        f"{goal} site:.gov",
        f"{goal} site:.org",
        f"{goal} list",
        f"{goal} pdf"
    ]))[:8]
    return state

def search_node(state: State) -> State:
    qset = state.get("queries", [])
    results = []
    for q in qset:
        a = state["_searx"](q, 15)
        if len(a) < 15:
            a += state["_google"](q, 15 - len(a))
        for r in a:
            r["q"] = q
        results.extend(a)
    # dedupe by URL
    seen, ded = set(), []
    for r in results:
        u = r.get("url")
        if u and u not in seen:
            seen.add(u)
            ded.append(r)
    state["results"] = ded
    return state

def rank_node(state: State) -> State:
    goal = state["goal"]
    state["ranked"] = state["_ranker"](state["results"], goal)
    return state

def gaps_node(state: State) -> State:
    """Propose refinements based on cities missing from validated text + domain nudges."""
    goal = state["goal"]
    validated_text = state.get("_validated_text", "").lower()
    proposed = []
    for c in COMMON_CITIES:
        if c not in validated_text:
            proposed.append(f"{goal} {c}")
    proposed += [f"{goal} site:.edu", f"{goal} site:.ca"]
    proposed = list(dict.fromkeys(proposed))[: state.get("_max_new", 8)]
    state["refinements"] = proposed
    return state

def refine_node(state: State) -> State:
    qs = state.get("queries", [])
    refs = [q for q in state.get("refinements", []) if q not in qs]
    state["queries"] = (qs + refs)[:16]
    return state

def build_graph():
    g = StateGraph(State)
    g.add_node("plan", plan_node)
    g.add_node("search", search_node)
    g.add_node("rank", rank_node)
    g.add_node("gaps", gaps_node)
    g.add_node("refine", refine_node)
    g.set_entry_point("plan")
    g.add_edge("plan", "search")
    g.add_edge("search", "rank")
    g.add_edge("rank", "gaps")
    g.add_edge("gaps", "refine")
    g.add_edge("refine", "search")
    g.add_edge("rank", END)
    return g.compile()
