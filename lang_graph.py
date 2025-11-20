# lang_graph.py
from __future__ import annotations
from typing import TypedDict, Dict, Any, List, Optional

# LangGraph (0.2+)
from langgraph.graph import StateGraph, START, END

from logger import logger  # Import the global logger

# If you already split these to modules, you can keep using them.
# Here we keep profile in-file (small) and call services.llm_json / refine executor via services functions.

def _wrap_node(name: str, fn, services):
    def _inner(state):
        services.trace("ENTER", node=name, state=state)
        out = fn(state)
        services.trace("EXIT", node=name, state=out)
        return out
    return _inner

class State(TypedDict, total=False):
    plan_id: str
    goal: str
    plan: Dict[str, Any]

    # pipeline
    queries: List[str]
    raw_results: List[Dict[str, Any]]
    validated: List[Dict[str, Any]]
    scraped: List[Dict[str, Any]]
    extracted: List[Dict[str, Any]]
    aggregated: Dict[str, Any]            # {"items": [...]}

    # profiling & refinement
    profile: Dict[str, Any]
    refine_plan: Dict[str, Any]
    refine_gains: List[Dict[str, Any]]

    # session / control
    session: Dict[str, Any]               # {"round_idx": int, "summary": str}

# ------------------------
# Utilities used by nodes
# ------------------------

def _ensure(state: State, key: str, default):
    if key not in state or state[key] is None:
        state[key] = default
    return state[key]

def _write_step(services, name: str, payload: Dict[str, Any]):
    try:
        services.db.write_step(name, payload)
    except Exception as e:
        print(f"[warn] write_step({name}) failed: {e}")

def _write_artifact(services, kind: str, payload: Dict[str, Any]):
    try:
        services.db.write_artifact(kind, payload)
    except Exception as e:
        print(f"[warn] write_artifact({kind}) failed: {e}")

# ------------------------
# Nodes
# ------------------------
def plan_node(state, services):
    import re

    # Diagnostic: log incoming state keys and plan shape to help debug why
    # the plan sometimes appears missing at runtime.
    try:
        logger.log("INFO", f"[plan_node] incoming state keys: {list(state.keys())}")
        logger.log("INFO", f"[plan_node] plan(raw): {str(state.get('plan'))[:400]}")
    except Exception:
        pass

    goal = (state.get("goal") or "").strip()
    plan = state.get("plan")

    # Enforce that a planner (LLM/DSPy) provides a plan with queries.
    # Do NOT silently fall back to deterministic, goal-derived query templates.
    if not plan or not isinstance(plan, dict):
        logger.log("ERROR", "[plan] Missing LLM plan; DSPy planner required. Aborting to avoid fallback queries.")
        raise RuntimeError("LLM plan missing. DSPy planner must provide a plan with 'queries'.")

    seeds = [q.strip() for q in (plan.get("queries") or []) if q and str(q).strip()]
    seeds = list(dict.fromkeys(seeds))

    # Detect common deterministic fallback patterns like
    # [goal, f"{goal} list", f"{goal} directory", f"{goal} website"].
    def _looks_like_fallback(qs: list[str]) -> bool:
        if not qs or not goal:
            return False
        # match queries that begin with the goal and end with a small set
        # of fallback suffixes (list/directory/website/site)
        suffix_re = re.compile(rf"^{re.escape(goal)}\s+(list|directory|website|site|list of|directory of)$", re.IGNORECASE)
        count = 0
        for q in qs:
            try:
                if isinstance(q, str) and suffix_re.search(q.strip()):
                    count += 1
            except Exception:
                continue
        return count >= 2

    allow_fallback = bool(services.controls.get("allow_fallback_queries", False))
    if _looks_like_fallback(seeds) and not allow_fallback:
        logger.log("ERROR", "[plan] Detected fallback-style queries in plan; aborting. Set services.controls['allow_fallback_queries']=True to override.")
        raise RuntimeError("Detected fallback-style queries in plan; aborting to enforce LLM-only queries.")

    _write_step(services, "plan", {
        "goal": goal,
        "plan_id": state.get("plan_id"),
        "queries_preview": seeds[:10]
    })

    return {
        "plan": plan,
        "queries": seeds,
        "session": {"round_idx": 1, "zero_hit_streak": 0, "zero_gain_streak": 0},
        "failed_urls": [],
        "failed_domains": [],
        "seen_urls": []
    }




def search_node(state, services):
    queries = state.get("queries") or []
    curr_round = int((state.get("session") or {}).get("round_idx", 1))
    raw, per_query_hits = services.execute_search(queries, round_idx=curr_round)
    # preserve current round_idx when returning session updates so we don't
    # accidentally overwrite round_idx (shallow merge semantics).
    curr_round = int((state.get("session") or {}).get("round_idx", 1))
    prev = int((state.get("session") or {}).get("zero_hit_streak", 0))
    zero_hit = 0 if (raw and len(raw) > 0) else (prev + 1)

    services.db.write_search_results(curr_round, raw)
    # persist per-query mapping for auditing (query -> hits list)
    try:
        services.db.write_artifact(f"search_per_query_{curr_round:03d}", {"round": curr_round, "per_query_hits": per_query_hits})
    except Exception:
        pass
    services.db.write_step("search", {"queries": len(queries), "raw_results": len(raw or [])})

    # Structured round logging for search
    try:
        per_query_counts = {q: len(v or []) for q, v in per_query_hits.items()}
        services.write_round_doc(curr_round, "search", {
            "queries": queries,
            "total_raw_results": len(raw or []),
            "deduped_results": len(raw or []),
            "per_query_counts": per_query_counts
        })
    except Exception:
        pass

    seen = set(state.get("seen_urls") or [])
    for r in raw or []:
        u = r.get("url")
        if u: seen.add(u)

    # also attach per_query_hits to the state for downstream nodes
    state["per_query_hits"] = per_query_hits

    # return session with round_idx preserved to avoid replacing it
    return {
        "raw_results": raw,
        "seen_urls": list(seen),
        "session": {"round_idx": curr_round, "zero_hit_streak": zero_hit}
    }


def validate_node(state, services):
    goal = state.get("goal", "")
    raw = state.get("raw_results") or []  # [{'url','title','snippet',...}, ...]
    queries = state.get("queries") or []
    
    logger.log("INFO", f"[validate] Processing {len(raw)} raw results")

    top_k = int(services.controls.get("validate_top_k", 50))
    threshold = float(services.controls.get("validate_threshold", 0.60))

    # Track per-query validation results
    per_query = {}
    for idx, query in enumerate(queries):
        query_results = [r for r in raw if r.get("query_idx") == idx]
        if query_results:
            per_query[query] = {
                "hits": len(query_results),
                "urls": [r.get("url") for r in query_results]
            }

    # pass raw results so the ranker can use title+snippet signals
    ranked_urls = services.execute_validate(goal, raw, top_k=top_k, threshold=threshold) or []
    logger.log("INFO", f"[validate] After ranking: {len(ranked_urls)} URLs (top_k={top_k}, threshold={threshold})")

    # Update per-query stats with kept URLs
    for query, stats in per_query.items():
        kept = [u for u in ranked_urls if u in stats["urls"]]
        stats["kept"] = len(kept)
        stats["discarded"] = stats["hits"] - len(kept)
        stats["kept_urls"] = kept

    deny_domains = set(state.get("failed_domains", []))
    def _allowed(u: str) -> bool:
        return not any(d in (u or "") for d in deny_domains)

    validated, seen = [], set()
    for u in ranked_urls:
        if not u or u in seen: 
            continue
        if not _allowed(u):
            continue
        seen.add(u)
        validated.append(u)
        
    # Write validation results to Firestore
    services.write_round_doc(state["session"]["round_idx"], "validate", {
        "raw_count": len(raw),
        "kept_count": len(validated),
        "queries": queries,
        "per_query": per_query
    })

    r = int((state.get("session") or {}).get("round_idx", 1))
    services.db.write_validated(r, validated)
    services.db.write_step("validate", {"ranked": len(ranked_urls), "validated": len(validated)})

    # Persist which URLs were kept/discarded per query (if we have per_query_hits)
    try:
        per_query_hits = state.get("per_query_hits") or {}
        per_query_validation = []
        validated_set = set(validated or [])
        for q, hits in (per_query_hits or {}).items():
            raw_urls = [h.get("url") for h in (hits or []) if h.get("url")]
            kept = [u for u in raw_urls if u in validated_set]
            discarded = [u for u in raw_urls if u not in validated_set]
            per_query_validation.append({"query": q, "raw_count": len(raw_urls), "kept": kept, "discarded": discarded})

        if per_query_validation:
            try:
                services.db.write_artifact(f"validate_per_query_{r:03d}", {"round": r, "per_query": per_query_validation})
            except Exception:
                pass
    except Exception:
        pass

    return {"validated": validated}



def scrape_node(state, services):
    validated = state.get("validated") or []
    scrapes = services.execute_scrape(validated)
    state["scrapes"] = scrapes

    scraped_urls = {s.get("url") for s in (scrapes or []) if s.get("url")}
    failed = sorted(list(set(validated) - scraped_urls))

    failed_urls = set(state.get("failed_urls", []))
    failed_urls.update([u for u in failed if u])
    state["failed_urls"] = list(failed_urls)

    failed_domains = set(state.get("failed_domains", []))
    from urllib.parse import urlparse
    for u in failed:
        try:
            net = urlparse(u).netloc
            if net:
                failed_domains.add(net)
        except Exception:
            pass
    state["failed_domains"] = list(failed_domains)

    #  actually persist scrapes so you can inspect them
    r = int((state.get("session") or {}).get("round_idx", 1))
    for s in scrapes or []:
        if "html" in s:
            s.pop("html", None)
    services.db.write_scrapes(r, scrapes)
    services.db.write_step("scrape", {
        "validated": len(validated or []),
        "scraped": len(scrapes or []),
        "failed_urls": len(state.get("failed_urls") or []),
        "failed_domains": len(state.get("failed_domains") or [])
    })
    # Structured round logging for scrape
    try:
        services.write_round_doc(r, "scrape", {
            "validated_count": len(validated or []),
            "scraped_count": len(scrapes or []),
            "failed_urls": list(state.get("failed_urls") or []),
            "failed_domains": list(state.get("failed_domains") or [])
        })
    except Exception:
        pass
    for s in scrapes or []:
        s.pop("html", None)
        # if your scraper puts text elsewhere, normalize it once here
        if "clean_text" not in s:
            for k in ("text", "content", "cleaned", "body"):
                v = s.get(k)
                if isinstance(v, str) and v.strip():
                    s["clean_text"] = v
                    break

    return {
        "scrapes": scrapes,
        "scraped": scrapes,
        "failed_urls": list(failed_urls),
        "failed_domains": list(failed_domains)
    }




# lang_graph.py

def _coalesce_text_for_extract(s: dict) -> str:
    for k in ("clean_text", "text", "content", "cleaned", "body"):
        v = s.get(k)
        if isinstance(v, str) and v.strip():
            return v
    html = s.get("html")
    if isinstance(html, str) and html.strip():
        try:
            from bs4 import BeautifulSoup
            return BeautifulSoup(html, "lxml").get_text(" ", strip=True)
        except Exception:
            pass
    return ""

def extract_node(state, services):
    goal = state.get("goal", "")
    # accept either 'scrapes' or 'scraped' (some codepaths/typos used one or the other)
    scrapes = state.get("scrapes") or state.get("scraped") or []
    # quick debug: log how many scrapes we received at the node entry
    try:
        services.trace("INFO", "extract_node_received", extra={"scrapes_len": len(scrapes)})
    except Exception:
        pass
    print(f"   ⏳ [extract] processing {len(scrapes)} scraped items...")
    r = int((state.get("session") or {}).get("round_idx", 1))

    usable = []
    examples = []
    for s in scrapes:
        txt = _coalesce_text_for_extract(s)
        if txt and len(txt) >= int(services.controls.get("min_extract_text_len", 120)):
            # store the coalesced text back so execute_extract doesn't have to recompute
            s = dict(s)
            s["clean_text"] = txt
            usable.append(s)
            if len(examples) < 5:
                examples.append({"url": s.get("url"), "len": len(txt)})

    services.db.write_step("extract_input", {
        "round": r,
        "scrapes_total": len(scrapes),
        "usable_for_extract": len(usable),
        "examples": examples
    })

    items = services.execute_extract(
        usable,
        goal,
        existing_items=(state.get("aggregated") or {}).get("items", [])
    ) or []

    services.db.write_extracted(r, items)
    services.db.write_step("extract", {"extracted": len(items)})

    # Structured round logging for extract
    try:
        services.write_round_doc(r, "extract", {
            "scrapes_total": len(scrapes),
            "usable_for_extract": len(usable),
            "extracted_count": len(items),
            "examples": examples[:10]
        })
    except Exception:
        pass

    # return only the diff you changed
    return {"extracted": items}





def aggregate_node(state, services):
    before = state.get("aggregated") or {"items": []}
    if not isinstance(before, dict):
        before = {"items": []}

    extracted = state.get("extracted") or []
    after = services.execute_aggregate(before, extracted)
    if not (isinstance(after, dict) and isinstance(after.get("items"), list)):
        after = {"items": list(before.get("items") or [])}

    prev_total = len(before.get("items") or [])
    curr_total = len(after.get("items") or [])
    no_gain = int(state.get("__no_gain_streak", 0))
    no_gain = 0 if curr_total > prev_total else (no_gain + 1)

    services.db.write_aggregated(int((state.get("session") or {}).get("round_idx", 1)), after.get("items") or [])
    services.db.write_step("aggregate", {"prev_total": prev_total, "new_total": curr_total, "__no_gain_streak": no_gain})

    # Structured round logging for aggregate
    try:
        services.write_round_doc(int((state.get("session") or {}).get("round_idx", 1)), "aggregate", {
            "prev_total": prev_total,
            "new_total": curr_total,
            "no_gain_streak": no_gain
        })
    except Exception:
        pass

    return {
        "aggregated": after,
        "__prev_total_items": curr_total,
        "__no_gain_streak": no_gain
    }




# --- Generic, domain-agnostic profiling of the aggregated items ---
def profile_node(state, services):
    agg = state.get("aggregated")
    if isinstance(agg, dict) and isinstance(agg.get("items"), list):
        items = agg["items"]
    elif isinstance(agg, list):
        items = agg
    else:
        items = []

    n = len(items)
    def cov(f):
        return round(sum(1 for it in items if isinstance(it, dict) and (it.get(f) or "").strip()) / n, 3) if n else 0.0

    from urllib.parse import urlparse
    dom = {}
    sample_missing_website = []
    sample_missing_address = []
    sample_missing_phone = []
    cities = {}

    for it in items:
        if isinstance(it, dict):
            u = (it.get("source_url") or it.get("url") or "").strip()
            try:
                net = urlparse(u).netloc
                if net:
                    dom[net] = dom.get(net, 0) + 1
            except Exception:
                pass

            if not (it.get("website") or "").strip() and len(sample_missing_website) < 5:
                sample_missing_website.append({"name": it.get("name"), "url": u})
            if not (it.get("address") or "").strip() and len(sample_missing_address) < 5:
                sample_missing_address.append({"name": it.get("name"), "url": u})
            if not (it.get("phone") or "").strip() and len(sample_missing_phone) < 5:
                sample_missing_phone.append({"name": it.get("name"), "url": u})

            # cheap city extraction: look for comma-separated tokens in address
            addr = (it.get("address") or "").strip()
            if addr and "," in addr:
                try:
                    city = addr.split(",")[-2].strip()
                    if city:
                        cities[city] = cities.get(city, 0) + 1
                except Exception:
                    pass

    top_domains = [{"domain": d, "count": c} for d, c in sorted(dom.items(), key=lambda x: -x[1])[:10]]

    profile = {
        "item_count": n,
        "field_coverage": {
            "name": cov("name"),
            "address": cov("address"),
            "phone": cov("phone"),
            "website": cov("website"),
        },
        "top_domains": top_domains,
        "domain_diversity": round(len(dom) / n, 3) if n else 0.0,
        "sample_missing": {
            "website": sample_missing_website,
            "address": sample_missing_address,
            "phone": sample_missing_phone,
        },
        "top_cities": [{"city": c, "count": cnt} for c, cnt in sorted(cities.items(), key=lambda x: -x[1])[:10]]
    }
    # Structured round logging for profile
    try:
        r = int((state.get("session") or {}).get("round_idx", 1))
        services.write_round_doc(r, "profile", {
            "total_items": n,
            "field_coverage": {
                "name": cov("name"),
                "address": cov("address"),
                "phone": cov("phone"),
                "website": cov("website")
            },
            "domain_distribution": dom,
            "sample_missing_website": sample_missing_website[:10],
            "sample_missing_address": sample_missing_address[:10],
            "sample_missing_phone": sample_missing_phone[:10]
        })
    except Exception:
        pass

    # DB write
    r = int((state.get("session") or {}).get("round_idx", 1))
    services.db.write_round(r, {"profile": profile})  # small dict; no chunking needed

    # Return snapshot and also attach under 'profile' for convenience
    return {"profile_snapshot": profile, "profile": profile}





def critic_node(state, services):
    goal    = (state.get("goal") or "").strip()
    items   = (state.get("aggregated") or {}).get("items", []) or []
    profile = state.get("profile") or state.get("profile_snapshot") or {}
    seen_qs = state.get("queries") or []
    failed  = state.get("failed_domains") or []

    # Start with a deterministic, rule-based critic that produces actionable 'actions'
    actions = []
    fcov = profile.get("field_coverage", {})
    item_count = int(profile.get("item_count", len(items)))
    top_domains = [d.get("domain") for d in (profile.get("top_domains") or [])]

    # If nothing found, analyze raw content and suggest focused queries
    if item_count == 0:
        # Get raw content from scrapes for analysis
        raw_scrapes = state.get("scrapes", [])
        raw_text = ""
        hospital_names = []
        
        for s in raw_scrapes:
            if isinstance(s, dict):
                text = s.get("clean_text", "") or s.get("text", "")
                raw_text += "\n" + text
                
                # Extract potential hospital names 
                import re
                hosp_matches = re.findall(r"(?im)(?:^|\n)([A-Z][^\n]{2,70}(?:Hospital|Medical Center|Healthcare)[^\n]{0,40}(?:$|\n))", text)
                hospital_names.extend(hosp_matches)

        if hospital_names:
            # Found hospital names but extraction failed - try more targeted queries
            actions.append({
                "query_templates": [
                    "{hosp} Toronto location",
                    "{hosp} Toronto contact information",
                    "{hosp} Toronto address phone website"
                ],
                "slots": {"hosp": h.strip()} 
            } for h in hospital_names[:3])  # Limit to top 3
            
        elif "hospital" in raw_text.lower():
            # Found hospital mentions but no clear names - try directory approach
            actions.append({
                "query_templates": [
                    "{goal} Ontario hospitals directory",
                    "{goal} Ontario hospitals list", 
                    "{goal} site:gov.on.ca",
                    "{goal} official website"
                ],
                "slots": {"goal": goal}
            })

    # If a single domain dominates results, ask to exclude it so we find other sources
    if item_count and top_domains:
        dominant = top_domains[0]
        dom_count = 0
        try:
            dom_count = next((d.get("count") for d in profile.get("top_domains", []) if d.get("domain") == dominant), 0)
        except Exception:
            dom_count = 0
        if dom_count and dom_count / max(1, item_count) > 0.6:
            actions.append({
                "query_templates": ["{goal} hospitals -site:%s" % dominant],
                "slots": {"goal": goal},
                "negative_terms": ["site:%s" % dominant]
            })

    # If websites are missing for many items, search for contact pages / official websites
    if fcov.get("website", 0.0) < 0.35:
        actions.append({
            "query_templates": ["{goal} hospital contact", "{goal} hospital website", "{goal} contact info"],
            "slots": {"goal": goal}
        })

    # If phone numbers are missing, try queries focused on phone/contact
    if fcov.get("phone", 0.0) < 0.35:
        actions.append({
            "query_templates": ["{goal} hospital phone", "{goal} hospital contact number"],
            "slots": {"goal": goal}
        })

    # If address coverage is low, include location-aware queries
    if fcov.get("address", 0.0) < 0.35:
        actions.append({
            "query_templates": ["{goal} hospital address", "{goal} location address"],
            "slots": {"goal": goal}
        })

    # As a last pass, allow an LLM to refine or expand these actions if configured to do so
    use_llm = bool(services.controls.get("use_llm_critics", False))
    llm_result = {}
    if use_llm:
        payload = {
            "goal": goal,
            "summary": profile,
            "seen_queries_sample": seen_qs[-15:],
            "failed_domains": failed[:20]
        }
        llm_result = services.llm_json(
            "Given the goal and a dataset profile, return JSON: {actions: [...], guardrails: {...}, notes: string}. "
            "Each action should include query_templates (list), slots (dict), optional negative_terms (list) and allowed_domains (list).",
            payload
        ) or {}

    # Merge LLM actions with rule-based actions (LLM actions appended if any)
    l_actions = llm_result.get("actions") or []
    merged_actions = actions + l_actions

    # If no actions were generated, provide a conservative fallback so the
    # system can still attempt another refinement round. This helps avoid
    # silent-stops where no refine plan is produced.
    if not merged_actions:
        fallback_qs = [
            "{goal} Ontario hospitals list",
            "{goal} Ontario hospitals directory",
            "{goal} hospital contact phone",
            "{goal} hospital website site:.gov.on.ca",
            "{goal} hospital address Ontario"
        ]
        merged_actions = [{
            "query_templates": fallback_qs,
            "slots": {"goal": goal},
            "notes": "fallback_queries_from_critic"
        }]

    refine_plan_next = {
        "actions": merged_actions,
        "guardrails": llm_result.get("guardrails") or {"max_actions_per_round": services.controls.get("max_actions_per_round", 3)},
        "notes": llm_result.get("notes", "rule_based_critic_used")
    }

    r = int((state.get("session") or {}).get("round_idx", 1))
    try:
        services.db.write_artifact(f"critic_round_{r:03d}", refine_plan_next)
    except Exception:
        pass

    return {"refine_plan_next": refine_plan_next}



def refine_node(state, services):
    """
    Consume `refine_plan` and materialize *queries only*.
    Do NOT call search/validate/scrape/extract/aggregate here.
    """
    plan = state.get("refine_plan") or {}
    actions = (plan.get("actions") or [])[: services.controls.get("max_actions_per_round", 3)]
    if not actions:
        _write_step(services, "refine", {"note": "no_actions"})
        return {"queries": state.get("queries") or []}

    bad_domains = set(state.get("failed_domains", []))
    new_queries_all = []

    def _domain_ok(q: str) -> bool:
        ql = (q or "").lower()
        if "site:" in ql:
            try:
                dom = ql.split("site:", 1)[1].split()[0]
                return dom not in bad_domains
            except Exception:
                pass
        return True

    for idx, act in enumerate(actions):
        templates = act.get("query_templates") or []
        slots     = act.get("slots") or {}
        deny      = act.get("negative_terms") or []
        allow     = act.get("allowed_domains") or []

        qs = []
        for t in templates:
            qs += services.bind_template(t, slots, cap=services.controls.get("max_slot_bindings", 100))

        qs = services.dedupe_queries(qs, seen=state.get("queries", []))
        qs = services.filter_queries(qs, deny_terms=deny)
        qs = services.filter_domains(qs, allowed=allow)
        qs = [q for q in qs if _domain_ok(q)]
        qs = qs[: services.controls.get("max_queries_per_action", 40)]

        _write_step(services, "refine_action_materialize", {
            "action_idx": idx, "queries_preview": qs[:10], "total": len(qs)
        })
        new_queries_all.extend(qs)

    updated_queries = (state.get("queries") or []) + new_queries_all
    _write_artifact(services, "refine_round", {
        "round": int((state.get("session") or {}).get("round_idx", 1)),
        "queries_added": len(new_queries_all)
    })

    # IMPORTANT: return a diff with *only* queries (no 'extracted'!)
    return {"queries": updated_queries}




def round_guard_node(state, services):
    r = state["session"].get("round_idx", 1)
    zero_hit = state["session"].get("zero_hit_streak", 0)
    zero_gain = state["session"].get("zero_gain_streak", 0)
    max_rounds = services.controls.get("max_rounds", 6)

    # hard stops
    # debug trace
    try:
        print(f"[STOP-GUARD] round={r} max_rounds={max_rounds} zero_hit={zero_hit} zero_gain={zero_gain}")
    except Exception:
        pass
    if r >= max_rounds:
        return {"stop": True, "reason": "max_rounds"}
    if zero_hit >= 2:
        return {"stop": True, "reason": "zero_hit_streak"}
    if zero_gain >= 2:
        return {"stop": True, "reason": "zero_gain_streak"}

    return {"stop": False}

def stop_check_node(state, services):
    sess = state.get("session") or {}
    r        = int(sess.get("round_idx", 1))
    max_r    = int(services.controls.get("max_rounds", 1))  # respect max_rounds=1
    zero_hit = int(sess.get("zero_hit_streak", 0))
    no_gain  = int(state.get("__no_gain_streak", 0))
    prev     = int(state.get("__prev_total_items", 0))
    curr     = len((state.get("aggregated") or {}).get("items") or [])
    new_ents = curr - prev

    decision = "CONTINUE"
    reason   = ""
    # debug trace
    try:
        print(f"[STOP-CHECK] r={r} max_r={max_r} zero_hit={zero_hit} no_gain={no_gain} prev={prev} curr={curr} new_ents={new_ents}")
    except Exception:
        pass
    if r >= max_r:
        decision, reason = "STOP", "max_rounds"
    elif zero_hit >= 2:
        decision, reason = "STOP", "zero_hit_streak"
    elif no_gain >= 2 or (new_ents <= 0 and not (state.get("validated") or [])):
        decision, reason = "STOP", "no_progress"

    services.db.write_step("stop_check", {
        "round": r, "decision": decision, "reason": reason,
        "zero_hit_streak": zero_hit, "no_gain_streak": no_gain, "new_entities": new_ents
    })

    # Update only the streak counters as top-level keys (not session)
    ret = {
        "__decision": decision,
        "__prev_total_items": curr,
        "__no_gain_streak": (no_gain + 1) if new_ents <= 0 else 0
    }
    return ret



def bump_round_node(state, services):
    sess = state.get("session") or {}
    new_round = int(sess.get("round_idx", 1)) + 1
    ret = {"session": {"round_idx": new_round}}  # ONLY this field

    if "refine_plan_next" in state:
        ret["refine_plan"] = state["refine_plan_next"]  # promote plan
    services.db.write_step("bump_round", {"round_idx": new_round})
    return ret




# ------------------------
# Graph builder
# ------------------------

def build_graph(services):
    g = StateGraph(State)

    # nodes 
    g.add_node("plan",      lambda s: plan_node(s, services))
    g.add_node("search",    lambda s: search_node(s, services))
    g.add_node("validate",  lambda s: validate_node(s, services))
    g.add_node("scrape",    lambda s: scrape_node(s, services))
    g.add_node("extract",   lambda s: extract_node(s, services))
    g.add_node("aggregate", lambda s: aggregate_node(s, services))
    g.add_node("profile",   lambda s: profile_node(s, services))
    g.add_node("critic",    lambda s: critic_node(s, services))
    g.add_node("refine",    lambda s: refine_node(s, services))
    g.add_node("stop_check",lambda s: stop_check_node(s, services))    # ← add ONCE
    g.add_node("bump_round", lambda s: bump_round_node(s, services))

    # edges
    g.add_edge(START, "plan")
    g.add_edge("plan", "search")
    g.add_edge("search", "validate")
    g.add_edge("validate", "scrape")
    g.add_edge("scrape", "extract")
    g.add_edge("extract", "aggregate")
    g.add_edge("aggregate", "profile")
    g.add_edge("profile", "critic")
    
    def _stop_decision(s):
        return s.get("__decision", "CONTINUE")

    g.add_conditional_edges(
        "stop_check",
        _stop_decision,
        {"STOP": "__end__", "CONTINUE": "bump_round"}
    )
    
    g.add_edge("bump_round", "refine")
    g.add_edge("refine", "search")

    # branch: refine if actions, otherwise go to stop_check
    def _has_actions(state):
        # critic returns a `refine_plan_next` payload; bump_round promotes it
        # to `refine_plan`. Check both keys so the conditional sees actions
        # immediately after the critic node and doesn't short-circuit to
        # stop_check.
        plan = state.get("refine_plan") or state.get("refine_plan_next") or {}
        return bool(plan.get("actions"))

    # If critic produced actions, advance to bump_round which will promote
    # the critic's `refine_plan_next` into `refine_plan` and increment the
    # round. However, avoid bumping past the configured max_rounds; if the
    # current round is already at or above max_rounds, route to stop_check
    # so the stop logic can end the run. This keeps the round cap enforced
    # even when the critic keeps producing actions.
    g.add_conditional_edges(
        "critic",
        lambda s: (
            "bump_round"
            if (_has_actions(s) and int((s.get("session") or {}).get("round_idx", 1)) < int(services.controls.get("max_rounds", 2)))
            else "stop_check"
        ),
        {"bump_round": "bump_round", "stop_check": "stop_check"},
    )

    # after refine, back to aggregate
    # g.add_edge("refine", "aggregate")

    



    return g.compile()

