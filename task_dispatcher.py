# task_dispatcher.py
# Uses your lang_graph.py to orchestrate. Provides Services (Planner, llm_json via DSPy, search, validate,
# scrape, extract, aggregate), Firestore logging (sanitized), and a recursion_limit on invoke.

from __future__ import annotations
import os
import json
import uuid
import time
from typing import Any, Dict, List, Optional, Iterable
from urllib.parse import urlparse
from extract import Extractor
from logger import DualLogger, logger  # Import the global logger

import requests
from dotenv import find_dotenv, load_dotenv
env = find_dotenv()
if env:
    load_dotenv(env)

# --- Lang graph  ---
from lang_graph import build_graph  

# --- optional deps / your modules ---
try:
    import dspy
    _DSPY = True
except Exception:
    dspy = None
    _DSPY = False

try:
    from firebase import db  #  firebase.py should initialize Firestore
except Exception:
    db = None

try:
    import plan_research_task as plan_mod
except Exception:
    plan_mod = None

try:
    import semantic_ranker as sr
except Exception:
    sr = None

try:
    import url_scraper as us
except Exception as e:
    us = None
    print(f"[WARN] optional import 'url_scraper' failed: {e}")

try:
    import aggregate as agg
except Exception:
    agg = None

try:
    import stopping as stop_mod
except Exception:
    stop_mod = None

from services.db_adapter import PlanRunDbAdapter

# ============== Terminal pretty logger ==============
RESET="\033[0m"; BOLD="\033[1m"; CYAN="\033[36m"; YELLOW="\033[33m"; GREEN="\033[32m"; RED="\033[31m"; GRAY="\033[90m"
from logger import DualLogger
log = DualLogger("").log  # Empty run_id for now

# ============== Helpers ==============
def _is_valid_url(u: str) -> bool:
    try:
        p = urlparse(u)
        return p.scheme in ("http","https") and bool(p.netloc)
    except Exception:
        return False

def _dedupe_by_url(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen, out = set(), []
    for r in rows:
        u = (r.get("url") or "").strip()
        if not u or not _is_valid_url(u): 
            continue
        if u in seen: 
            continue
        seen.add(u); out.append(r)
    return out

def _truncate(obj: Any, n: int = 240) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False)
    except Exception:
        s = str(obj)
    return s if len(s) <= n else s[:n] + "…"




# ============== Firestore adapter with sanitization (avoid nested arrays) ==============
def _sanitize_for_firestore(obj: Any) -> Any:
    # Firestore disallows arrays of arrays. Convert list-of-tuples/lists into list of maps.
    if isinstance(obj, dict):
        return {k: _sanitize_for_firestore(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        if any(isinstance(x, (list, tuple)) for x in obj):
            out = []
            for x in obj:
                if isinstance(x, (list, tuple)) and len(x) == 2 and all(not isinstance(e, (list, tuple, dict)) for e in x):
                    out.append({"k": _sanitize_for_firestore(x[0]), "v": _sanitize_for_firestore(x[1])})
                else:
                    out.append({"value": _sanitize_for_firestore(x)})
            return out
        return [ _sanitize_for_firestore(x) for x in obj ]
    return obj



# ============== DSPy-backed JSON caller + Planner fallback ==============
def _init_dspy():
    if not _DSPY:
        return None
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY") or os.getenv("OPENAI_KEY")
    lm_id = os.getenv("DSPY_LM", "openai/gpt-4o-mini")
    if not api_key:
        logger.log("WARN", "OPENAI_API_KEY not set; llm_json/plan will fallback.")
        return None
    try:
        lm = dspy.LM(lm_id, api_key=api_key)
        dspy.settings.configure(lm=lm, cache=False)
        logger.log("OK", f"DSPy LM loaded: {lm_id}")
        return lm
    except Exception as e:
        logger.log("WARN", f"DSPy init failed ({lm_id}): {e}")
        return None

class _SigJSON(dspy.Signature if _DSPY else object):
    if _DSPY:
        system = dspy.InputField()
        user = dspy.InputField()
        json_out = dspy.OutputField()

class _SigPlan(dspy.Signature if _DSPY else object):
    if _DSPY:
        goal = dspy.InputField()
        plan = dspy.OutputField()

class DSPyHelpers:
    def __init__(self):
        self.lm = _init_dspy()

    def llm_json(self, system: str, user: Dict[str, Any]) -> Dict[str, Any]:
        if _DSPY and self.lm is not None:
            try:
                pred = dspy.Predict(_SigJSON)(system=system, user=user)
                out = getattr(pred, "json_out", None)
                if out is None and isinstance(pred, dict):
                    out = pred.get("json_out")
                if isinstance(out, str):
                    return json.loads(out)
                if isinstance(out, dict):
                    return out
                if isinstance(out, list):
                    return {"list": out}
            except Exception as e:
                logger.log("WARN", f"DSPy llm_json failed; fallback: {e}")
        # minimal fallback
        return {"actions": [], "notes": "fallback"}

    def plan(self, goal: str) -> Dict[str, Any]:
        # Only use LLM-backed planner. No fallback allowed.
        if _DSPY and self.lm is not None:
            try:
                # Always request 4 LLM-generated queries (original will be prepended)
                system = (
                    "You are a research planner. Return a JSON object with keys: "
                    "goal (string) and queries (array of 4 diverse, high-quality search queries that broaden the original goal). "
                    "Output must be valid JSON only."
                )
                user = {"goal": goal, "instructions": "Return JSON: {\"goal\": ..., \"queries\": [4 queries] }"}
                j = self.llm_json(system, user)
                queries = None
                if isinstance(j, dict):
                    if j.get("queries") and isinstance(j.get("queries"), list):
                        queries = j.get("queries")
                    elif j.get("list") and isinstance(j.get("list"), list):
                        queries = j.get("list")
                    elif isinstance(j.get("json_out"), dict) and j.get("json_out").get("queries"):
                        queries = j.get("json_out").get("queries")
                    elif isinstance(j.get("json_out"), list):
                        queries = j.get("json_out")
                elif isinstance(j, list):
                    queries = j

                # Clean and dedupe queries, ensure exactly 4 LLM queries
                clean_qs = []
                if queries:
                    clean_qs = [str(q).strip() for q in queries if str(q).strip()]
                # Remove any queries identical to the goal
                clean_qs = [q for q in clean_qs if q.lower() != goal.lower()]
                # Deduplicate
                seen = set()
                unique_qs = []
                for q in clean_qs:
                    if q not in seen:
                        seen.add(q)
                        unique_qs.append(q)
                # Pad or trim to exactly 4
                while len(unique_qs) < 4:
                    unique_qs.append(f"{goal} alternative {len(unique_qs)+1}")
                unique_qs = unique_qs[:4]
                # Prepend the original goal
                final_queries = [goal] + unique_qs
                return {
                    "goal": j.get("goal") if isinstance(j, dict) and j.get("goal") else goal,
                    "steps": ["search","validate","refine","scrape","extract","aggregate"],
                    "queries": final_queries
                }
            except Exception as e:
                logger.log("ERROR", f"DSPy plan failed, no fallback allowed: {e}")
                raise RuntimeError("LLM planner failed and fallback is disabled.")
        raise RuntimeError("LLM planner required but not available.")



# ============== Services object expected by lang_graph.py ==============
class Services:
    def __init__(self, searxng_url: Optional[str], google_key: Optional[str], google_cx: Optional[str], max_rounds: int):
        self.run_id = str(uuid.uuid4())
        self.logger = DualLogger(self.run_id)
        self.write_round_doc = self.logger.write_round_doc
        
        self.controls = {
            "max_rounds": max_rounds,
            "max_actions_per_round": 3,
            "max_slot_bindings": 100,
            "max_queries_per_action": 40,
        }
        # Enable LLM-based critics by default when DSPy/LLM is configured.
        # This can be toggled at runtime by modifying services.controls["use_llm_critics"].
        try:
            self.controls["use_llm_critics"] = bool(os.getenv("OPENAI_API_KEY"))
        except Exception:
            self.controls["use_llm_critics"] = False
        self.searxng_url = (searxng_url or os.getenv("SEARXNG_URL") or "").rstrip("/")
        self.google_key = google_key or os.getenv("GOOGLE_API_KEY")
        self.google_cx  = google_cx  or os.getenv("GOOGLE_CX")
        self.req_timeout = 12

        # adapters
        self.db = None  # will be set after plan creation
        self.context = self     # to support remember_json
        self._dspy = DSPyHelpers()
        self._ranker_model = None
        
        # Extractor
        self.extractor = Extractor(
            model_name=os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            chunk_size=int(self.controls.get("extract_chunk_size", 3000)),
            overlap=250,
            min_text_len=int(self.controls.get("min_extract_text_len", 100)),
            max_items_per_chunk=int(self.controls.get("max_items_per_chunk", 40)),
            max_items_regex_fallback=int(self.controls.get("max_items_regex_fallback", 20)),
        )

    # ---- tracing ----
    def trace(self, kind: str, node: str, state: Dict[str, Any] | None = None, extra: Dict[str, Any] | None = None):
        msg = f"{node} | {kind}"
        if extra: msg += f" | {extra}"
        logger.log("STEP" if kind in ("ENTER","EXIT","STOP") else "INFO", msg)

    # ---- ranker ----
    def _get_ranker_model(self):
        if self._ranker_model is None:
            self._ranker_model = sr.load_model("all-MiniLM-L6-v2")
            logger.log("OK", "[validate] loaded all-MiniLM-L6-v2")
        return self._ranker_model

    # ---- planner ----
    def plan(self, goal: str) -> Dict[str, Any]:
        plan = self._dspy.plan(goal)
        # enforce shapes
        plan_id = str(uuid.uuid4())
        plan["plan_id"] = plan_id
        plan["goal"] = plan.get("goal") or goal
        # init db adapter with plan id
        self.db = PlanRunDbAdapter(plan_id=plan_id, run_id=self.run_id)
        # save in research_plans/{plan_id}
        if db:
            db.collection("research_plans").document(plan_id).set(_sanitize_for_firestore(plan), merge=True)
        # also write to run
        self.db.write_artifact("plan", plan)
        logger.log("OK", f"Saved plan {plan_id} to Firestore.")
        return plan

    # ---- llm_json ----
    def llm_json(self, system: str, user: Dict[str, Any]) -> Dict[str, Any]:
        return self._dspy.llm_json(system, user)

    # ---- memory hook ----
    def remember_json(self, key: str, payload: Dict[str, Any]):
        self.db.remember_json(key, payload)

    # ---- query utilities used by refine nodes ----
    def bind_template(self, template: str, slots: Dict[str, List[str]], cap: int = 100) -> List[str]:
        if not slots:
            return [template]
        keys = list(slots.keys())
        vals = [slots[k] for k in keys]
        out = []
        def _recur(i: int, curr: str):
            if len(out) >= cap:
                return
            if i == len(keys):
                out.append(curr)
                return
            k = keys[i]
            for v in vals[i]:
                _recur(i+1, curr.replace(f"{{{k}}}", str(v)))
        _recur(0, template)
        return out[:cap]

    def dedupe_queries(self, qs: List[str], seen: List[str] | None = None) -> List[str]:
        sset = set((seen or []))
        out = []
        for q in qs:
            qn = " ".join(q.split())
            if qn and qn not in sset:
                sset.add(qn); out.append(qn)
        return out

    def filter_queries(self, qs: List[str], deny_terms: List[str]) -> List[str]:
        deny = [t.lower() for t in (deny_terms or [])]
        out = []
        for q in qs:
            low = q.lower()
            if any(t in low for t in deny):
                continue
            out.append(q)
        return out

    def filter_domains(self, qs: List[str], allowed: List[str]) -> List[str]:
        if not allowed:
            return qs
        out = []
        for q in qs:
            if "site:" in q:
                out.append(q)
            else:
                out.extend([f"{q} site:{dom}" for dom in allowed])
        return out

    # ---- should_stop adapter (graph's stop_check node will call this) ----
    def should_stop(self, before: Dict[str, Any], after: Dict[str, Any], *, round_idx: int, max_rounds: int) -> bool:
        # Prefer your stopping module if present
        if stop_mod and hasattr(stop_mod, "should_stop"):
            try:
                return bool(stop_mod.should_stop(before, after,
                                                 min_new_urls=3, min_new_domains=1, min_new_entities=5,
                                                 round_idx=round_idx, max_rounds=max_rounds))
            except Exception as e:
                logger.log("WARN", f"stopping.should_stop error: {e}")
        # fallback: max rounds or no growth
        prev = len(before.get("items", []) or [])
        curr = len(after.get("items", []) or [])
        if round_idx >= max_rounds: 
            return True
        if curr - prev <= 0:
            return True
        return False

    # ---- search/validate/scrape/extract/aggregate ----
    def execute_search(self, queries: List[str], round_idx: int | None = None) -> tuple[list[Dict[str, Any]], Dict[str, list]]:
        """
        Execute search for each query and return (combined_results, per_query_hits).
        """
        out: List[Dict[str, Any]] = []
        per_query_hits: Dict[str, list] = {}
        searx_total = 0

        for idx, q in enumerate(queries or []):
            hits: List[Dict[str, Any]] = []
            if self.searxng_url:
                try:
                    hits = self._searxng(q, n=15)
                    if int(len(hits)) != 0:
                        logger.log("INFO", f"[search] SearXNG hits for {q!r}: {len(hits)}")
                except Exception as e:
                    logger.log("WARN", f"[searxng] {e}")

            if int(len(hits)) == 0 and self.google_key and self.google_cx:
                logger.log("INFO", "[search] SearXNG=0 → Google fallback")
                try:
                    hits = self._google(q, n=10)
                    print(f"Google hits for {q!r}: {len(hits)}")
                except Exception as e:
                    logger.log("WARN", f"[google] {e}")

            # keep only valid-URL hits
            valid_hits = [h for h in (hits or []) if _is_valid_url(h.get("url", ""))]
            per_query_hits[q] = valid_hits

            # write per-query artifact for auditing (truncate heavy fields)
            try:
                if hasattr(self, 'db') and self.db:
                    art_key = f"search_query_{(round_idx or 0):03d}_{idx:03d}"
                    preview = [{"url": h.get("url"), "title": (h.get("title") or '')[:200], "snippet": (h.get("snippet") or '')[:400]} for h in valid_hits]
                    self.db.write_artifact(art_key, {"round": round_idx, "query": q, "hits_count": len(valid_hits), "hits_preview": preview})
            except Exception:
                pass

            out.extend(valid_hits)
            searx_total += len(hits or [])

        # Write all raw results before deduplication
        raw_count = len(out)
        logger.log("INFO", f"[search] Total raw results before dedupe: {raw_count}")

        # now dedupe combined results by url
        out = _dedupe_by_url(out)
        logger.log("INFO", f"[search] Results after dedupe: {len(out)}")

        # second-pass Google if nothing found
        if searx_total == 0 and self.google_key and self.google_cx and not out:
            logger.log("INFO", "[search] overall SearXNG=0 → second pass Google on all queries")
            for idx, q in enumerate(queries or []):
                for h in (self._google(q, n=10) or []):
                    if _is_valid_url(h.get("url", "")):
                        out.append(h)
                        per_query_hits.setdefault(q, []).append(h)

            out = _dedupe_by_url(out)

        return out, per_query_hits

    def _searxng(self, query: str, n: int = 15) -> List[Dict[str, Any]]:
        url = f"{self.searxng_url}/search"
        resp = requests.get(url, params={"q":query,"format":"json","language":"en","safesearch":1,"categories":"general"}, timeout=self.req_timeout)
        if resp.status_code == 403:
            raise RuntimeError(f"403 FORBIDDEN for {url} (check instance/settings)")
        resp.raise_for_status()
        data = resp.json()
        res = (data.get("results") or [])[:n]
        return [{"url": r.get("url"), "title": r.get("title"), "snippet": r.get("content")} for r in res if r.get("url")]

    def _google(self, query: str, n: int = 10) -> List[Dict[str, Any]]:
        if not (self.google_key and self.google_cx):
            return []
        url = "https://www.googleapis.com/customsearch/v1"
        params = {"key": self.google_key, "cx": self.google_cx, "q": query, "num": min(n,10)}
        out = []
        while True:
            r = requests.get(url, params=params, timeout=self.req_timeout)
            r.raise_for_status()
            data = r.json()
            items = data.get("items", []) or []
            for it in items:
                u = it.get("link")
                if _is_valid_url(u):
                    out.append({"url": u, "title": it.get("title"), "snippet": it.get("snippet")})
            if not items or len(out) >= n:
                break
            nxt = (data.get("queries", {}).get("nextPage") or [{}])[0].get("startIndex")
            if not nxt:
                break
            params["start"] = nxt
        return out[:n]

    def execute_validate(self, goal: str, candidates, top_k: int | None = None, threshold: float | None = None) -> list[str]:
        """
        Uses semantic_ranker.rank_results_by_similarity over title+snippet, with URL fallback.
        Accepts either:
        - candidates: List[Dict] with keys url,title?,snippet?
        - candidates: List[str] of URLs
        Returns: List[str] validated (ranked & filtered) URLs.
        """
        if top_k is None:
            top_k = int(self.controls.get("validate_top_k", 50))
        if threshold is None:
            threshold = float(self.controls.get("validate_threshold", 0.60))

        # Normalize inputs to a list of result dicts
        raw: list[dict] = []
        if isinstance(candidates, list) and candidates:
            if isinstance(candidates[0], str):
                # got a list of URLs only
                raw = [{"url": u, "title": "", "snippet": ""} for u in candidates if _is_valid_url(u)]
            elif isinstance(candidates[0], dict):
                raw = [r for r in candidates if isinstance(r, dict) and _is_valid_url((r.get("url") or "").strip())]

        if not raw:
            logger.log("WARN", "[validate] no candidates to rank")
            return []

        # If title+snippet are blank (some engines do that), fall back to URL text so it can still score
        for r in raw:
            if not (r.get("title") or r.get("snippet")):
                r["snippet"] = r.get("url", "")

        model = self._get_ranker_model()

        # Compute per-candidate scores so we can persist rejected examples with scores
        scores = {}
        for cand in raw:
            try:
                sc = sr.score_result_similarity(cand, goal, model).get("similarity_score", 0.0)
            except Exception:
                sc = 0.0
            scores[cand.get("url")] = float(sc)

        ranked = sr.rank_results_by_similarity(
            results=raw,
            query=goal,
            model=model,
            top_k=top_k,
            threshold=threshold
        ) or []

        urls = [r["url"] for r in ranked if _is_valid_url(r.get("url",""))]

        # persist validated + rejected to DB (under the current round)
        try:
            r_idx = int((self.state.get("session") or {}).get("round_idx", 1)) if hasattr(self, "state") else 1
            if hasattr(self, 'db') and self.db:
                # write validated URLs (simple list)
                self.db.write_validated(r_idx, urls)
                # build rejected list with scores for auditing
                rejected = []
                kept_set = set(urls)
                for cand in raw:
                    u = cand.get("url")
                    if not u or u in kept_set:
                        continue
                    rejected.append({"url": u, "score": scores.get(u, 0.0)})
                if rejected:
                    self.db.write_validate_rejected(r_idx, rejected)
        except Exception:
            pass

        logger.log("INFO", f"[validate] kept {len(urls)}/{len(raw)} (top_k={top_k}, thr={threshold}) | rejected={len(raw)-len(urls)}")
        return urls

    def execute_scrape(self, urls: List[str]) -> List[Dict[str, Any]]:
        if us and hasattr(us, "scrape_many"):
            logger.log("INFO", f"[scrape] using url_scraper.scrape_many on {len(urls)} urls")
            try:
                rows = us.scrape_many(urls) or []
                nonempty = sum(1 for r in rows if (r.get("clean_text") or r.get("text") or "").strip())
                logger.log("INFO", f"[scrape] url_scraper.scrape_many returned {len(rows)} rows ({nonempty} with text)")
                return rows
            except Exception as e:
                logger.log("WARN", f"[scrape] url_scraper.scrape_many error: {e}")
        # fallback: GET a few pages
        logger.log("INFO", f"[scrape] FALLBACK requests.get on {len(urls)} urls")
        rows = []
        for u in urls[:20]:
            try:
                r = requests.get(u, timeout=self.req_timeout)
                txt = r.text or ""
                logger.log("INFO", f"[scrape] GET {u} -> {r.status_code} len={len(txt)}")
                rows.append({"url": u, "text": txt})
            except Exception as e:
                logger.log("WARN", f"[scrape] GET {u} err: {e}")
        return rows

    def execute_extract(self, scrapes: list[dict], goal: str, *, existing_items: list[dict] | None = None) -> list[dict]:
        r = int((self.state.get("session") or {}).get("round_idx", 1)) if hasattr(self, "state") else 1
        return self.extractor.extract(
            scrapes=scrapes,
            goal=goal,
            db=self.db,                  # adapter with write_artifact / write_step / write_extracted
            plan_id=getattr(self, "plan_id", None),
            run_id=self.run_id,
            round_idx=r,
            controls=self.controls,
            existing_items=existing_items,
        )


    def execute_aggregate(self, before: Dict[str, Any], extracted: List[Dict[str, Any]]) -> Dict[str, Any]:
        if agg and hasattr(agg, "consolidate_and_dedupe"):
            try:
                out = agg.consolidate_and_dedupe(extracted)
                if isinstance(out, dict) and "items" in out:
                    return out
            except Exception as e:
                logger.log("WARN", f"[aggregate] error: {e}")
        # fallback: exact dedupe (name+address)
        items = (before.get("items") or []) + (extracted or [])
        def key(it): 
            return (str(it.get("name","")).strip().lower(), str(it.get("address","")).strip().lower())
        seen=set(); ded=[]
        for it in items:
            k=key(it)
            if not any(k): 
                continue
            if k in seen: 
                continue
            seen.add(k); ded.append(it)
        return {"items": ded}


# ============== TaskDispatcher that uses lang_graph.py ==============
class TaskDispatcher:
    def __init__(self,
                 searxng_url: Optional[str] = None,
                 max_rounds: int = 6,
                 recursion_limit: Optional[int] = None):
        self.searxng_url = searxng_url
        self.google_cse_key = os.getenv("GOOGLE_API_KEY") 
        self.google_cse_cx  = os.getenv("GOOGLE_CX") 
        self.max_rounds = max_rounds
        # default recursion_limit (LangGraph internal) ~ 4x rounds to be safe
        self.recursion_limit = 40
        # self.controls.update({
        #     "validate_top_k": 50,
        #     "validate_threshold": 0.60,
        # })

    def run(self, task: str) -> Dict[str, Any]:
        services = Services(self.searxng_url, self.google_cse_key, self.google_cse_cx, self.max_rounds)

        logger.log("STEP", f"LangGraph dispatch → {services.run_id} | goal = '{task}'")
        if _DSPY:
            logger.log("INFO", "DSPy available " + ("(LM configured)" if services._dspy.lm else "(LM NOT configured; fallback JSON)"))
        else:
            logger.log("INFO", "DSPy not installed; fallback JSON only.")

        # 1) PLAN (explicit step here, then graph uses it)
        plan = services.plan(task)  # saves to Firestore research_plans/{plan_id}
        seed_queries = plan.get("queries") or [task]
        services.db.write_run_root({
            "goal": task,
            "plan_id": plan.get("plan_id"),
            "status": "running",
            "seed_queries": seed_queries[:10]
        })

        # 2) Build graph and start with initial state that includes the goal AND the plan (so your graph can use it)
        app = build_graph(services)

        init_state = {
            "goal": plan.get("goal") or task,
            "plan_id": plan.get("plan_id"),
            "plan": plan,                # <-- your lang_graph can read this if it wants
            "queries": seed_queries,     # seed the first search
            # optional: any session scaffolding your graph expects
            "session": {"round_idx": 1}
        }

        # 3) Invoke with a recursion_limit to avoid GraphRecursionError
        state = app.invoke(init_state, config={"recursion_limit": self.recursion_limit})
        state.setdefault("session", {})
        state["session"].setdefault("round_idx", 1)
        state["session"].setdefault("zero_hit_streak", 0)
        state["session"].setdefault("zero_gain_streak", 0)
        state.setdefault("failed_urls", [])     # lists only (Firestore-safe)
        state.setdefault("failed_domains", [])
        state.setdefault("seen_urls", [])       # optional visibility in DB

        # persist final queries executed (fix: ensure Firestore shows exact queries run)
        try:
            final_qs = list(state.get("queries") or [])
            services.db.write_run_root({"final_queries": final_qs[:100]})
            logger.log("INFO", f"Final queries (count={len(final_qs)}): {final_qs[:8]}")
        except Exception:
            pass

        # attach run id so callers can find Firestore artifacts
        try:
            state["run_id"] = services.run_id
        except Exception:
            pass

        # small summary
        items = len(((state.get("aggregated") or {}).get("items") or []))
        logger.log("OK", f"Dispatch end | items={items} | rounds≈{(state.get('session') or {}).get('round_idx', '?')}")
        return state
