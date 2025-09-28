# task_dispatcher.py
import os
import json
import requests
from urllib.parse import urlparse
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

from firebase import db
from plan_research_task import plan_research_task, save_plan_to_firebase
from context_vector_store import ContextVectorStore
from terminal_aesthetics import Spinner
from url_scraper import read_website_fast, read_website_full, extract_information
from aggregate import execute_aggregate
from semantic_ranker import load_model, rank_results_by_similarity
from lang_graph import build_graph
from types import SimpleNamespace
from stopping import compute_url_metrics, should_stop

# ——— Env Setup ———
dotenv_path = find_dotenv()
if dotenv_path:
    load_dotenv(dotenv_path)

SEARXNG_URL    = os.getenv("SEARXNG_URL")
GOOGLE_CX      = os.getenv("GOOGLE_CX")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_RANDY_KEY")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4.1-2025-04-14")



def is_valid_url(url: str) -> bool:
    p = urlparse(url or "")
    return p.scheme in ("http", "https") and bool(p.netloc)

class TaskDispatcher:
    def __init__(self):
        self.plans_col = db.collection("research_plans")
        self.openai    = OpenAI(api_key=OPENAI_API_KEY)
        self.vector_store = ContextVectorStore()
        self.semantic_model = load_model()
        # self._app = build_graph()

    # ---- plan helpers ----
    def fetch_or_create_plan(self, task: str):
        docs = list(self.plans_col.where("dispatched", "==", False).stream())
        if not docs:
            with Spinner("Generating new plan"):
                plan = plan_research_task(task)
                plan["dispatched"] = False
                if not (plan.get("goal") or "").strip():
                    plan["goal"] = task.strip()
                save_plan_to_firebase(plan)
            docs = [self.plans_col.document(plan["plan_id"]).get()]

        out = []
        for d in docs:
            pdata = d.to_dict() or {}
            if not (pdata.get("goal") or "").strip():
                pdata["goal"] = task.strip()
                self.plans_col.document(d.id).update({"goal": pdata["goal"]})
            out.append((d.id, pdata))
        return out

    def mark_dispatched(self, plan_id):
        self.plans_col.document(plan_id).update({"dispatched": True})

    # ---- Firestore save/get adapters used by the graph ----
    def _save_subtask(self, plan_id: str, sub_id: str, payload: dict):
        self.plans_col.document(plan_id) \
            .collection("results") \
            .document(sub_id) \
            .set(payload)

    def _get_subtask(self, plan_id: str, sub_id: str) -> dict:
        return self.plans_col.document(plan_id) \
            .collection("results") \
            .document(sub_id) \
            .get().to_dict() or {}

    # ---- Search adapters ----
    def _searxng_search(self, query, desired):
        try:
            r = requests.get(
                f"{SEARXNG_URL}/search",
                params={"q": query, "format": "json", "categories": "general"},
                timeout=10
            )
            r.raise_for_status()
            hits = r.json().get("results", [])[:desired]
            return [
                {"title": h.get("title",""), "snippet": h.get("content",""), "url": h.get("url","")}
                for h in hits if h.get("url")
            ]
        except Exception:
            return []

    def _google_search(self, query, desired):
        results, start = [], 1
        while len(results) < desired:
            to_fetch = min(10, desired - len(results))
            resp = requests.get(
                "https://www.googleapis.com/customsearch/v1",
                params={"key": GOOGLE_API_KEY, "cx": GOOGLE_CX, "q": query, "num": to_fetch, "start": start},
                timeout=10
            ).json()
            items = resp.get("items", [])
            if not items:
                break
            for i in items:
                results.append({"title": i["title"], "snippet": i["snippet"], "url": i["link"]})
                if len(results) >= desired:
                    break
            start += len(items)
        return results

    # ---- Ranker adapter for graph (returns sorted with "score") ----
    def _ranker(self, results: list, query: str) -> list:
        ranked = rank_results_by_similarity(results, query, self.semantic_model, top_k=None, threshold=None)
        for e in ranked:
            e["score"] = e.pop("similarity_score", 0.0)
        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked

    # ---- Vector store adapter ----
    def _vector_add(self, text: str, metadata: dict):
        self.vector_store.add(text, metadata=metadata)

    # ---- Main: run with LangGraph as the dispatcher ----
    def run(self, task: str):
        plans = self.fetch_or_create_plan(task)
        for plan_id, plan in plans:
            print(f"LangGraph dispatch → {plan_id} | goal = {plan['goal']!r}")

            # --- services bound to THIS plan_id (no callables in state) ---
            services = SimpleNamespace(
                searx=lambda q, n=15: self._searxng_search(q, n),
                google=lambda q, n=15: self._google_search(q, n),
                ranker=lambda results, query: self._ranker(results, query),
                is_valid_url=is_valid_url,
                save_subtask=lambda sub_id, payload: self._save_subtask(plan_id, sub_id, payload),
                get_subtask=lambda sub_id: self._get_subtask(plan_id, sub_id),
                vector_add=lambda text, meta: self._vector_store_add(plan_id, text, meta),
                read_fast=read_website_fast,
                read_full=read_website_full,
                extract=lambda goal, text, existing: extract_information(goal, text, existing),
                aggregate=lambda pid, sub: execute_aggregate(pid, sub),
                compute_url_metrics=lambda urls: compute_url_metrics(urls),
                should_stop=lambda before, after, **kw: should_stop(before, after, **kw),
                fallback_goal=(plan.get("goal") or task or "web data gathering"),
            )

            app = build_graph(services)

            init_state = {
                "plan_id": plan_id,
                "goal": (plan.get("goal") or task or "").strip(),
                "threshold": 0.6,
                "max_per_query": 25,
                "max_queries": 24,
                "max_new_queries": 8,
                "max_rounds": 2,
            }

            with Spinner("LangGraph running"):
                # Optional: stream to verify state is present at entry
                # for ev in app.stream(init_state, stream_mode="updates"):
                #     print(ev)
                app.invoke(init_state)

            self.mark_dispatched(plan_id)
            print(f"✓ Plan {plan_id} completed via LangGraph.\n")
        
    def _vector_store_add(self, plan_id: str, text: str, meta: dict):
        self.vector_store.add(text, metadata={**(meta or {}), "plan_id": plan_id})    
        
