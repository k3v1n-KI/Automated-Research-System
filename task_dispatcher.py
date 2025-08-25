import os
import sys
import json
import time
import threading
import itertools
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
from refine_graph import build_graph
from stopping import compute_url_metrics, compute_entity_metrics, should_stop

# ——— Env Setup ———
dotenv_path = find_dotenv()
if dotenv_path:
    load_dotenv(dotenv_path)

SEARXNG_URL    = os.getenv("SEARXNG_URL") 
GOOGLE_CX      = os.getenv("GOOGLE_CX")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_RANDY_KEY")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-4.1-2025-04-14")


def is_valid_url(url: str) -> bool:
    p = urlparse(url or "")
    return p.scheme in ("http", "https") and bool(p.netloc)


class TaskDispatcher:
    def __init__(self):
        self.plans_col = db.collection("research_plans")
        self.openai    = OpenAI(api_key=OPENAI_API_KEY)
        # RAG vector store for context retrieval
        self.vector_store = ContextVectorStore()
        # Load semantic model for ranking
        self.semantic_model = load_model()
        self._refine_app = build_graph()

    # ——— History ———

    def _load_history(self, plan_id):
        doc = self.plans_col.document(plan_id).get().to_dict() or {}
        return doc.get("history", [])

    def _append_history(self, plan_id, role, content):
        history = self._load_history(plan_id)
        history.append({"role": role, "content": content})
        self.plans_col.document(plan_id).update({"history": history})

    # ——— Plan Fetch/Create ———

    def fetch_or_create_plan(self, task):
        docs = list(self.plans_col.where("dispatched", "==", False).stream())
        if not docs:
            with Spinner("Generating new plan"):
                plan = plan_research_task(task)
                plan["dispatched"] = False
                save_plan_to_firebase(plan)
            print(f"→ Created plan {plan['plan_id']}")
            docs = [self.plans_col.document(plan["plan_id"]).get()]
        return [(d.id, d.to_dict()) for d in docs]

    def mark_dispatched(self, plan_id):
        self.plans_col.document(plan_id).update({"dispatched": True})

    def save_results(self, plan_id, sub_id, payload):
        self.plans_col.document(plan_id) \
                     .collection("results") \
                     .document(sub_id) \
                     .set({"results": payload})

    # ——— Search ———

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
        except:
            return []

    def _google_search(self, query, desired):
        results, start = [], 1
        while len(results) < desired:
            to_fetch = min(10, desired - len(results))
            resp = requests.get(
                "https://www.googleapis.com/customsearch/v1",
                params={
                    "key": GOOGLE_API_KEY,
                    "cx": GOOGLE_CX,
                    "q": query,
                    "num": to_fetch,
                    "start": start
                },
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

    def execute_search(self, plan_id, sub):
        p       = sub["params"]
        # always fetch at least 25
        desired = max(p.get("limit", 25), 25)
        all_hits = []
        with Spinner(f"[{sub['id']}] Searching"):
            for q in p["query_variations"]:
                hits = self._searxng_search(q, desired)
                if len(hits) < desired:
                    hits += self._google_search(q, desired - len(hits))
                all_hits.extend(hits)

        # dedupe
        seen, deduped = set(), []
        for item in all_hits:
            if item["url"] not in seen:
                seen.add(item["url"])
                deduped.append(item)
            if len(deduped) >= desired:
                break

        self.save_results(plan_id, sub["id"], deduped)
        print(f"\r   ✓ [search] fetched {len(deduped)} URLs{' '*10}")
        
        # --- Summarize & store context ---
        summary = (
            f"Search subtask {sub['id']} returned {len(deduped)} URLs: "
            f"{', '.join(d['url'] for d in deduped[:5])} ..."
        )
        print("   → Summary: ", end="")
        print(f"   → {summary}")
        self.vector_store.add(
            summary,
            metadata={"plan_id": plan_id, "subtask_id": sub["id"], "type": "search"}
        )
    # ——— Scrape ———

    def execute_scrape(self, plan_id: str, sub: dict):
        """
        Behavior for "scrape" subtasks:
        Params schema:
          { "source_subtasks": [<validate_subtask_id>, ...] }

        Instead of scraping one URL, this will:
        - Fetch all URLs from each source_subtask's results
        - Scrape each page in turn
        - Extract new items via GPT-4
        - Dedupe across pages
        - Save the combined list under this subtask
        """
        sources = sub["params"].get("source_subtasks", [])
        coll    = self.plans_col.document(plan_id).collection("results")

        # 1) Collect every URL from your validation step(s)
        urls = []
        for src in sources:
            doc = coll.document(src).get().to_dict() or {}
            for item in doc.get("results", []):
                url = item.get("url", "").strip()
                if url:
                    urls.append(url)

        if not urls:
            print(f"   ⚠️ [scrape] no URLs found in {sources}, skipping")
            return

        # 2) Gather existing items for de-duplication context
        existing = []
        for src in sources:
            doc = coll.document(src).get().to_dict() or {}
            existing.extend(doc.get("results", []))

        all_extracted = []
        goal = self.plans_col.document(plan_id).get().to_dict()["goal"]

        # 3) Scrape & extract per URL
        for url in urls:
            text = ""
            try:
                with Spinner(f"[{sub['id']}] Fast scrape {url[:30]}"):
                    text = read_website_fast(url)
            except Exception:
                text = ""

            if not text:
                try:
                    with Spinner(f"[{sub['id']}] Full scrape {url[:30]}"):
                        text = read_website_full(url)
                except Exception as e:
                    print(f"   ⚠️ [scrape] failed for {url}: {e}")
                    continue

            if not text.strip():
                print(f"   ⚠️ [scrape] empty content for {url}, skipping")
                continue

            try:
                with Spinner(f"[{sub['id']}] Extracting {url[:30]}"):
                    extracted = extract_information(goal, text, existing)
            except Exception as e:
                print(f"   ⚠️ [scrape] extraction error {url}: {e}")
                continue

            all_extracted.extend(extracted or [])

        # 4) Deduplicate the extracted items globally
        unique, seen = [], set()
        for item in all_extracted:
            key = json.dumps(item, sort_keys=True)
            if key not in seen:
                seen.add(key)
                unique.append(item)

        # 5) Save combined results
        self.save_results(plan_id, sub["id"], unique)
        print(f"\r   ✓ [scrape] {len(unique)} total items from {len(urls)} pages{' ' * 20}")

    # ——— Validate ———

    def _gpt_rank_results(self, plan_id, items, threshold, sub):
        """
        Batch the items into chunks and call GPT-4 on each to avoid context overflow.
        Returns a final deduped + sorted list of {title,snippet,url,score}.
        """
        plan    = self.plans_col.document(plan_id).get().to_dict()
        goal    = plan["goal"]
        system_msg = {"role":"system", "content":"You apply scoring heuristics precisely."}
        # Retrieve top-5 relevant summaries for this goal
        context_snippets = self.vector_store.query(goal, top_k=5)
        context_block    = "\n\n".join(context_snippets)
        print(f"   → Context for validation: {context_block[:100]}...")
        batch_size = 20
        all_ranked = []

        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            items_json = json.dumps(batch, indent=2)

            user_prompt = (
                f"Context from past subtasks:\n{context_block}\n\n"
                f"Research goal:\n  \"{goal}\"\n\n"
                f"Batch {start//batch_size+1} of {((len(items)-1)//batch_size)+1}, "
                f"{len(batch)} results:\n{items_json}\n\n"
                "Score each entry 0–5 on relevance, authority, specificity, recency, extractability.\n"
                "Compute final_score = relevance*0.3 + authority*0.25 + specificity*0.2 "
                "+ recency*0.15 + extractability*0.1\n"
                f"Return a JSON array of {{'title','snippet','url','score'}} "
                f"including only entries with score ≥ {threshold}. No markdown."
            )

            # we only append this one user prompt into history if you like,
            # or skip history entirely to save tokens:
            # self._append_history(plan_id, "user", user_prompt)

            with Spinner(f"Ranking batch {start//batch_size+1}"):
                resp = self.openai.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[system_msg, {"role":"user", "content":user_prompt}],
                    temperature=0.0
                ).choices[0].message.content

            # self._append_history(plan_id, "assistant", resp)
            
            
            try:
                partial = json.loads(resp)
            except json.JSONDecodeError:
                partial = []
            all_ranked.extend(partial)

        # Dedupe by URL
        seen, unique = set(), []
        
        # Summarize validation results for context
        val_summary = (
            f"Validate subtask filtered to {len(unique)} URLs above threshold {threshold}."
        )
        self.vector_store.add(
            val_summary,
            metadata={"plan_id": plan_id, "subtask_id": sub.get("id"), "type": "validate"}
        )
        for item in all_ranked:
            url = item.get("url")
            if url and url not in seen:
                seen.add(url)
                unique.append(item)

        # Final sort
        unique.sort(key=lambda x: x["score"], reverse=True)
        return unique
    
    def _semantic_rank_results(self, plan_id, items, threshold, sub):
        """
        Use semantic similarity to rank items against the goal.
        Returns a final deduped + sorted list of {title,snippet,url,similarity_score}.
        """
        # load goal
        goal = self.plans_col.document(plan_id).get().to_dict().get("goal", "")
        ranked = rank_results_by_similarity(items, goal, self.semantic_model, top_k=None, threshold=None)
        for e in ranked:
            e["score"] = e.pop("similarity_score", 0.0)
        # apply threshold if provided (0–1)
        return [e for e in ranked if e["score"] >= threshold]

    def execute_validate(self, plan_id, sub):
        p, coll = sub["params"], self.plans_col.document(plan_id).collection("results")
        threshold = p["threshold"]
        items = []
        for src in p["source_subtasks"]:
            items.extend(coll.document(src).get().to_dict().get("results", []))
        if not items:
            print(f"   ⚠️ [validate] no items for {sub['id']}")
            return

        # ranked = self._gpt_rank_results(plan_id, items, threshold, sub)
        ranked = self._semantic_rank_results(plan_id, items, threshold, sub)
        # now drop malformed URLs
        filtered = [i for i in ranked if is_valid_url(i.get("url",""))]

        self.save_results(plan_id, sub["id"], filtered)
        print(f"\r   ✓ [validate] {len(filtered)} passed (thr={threshold}){' '*10}")
        
        # Summarize & store
        summary = f"Validate subtask {sub['id']} yielded {len(filtered)} valid URLs."
        self.vector_store.add(
            summary,
            metadata={"plan_id": plan_id, "subtask_id": sub["id"], "type": "validate"}
        )

    # ad-hoc validate for refined urls
    def _validate_ad_hoc(self, plan_id: str, urls: list[dict], threshold: float = 0.6) -> list[dict]:
        ranked = self._semantic_rank_results(plan_id, urls, threshold, sub={"id": "adhoc"})
        return [i for i in ranked if is_valid_url(i.get("url",""))]


    # NEW: execute_refine
    def execute_refine(self, plan_id: str, sub: dict):
        """
        Params: {"source_subtasks":[validate_id], "max_new_queries": int}
        Saves:  results/{sub['id']}:
                {
                "proposed_queries": [...],
                "refine_fetch": [...],           # raw fetched URLs
                "refine_validated": [...]        # optional: filtered via semantic ranker
                }
        """
        plan = self.plans_col.document(plan_id).get().to_dict() or {}
        goal = plan.get("goal", "")

        # collect validated text from source subtasks
        coll = self.plans_col.document(plan_id).collection("results")
        validated_text = ""
        for sid in sub["params"].get("source_subtasks", []):
            data = coll.document(sid).get().to_dict() or {}
            for r in data.get("results", []):
                validated_text += " " + (r.get("title", "") + " " + r.get("snippet", ""))

        # run the refinement graph with dispatcher search + ranker
        with Spinner(f"[{sub['id']}] Refining queries"):
            out = self._refine_app.invoke({
                "goal": goal,
                "_searx": lambda q, num=15: self._searxng_search(q, num),
                "_google": lambda q, num=15: self._google_search(q, num),
                "_ranker": lambda results, query: rank_results_by_similarity(results, query, self.semantic_model, top_k=200),
                "_validated_text": validated_text,
                "_max_new": int(sub["params"].get("max_new_queries", 8)),
            })

        proposed = out.get("refinements", [])
        ranked   = out.get("ranked", [])  # ranked results from both rounds (capped to top-200 in ranker)
        # fetch again just for explicit storage (optional)
        fetched = []
        for q in proposed:
            hits = self._searxng_search(q, 15)
            if len(hits) < 15:
                hits += self._google_search(q, 15 - len(hits))
            for h in hits:
                h["q"] = q
            fetched.extend(hits)
        # dedupe fetched by URL
        seen, ded = set(), []
        for it in fetched:
            u = it.get("url")
            if u and u not in seen:
                seen.add(u); ded.append(it)

        # optional: validate refined URLs via semantic ranker
        refined_valid = self._validate_ad_hoc(plan_id, ded, threshold=0.6)

        payload = {
            "proposed_queries": proposed,
            "refine_fetch": ded,
            "refine_validated": refined_valid,
            "ranked_preview": ranked[:100],
        }
        self.save_results(plan_id, sub["id"], payload)
        print(f"\r   ✓ [refine] {len(proposed)} queries → {len(ded)} URLs → {len(refined_valid)} validated{' '*10}")

    # COVERAGE DELTA (URLs & Entities)
    def _coverage_delta_report(self, plan_id: str, before_urls: list, after_urls: list, before_entities: list, after_entities: list):
        url_prev  = compute_url_metrics(before_urls)
        url_curr  = compute_url_metrics(after_urls)
        ent_prev  = compute_entity_metrics(before_entities)
        ent_curr  = compute_entity_metrics(after_entities)

        print("   — Coverage delta —")
        print(f"     URLs:     {url_prev['n_urls']} → {url_curr['n_urls']}  (domains: {url_prev['n_domains']} → {url_curr['n_domains']})")
        print(f"     Entities: {ent_prev['n_entities']} → {ent_curr['n_entities']}")

    # helper: collect entities from a list of subtask IDs in results/
    def _collect_entities(self, plan_id: str, subtask_ids: list[str]) -> list:
        coll = self.plans_col.document(plan_id).collection("results")
        out = []
        for sid in subtask_ids:
            doc = coll.document(sid).get().to_dict() or {}
            # scraper/aggregate save either "results": [ {name,address}, ... ]
            out.extend(doc.get("results", []))
        return out

    # ITERATIVE DISPATCH WITH REFINEMENT & STOPPING
    def dispatch_with_refinement(self, task: str, max_rounds: int = 2):
        plans = self.fetch_or_create_plan(task)
        for plan_id, plan in plans:
            print(f"Dispatching plan {plan_id!r} → {plan['goal']!r}")

            search_tasks    = [s for s in plan["subtasks"] if s["type"] == "search"]
            validate_tasks  = [s for s in plan["subtasks"] if s["type"] == "validate"]
            refine_tasks    = [s for s in plan["subtasks"] if s["type"] == "refine"]
            scrape_tasks    = [s for s in plan["subtasks"] if s["type"] == "scrape"]
            aggregate_tasks = [s for s in plan["subtasks"] if s["type"] == "aggregate"]

            # 1) initial search/validate
            for s in search_tasks:   self.execute_search(plan_id, s)
            for v in validate_tasks: self.execute_validate(plan_id, v)

            # snapshot BEFORE refinement
            coll = self.plans_col.document(plan_id).collection("results")
            val_id = validate_tasks[0]["id"] if validate_tasks else None
            before_urls = (coll.document(val_id).get().to_dict() or {}).get("results", []) if val_id else []
            before_entities = []  # not scraped yet

            prev_url_metrics = compute_url_metrics(before_urls)

            # 2) refinement loop
            for r_idx, sub in enumerate(refine_tasks[:max_rounds]):
                self.execute_refine(plan_id, sub)
                refined_pack = coll.document(sub["id"]).get().to_dict() or {}
                refined_valid = refined_pack.get("refine_validated", []) or refined_pack.get("refine_fetch", [])

                # merge for URL metrics view
                merged_urls = before_urls + [u for u in refined_valid if u.get("url") not in {x.get("url") for x in before_urls}]
                curr_url_metrics = compute_url_metrics(merged_urls)

                print(f"   ↪ refine round {r_idx+1}: URLs {prev_url_metrics['n_urls']} → {curr_url_metrics['n_urls']} | "
                    f"domains {prev_url_metrics['n_domains']} → {curr_url_metrics['n_domains']}")

                if should_stop(prev_url_metrics, curr_url_metrics, min_new_urls=25, min_new_domains=5, min_new_entities=5,
                            max_rounds=max_rounds, round_idx=r_idx):
                    print("   ✓ stopping refinement (plateau or max rounds).")
                    before_urls = merged_urls
                    break

                before_urls = merged_urls
                prev_url_metrics = curr_url_metrics

            # 3) scrape — include refined validated sources automatically
            if scrape_tasks:
                for sub in scrape_tasks:
                    srcs = set(sub["params"].get("source_subtasks", []))
                    for rsub in refine_tasks:
                        srcs.add(rsub["id"])  # scraper will read refine_validated/refine_fetch aware logic (if you added it)
                    sub["params"]["source_subtasks"] = list(srcs)
                    self.execute_scrape(plan_id, sub)

            # 4) aggregate
            for sub in aggregate_tasks:
                self.execute_aggregate(plan_id, sub)

            # snapshot AFTER refinement & scraping
            # gather all URLs we ended with: validated + all refine docs
            after_urls = list(before_urls)  # include validated+refine
            for rsub in refine_tasks:
                pack = coll.document(rsub["id"]).get().to_dict() or {}
                after_urls.extend(pack.get("refine_validated", []) or pack.get("refine_fetch", []))

            # collect entities from all scrape subtasks (they save into their own subtask doc ids)
            scrape_ids = [s["id"] for s in scrape_tasks]
            after_entities = self._collect_entities(plan_id, scrape_ids)

            # 5) Coverage delta summary
            self._coverage_delta_report(plan_id, before_urls, after_urls, before_entities, after_entities)

            self.mark_dispatched(plan_id)
            print(f"Plan {plan_id!r} completed.\n")
        
    # ——— API Stub ———

    def execute_api(self, plan_id, sub):
        with Spinner(f"[{sub['id']}] API stub"):
            time.sleep(0.5)
        print(f"\r   ✓ [api] skipped{' '*10}")

    # ——— Aggregate ———

    def execute_aggregate(self, plan_id, sub):
        """
        Subtask type "aggregate":
        params = {
            "source_subtasks": [string],
            "output_format": string
        }
        """
        # 1) Run your standalone aggregator to get back the final list/dict
        final_items = execute_aggregate(plan_id, sub)

        # Normalize shape to a dict with key "results"
        if isinstance(final_items, list):
            payload = {"results": final_items}
        elif isinstance(final_items, dict) and "results" in final_items:
            payload = final_items
        else:
            # attempt to coerce any other dict into a results list if it looks like entities
            maybe_list = final_items if isinstance(final_items, list) else []
            payload = {"results": maybe_list}

        # Optional: dedupe entities by (name,address) to reduce near-duplicates at save time
        dedup = []
        seen_pairs = set()
        for e in payload.get("results", []):
            name = (e.get("name","") or "").strip().lower()
            addr = (e.get("address","") or "").strip().lower()
            key  = (name, addr)
            if key not in seen_pairs:
                seen_pairs.add(key)
                dedup.append(e)
        payload["results"] = dedup

        # 2) Save into Firestore
        self.save_results(plan_id, sub["id"], payload)

        # 3) Vector-store summary
        summary = f"Aggregate subtask {sub['id']} consolidated {len(dedup)} items."
        self.vector_store.add(
            summary,
            metadata={"plan_id": plan_id, "subtask_id": sub["id"], "type": "aggregate"}
        )

        # 4) Log
        print(f"   ✓ [aggregate] saved {len(dedup)} items for subtask {sub['id']}")

    # ——— Dispatch Loop ———

    def dispatch(self, task: str):
        plans = self.fetch_or_create_plan(task)
        for plan_id, plan in plans:
            print(f"Dispatching plan {plan_id!r} → {plan['goal']!r}")
            for sub in plan["subtasks"]:
                print(f" → {sub['id']} ({sub['type']})")
                t = sub["type"]
                if t == "search":
                    self.execute_search(plan_id, sub)
                elif t == "validate":
                    self.execute_validate(plan_id, sub)
                elif t == "refine":
                    self.execute_refine(plan_id, sub)  # ← NEW
                elif t == "scrape":
                    self.execute_scrape(plan_id, sub)
                elif t == "api":
                    self.execute_api(plan_id, sub)
                elif t == "aggregate":
                    self.execute_aggregate(plan_id, sub)
                else:
                    print(f"   ⚠️ Unknown type {t}")
            self.mark_dispatched(plan_id)
            print(f"Plan {plan_id!r} completed.\n")

