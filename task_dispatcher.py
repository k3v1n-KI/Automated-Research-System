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
from terminal_aesthetics import Spinner
from url_scraper import read_website_fast, read_website_full, extract_information
from aggregate import execute_aggregate

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

    def _gpt_rank_results(self, plan_id, items, threshold):
        """
        Batch the items into chunks and call GPT-4 on each to avoid context overflow.
        Returns a final deduped + sorted list of {title,snippet,url,score}.
        """
        plan    = self.plans_col.document(plan_id).get().to_dict()
        goal    = plan["goal"]
        system_msg = {"role":"system", "content":"You apply scoring heuristics precisely."}

        batch_size = 20
        all_ranked = []

        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            items_json = json.dumps(batch, indent=2)

            user_prompt = (
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
        for item in all_ranked:
            url = item.get("url")
            if url and url not in seen:
                seen.add(url)
                unique.append(item)

        # Final sort
        unique.sort(key=lambda x: x["score"], reverse=True)
        return unique

    def execute_validate(self, plan_id, sub):
        p, coll = sub["params"], self.plans_col.document(plan_id).collection("results")
        threshold = p["threshold"]
        items = []
        for src in p["source_subtasks"]:
            items.extend(coll.document(src).get().to_dict().get("results", []))
        if not items:
            print(f"   ⚠️ [validate] no items for {sub['id']}")
            return

        ranked = self._gpt_rank_results(plan_id, items, threshold)
        # now drop malformed URLs
        filtered = [i for i in ranked if is_valid_url(i.get("url",""))]

        self.save_results(plan_id, sub["id"], filtered)
        print(f"\r   ✓ [validate] {len(filtered)} passed (thr={threshold}){' '*10}")

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
        # 1) Run your standalone aggregator to get back the final list
        final_items = execute_aggregate(plan_id, sub)

        # 2) Save into Firestore under research_plans/{plan_id}/results/{sub_id}
        self.save_results(plan_id, sub["id"], final_items)

        # 3) Log
        print(f"   ✓ [aggregate] saved {len(final_items)} items for subtask {sub['id']}")

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
                elif t == "scrape":
                    self.execute_scrape(plan_id, sub)
                elif t == "validate":
                    self.execute_validate(plan_id, sub)
                elif t == "api":
                    self.execute_api(plan_id, sub)
                elif t == "aggregate":
                    self.execute_aggregate(plan_id, sub)
                else:
                    print(f"   ⚠️ Unknown type {t}")
            self.mark_dispatched(plan_id)
            print(f"Plan {plan_id!r} completed.\n")


