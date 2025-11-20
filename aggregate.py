# aggregate.py  (drop-in update)
import os
import json
import math
import numpy as np

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
from terminal_aesthetics import Spinner
from firebase import db
from context_vector_store import ContextVectorStore

# ——— Environment & Clients ———
dotenv_path = find_dotenv()
if dotenv_path:
    load_dotenv(dotenv_path)

plans_col    = db.collection("research_plans")
openai       = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL        = os.getenv("OPENAI_MODEL", "gpt-4")

# ——— Vector Store for RAG & Duplicate Detection ———
vector_store = ContextVectorStore()

# ======================================================================
# LEGACY (kept): batch LLM consolidation tied to plan_id/subtask
# ======================================================================
def execute_aggregate(plan_id: str, subtask: dict):
    """
    Legacy path used by earlier pipeline stages.
    1) Gather per-URL outputs from Firestore
    2) Use prior summary context
    3) Batch LLM consolidation
    4) Embedding-based soft dedupe
    5) Persist results and store new summary
    """
    # load goal
    plan = plans_col.document(plan_id).get().to_dict() or {}
    goal = plan.get("goal", "")

    # 1) Gather scraped items
    sources   = subtask["params"].get("source_subtasks", [])
    coll      = plans_col.document(plan_id).collection("results")
    all_items = []
    for src in sources:
        doc = coll.document(src).get().to_dict() or {}
        all_items.extend(doc.get("results", []))

    if not all_items:
        print(f"   ⚠️ No items to aggregate for {subtask['id']}")
        return []

    # 2) Retrieve last aggregate summary for context
    past_summaries = vector_store.query(goal, top_k=1)
    context_block  = past_summaries[0] if past_summaries else ""

    # 3) Batch‐wise GPT consolidation
    batch_size  = 100
    num_batches = math.ceil(len(all_items) / batch_size)
    consolidated = []

    for idx in range(num_batches):
        chunk      = all_items[idx*batch_size : (idx+1)*batch_size]
        chunk_json = json.dumps(chunk, ensure_ascii=False)

        prompt = f"""
Previously aggregated summary:
{context_block or 'None'}

You are finalizing the research goal:
  "{goal}"

Here is batch {idx+1}/{num_batches} containing {len(chunk)} items:
{chunk_json}

Please deduplicate within this batch—and avoid any items already mentioned above.
Return a JSON array of objects with keys "name" and "address" relevant to the goal.
Output raw JSON only.
"""
        with Spinner(f"[{subtask['id']}] Aggregating batch {idx+1}"):
            resp = openai.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role":"system", "content":"You consolidate scraped data into a list without repeats."},
                    {"role":"user",   "content":prompt}
                ],
                temperature=0.0
            ).choices[0].message.content

        try:
            partial = json.loads(resp)
        except json.JSONDecodeError:
            print(f"   ⚠️ Batch {idx+1} parse failed, skipping")
            partial = []

        consolidated.extend(partial)

    # 4) Approximate dedupe via embeddings
    seen_vecs = []
    final_items = []
    for item in consolidated:
        try:
            name    = (item.get("name") or "").strip()
            address = (item.get("address") or "").strip()
        except AttributeError:
            continue
        text = f"{name} {address}".strip()
        if not text:
            continue

        vec = vector_store.model.encode(text)
        if seen_vecs:
            sims = np.dot(np.array(seen_vecs), vec) / (np.linalg.norm(seen_vecs, axis=1) * np.linalg.norm(vec))
            if np.max(sims) > 0.90:
                continue
        final_items.append(item)
        seen_vecs.append(vec)

    # 5) Persist final results
    plans_col.document(plan_id).collection("results").document(subtask["id"]).set({"results": final_items})

    # 6) Store new summary in vector store
    summary = (
        f"Aggregate subtask {subtask['id']} produced "
        f"{len(final_items)} unique items: "
        + ", ".join((i.get('name') or '') for i in final_items[:5])
        + "..."
    )
    vector_store.add(
        summary,
        metadata={"plan_id": plan_id, "subtask_id": subtask["id"], "type": "aggregate"}
    )

    print(f"\r   ✓ [aggregate] {len(final_items)} total unique items saved{' '*10}")
    return final_items


# ======================================================================
# NEW (used by TaskDispatcher.Services.execute_aggregate): stateless,
# no-LLM aggregator for current LangGraph loop
# ======================================================================
def _sanitize_item(row: dict) -> dict:
    """Normalize fields and coerce to Firestore-safe, simple scalars."""
    if not isinstance(row, dict):
        return {}
    out = {
        "name":       str(row.get("name") or "").strip(),
        "address":    str(row.get("address") or "").strip(),
        "phone":      str(row.get("phone") or "").strip(),
        "website":    str(row.get("website") or "").strip(),
        "source_url": str(row.get("source_url") or row.get("url") or "").strip(),
    }
    # drop completely empty rows
    if not any(out.values()):
        return {}
    return out

def _exact_key(it: dict):
    return (
        (it.get("name") or "").strip().lower(),
        (it.get("address") or "").strip().lower(),
    )

def consolidate_and_dedupe(
    extracted: list[dict] | None,
    existing: list[dict] | None = None,
    sim_threshold: float = 0.90
) -> dict:
    """
    Stateless aggregator used by the dispatcher:
      - normalizes fields (name, address, phone, website, source_url)
      - exact dedupe on (name,address)
      - semantic soft-dedupe via SentenceTransformer (ContextVectorStore)
    Returns: {"items": [...]}  (Firestore-safe list of maps)
    """
    extracted = extracted or []
    existing  = existing or []

    # 1) normalize
    norm_existing = [x for x in (_sanitize_item(e) for e in existing) if x]
    norm_new      = [x for x in (_sanitize_item(e) for e in extracted) if x]

    # 2) exact dedupe using set of keys
    seen_keys = { _exact_key(e) for e in norm_existing if any(_exact_key(e)) }
    merged = list(norm_existing)
    for it in norm_new:
        k = _exact_key(it)
        if not any(k):
            continue
        if k in seen_keys:
            # prefer to fill missing website/phone from new record
            base = next((m for m in merged if _exact_key(m) == k), None)
            if base:
                if not base.get("website") and it.get("website"):
                    base["website"] = it["website"]
                if not base.get("phone") and it.get("phone"):
                    base["phone"] = it["phone"]
            continue
        seen_keys.add(k)
        merged.append(it)

    # 3) semantic soft-dedupe (cosine sim on MiniLM embeddings)
    kept: list[dict] = []
    kept_vecs = None  # lazy to avoid extra encode on empty
    for it in merged:
        text = f"{(it.get('name') or '').strip()} {(it.get('address') or '').strip()}".strip()
        if not text:
            continue
        vec = vector_store.model.encode(text)
        if kept_vecs is not None and len(kept_vecs):
            sims = np.dot(kept_vecs, vec) / (np.linalg.norm(kept_vecs, axis=1) * np.linalg.norm(vec))
            if float(np.max(sims)) >= sim_threshold:
                # treat as duplicate; optionally merge sparse fields
                j = int(np.argmax(sims))
                base = kept[j]
                if not base.get("website") and it.get("website"):
                    base["website"] = it["website"]
                if not base.get("phone") and it.get("phone"):
                    base["phone"] = it["phone"]
                continue
        # append and grow matrix
        kept.append(it)
        kept_vecs = np.vstack([kept_vecs, vec]) if kept_vecs is not None else np.expand_dims(vec, 0)

    return {"items": kept}
