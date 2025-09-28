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
openai       = OpenAI(api_key=os.getenv("OPENAI_RANDY_KEY"))
MODEL        = os.getenv("OPENAI_MODEL", "gpt-4")

# ——— Vector Store for RAG & Duplicate Detection ———
vector_store = ContextVectorStore()

def execute_aggregate(plan_id: str, subtask: dict):
    """
    1) Gather all scrape outputs.
    2) Retrieve past aggregate summary from vector store for context.
    3) Batch‐wise GPT consolidation with prior context to avoid duplicates.
    4) Approximate dedupe via embeddings.
    5) Persist results and store new summary to vector store.
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
        chunk_json = json.dumps(chunk, indent=2)

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
        # print("Here's the prompt for batch consolidation:")
        # print(prompt)
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
    seen_embeddings = []
    final_items     = []
    for item in consolidated:
        try:
            name    = item.get("name","").strip()
            address = item.get("address","").strip()
        except AttributeError:
            continue
        text    = f"{name} {address}"
        if not text:
            continue

        vec = vector_store.model.encode(text)
        if seen_embeddings:
            sims = np.dot(seen_embeddings, vec) / (
                np.linalg.norm(seen_embeddings, axis=1) * np.linalg.norm(vec)
            )
            if np.max(sims) > 0.9:
                continue

        final_items.append(item)
        seen_embeddings.append(vec)

    # 5) Persist final results
    plans_col \
      .document(plan_id) \
      .collection("results") \
      .document(subtask["id"]) \
      .set({"results": final_items})

    # 6) Store new summary in vector store
    summary = (
        f"Aggregate subtask {subtask['id']} produced "
        f"{len(final_items)} unique items: "
        + ", ".join(i["name"] for i in final_items[:5])
        + "..."
    )
    # print("Here's the summary of the aggregation:")
    # print(summary)
    vector_store.add(
        summary,
        metadata={"plan_id": plan_id, "subtask_id": subtask["id"], "type": "aggregate"}
    )

    print(f"\r   ✓ [aggregate] {len(final_items)} total unique items saved{' '*10}")
    return final_items
