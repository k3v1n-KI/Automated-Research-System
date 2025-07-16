import os
import json
import math

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
from terminal_aesthetics import Spinner
from firebase import db

# ——— Environment & Clients ———
dotenv_path = find_dotenv()
if dotenv_path:
    load_dotenv(dotenv_path)

plans_col   = db.collection("research_plans")
openai      = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL       = os.getenv("OPENAI_MODEL", "gpt-4")

def execute_aggregate(plan_id: str, subtask: dict):
    """
    Consolidate all scrape outputs via GPT-4 in chunks.
    Params:
      subtask["params"]["source_subtasks"]: list of validate subtask IDs
    Saves final deduped list under results/{subtask_id}.
    """
    # 1) Gather all scraped items from the named subtasks
    sources   = subtask["params"].get("source_subtasks", [])
    coll      = plans_col.document(plan_id).collection("results")
    all_items = []
    for src in sources:
        doc = coll.document(src).get().to_dict() or {}
        all_items.extend(doc.get("results", []))

    if not all_items:
        print(f"   ⚠️ No items to aggregate for subtask {subtask['id']}")
        return []

    # 2) Determine batch size to keep each payload ≲ ~2000 tokens
    #    Rough heuristic: assume 100 items ≈ 2000 tokens
    batch_size = 100
    num_batches = math.ceil(len(all_items) / batch_size)
    consolidated = []

    # 3) Process each batch
    for idx in range(num_batches):
        chunk = all_items[idx*batch_size : (idx+1)*batch_size]
        chunk_json = json.dumps(chunk, indent=2)

        user_prompt = f"""
                        You are finalizing the research goal:
                        "{plans_col.document(plan_id).get().to_dict().get('goal')}"

                        Here is batch {idx+1}/{num_batches} containing {len(chunk)} items:
                        {chunk_json}

                        Please deduplicate within this batch and return a JSON array of objects
                        with keys "name" and "address" (or other fields relevant to the goal).
                        Do NOT include duplicates. Output raw JSON only.
                    """

        with Spinner(f"[{subtask['id']}] Aggregating batch {idx+1}"):
            resp = openai.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system",  "content": "You consolidate scraped data into a list."},
                    {"role": "user",    "content": user_prompt}
                ],
                temperature=0.0
            ).choices[0].message.content

        try:
            partial = json.loads(resp)
        except json.JSONDecodeError:
            print(f"   ⚠️ Failed to parse batch {idx+1} response, skipping")
            partial = []

        consolidated.extend(partial)

    # 4) Global deduplication across batches
    seen, final_items = set(), []
    for item in consolidated:
        key = json.dumps(item, sort_keys=True)
        if key not in seen:
            seen.add(key)
            final_items.append(item)

    # 5) Persist results
    plans_col \
      .document(plan_id) \
      .collection("results") \
      .document(subtask["id"]) \
      .set({"results": final_items})

    print(f"\r   ✓ [aggregate] {len(final_items)} total items saved{' ' * 10}")
    return final_items
