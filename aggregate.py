import os
import json

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
from terminal_aesthetics import Spinner
from firebase import db



# ——— Environment & Clients ———
dotenv_path = find_dotenv()
if dotenv_path:
    # Load environment variables from .env file
    load_dotenv(dotenv_path)



plans_col = db.collection("research_plans")

# OpenAI client
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-2025-04-14")


# ——— History Helpers ———
def _load_history(plan_id: str) -> list:
    doc = plans_col.document(plan_id).get().to_dict() or {}
    return doc.get("history", [])

def _append_history(plan_id: str, role: str, content: str):
    history = _load_history(plan_id)
    history.append({"role": role, "content": content})
    plans_col.document(plan_id).update({"history": history})

# ——— Main Aggregation Function ———
def execute_aggregate(plan_id: str, subtask: dict):
    """
    Consolidate all scrape outputs into a final deduped list via GPT-4.
    Params schema for subtask:
      { "source_subtasks": [<validate_subtask_id>, ...] }
    Saves JSON array of {'name', 'address'} items under:
      research_plans/{plan_id}/results/{subtask_id}
    """
    # 1) Gather all scraped items
    sources = subtask["params"]["source_subtasks"]
    coll    = plans_col.document(plan_id).collection("results")
    all_items = []
    for src in sources:
        data = coll.document(src).get().to_dict() or {}
        all_items.extend(data.get("results", []))

    if not all_items:
        print(f"   ⚠️ No scrape outputs for aggregate {subtask['id']}")
        return

    # 2) Build the GPT-4 prompt with full context & history
    plan_doc = plans_col.document(plan_id).get().to_dict()
    goal     = plan_doc["goal"]
    history  = _load_history(plan_id)
    items_json = json.dumps(all_items, indent=2)

    user_prompt = f"""
You are finalizing the research goal:
  "{goal}"

The following items were extracted from individual pages:
{items_json}

Please consolidate these into a final, deduplicated JSON array of objects
with keys "name" and "address" (or other fields as relevant to the goal).
Do NOT include duplicates. Output raw JSON only.
"""

    # 3) Record user turn in history
    _append_history(plan_id, "user", user_prompt)

    # 4) Call GPT-4 with a spinner
    with Spinner(f"[{subtask['id']}] Aggregating"):
        resp = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=history + [
                {"role": "system", "content": "You consolidate scraped data into a final list."},
                {"role": "user",   "content": user_prompt}
            ],
            temperature=0.0
        ).choices[0].message.content

    # 5) Record assistant turn in history
    _append_history(plan_id, "assistant", resp)

    # 6) Parse and persist
    final_items = json.loads(resp)
    plans_col \
      .document(plan_id) \
      .collection("results") \
      .document(subtask["id"]) \
      .set({"results": final_items})

    print(f"\r   ✓ [aggregate] {len(final_items)} items saved{' ' * 10}")
