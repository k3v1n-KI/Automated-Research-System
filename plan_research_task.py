import os
import json
from uuid import uuid4
from dotenv import find_dotenv, load_dotenv

# OpenAI client (v1 interface)
from openai import OpenAI

# Firebase Admin SDK
from firebase import db

# Load environment variables from .env file
dotenv_path  = find_dotenv()
if dotenv_path:
    load_dotenv(dotenv_path)


# ——— Initialize OpenAI ———
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-2025-04-14")

client = OpenAI(api_key=OPENAI_API_KEY)

# ——— Planner Function ———
def plan_research_task(task_description: str) -> dict:
    system_prompt = (
        "You are an expert research assistant.  Output a single JSON object (no markdown, no extra text) "
        "with this exact schema:\n"
        "{\n"
        '  "goal": string,\n'
        '  "plan_id": string,\n'
        '  "subtasks": [\n'
        "    {\n"
        '      "id": string,\n'
        '      "type": string,       # one of "search","validate","refine","scrape","aggregate"\n'
        '      "description": string,\n'
        '      "params": object\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "PARAM SCHEMA (params must match exactly):\n"
        "  • search   → {\"query_variations\": [string], \"limit\": integer}\n"
        "  • validate → {\"source_subtasks\": [string], \"threshold\": number}\n"
        "  • refine   → {\"source_subtasks\": [string], \"max_new_queries\": integer}\n"
        "  • scrape   → {\"url\": string, \"source_subtasks\": [string]}\n"
        "  • aggregate→ {\"source_subtasks\": [string], \"output_format\": string}\n"
        "No other keys are allowed in params."
    )

    user_prompt = (
        f"The research goal is: \"{task_description}\".\n"
        "1. Generate **2–4 diverse search query variations** for broad coverage.\n"
        "2. Create a **search** subtask using those query_variations and a limit of 25.\n"
        "3. Create a **validate** subtask that cross-references the search subtask results with a **threshold of 0.6** (semantic similarity scale 0–1).\n"
        "3.5 Create a **refine** subtask that proposes **additional queries to cover gaps** (missed cities/regions, underrepresented domains) based on the validate results.\n"
        "    Use params: {\"source_subtasks\": [<validate subtask id>], \"max_new_queries\": 8}.\n"
        "4. For **each URL** returned by the validate subtask, create a **scrape** subtask:\n"
        "   - Fetch/clean the page using paragraph-based chunking (~3000 chars), then extract items relevant to the goal.\n"
        "   - Use params: {\"url\": <URL>, \"source_subtasks\": [<validate subtask id>]}\n"
        "5. Add an **aggregate** subtask that combines all scrape outputs into your desired output_format (e.g., JSON list of {name,address}).\n"
        "\n"
        "When extracting or describing, refer to the goal generically (e.g. “extract items relevant to the research goal”)."
    )

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user",   "content": user_prompt}],
        temperature=0.3,
    )

    plan = json.loads(resp.choices[0].message.content)
    plan["plan_id"] = str(uuid4())
    # ✅ Guarantee a usable goal
    raw_goal = (plan.get("goal") or "").strip()
    if not raw_goal:
        plan["goal"] = task_description.strip()
    else:
        plan["goal"] = raw_goal
    return plan

def save_plan_to_firebase(plan: dict):
    plan_id = plan.get("plan_id") or str(uuid4())
    db.collection("research_plans").document(plan_id).set({**plan, "dispatched": False})
    print(f"Saved plan {plan_id} to Firestore.")
