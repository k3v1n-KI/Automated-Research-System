from task_dispatcher import TaskDispatcher

"""
Running the SearXNG Container

cd searxng-docker/

sudo docker compose up -d

"""

if __name__ == "__main__":
    # Use the exact prompt requested by the user
    task = "Give me a list of hospitals in Ontario"
    # Allow multiple refinement rounds so critics can produce new queries.
    dispatcher = TaskDispatcher(max_rounds=3)  # run up to 3 rounds
    state = dispatcher.run(task)

    items = len((state.get("aggregated") or {}).get("items", []))
    print(f"Done. items={items} | plan_id={state.get('plan_id')} | queries_now={len(state.get('queries') or [])}")