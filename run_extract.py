from task_dispatcher import TaskDispatcher

if __name__ == '__main__':
    task = "Find hospitals in Toronto."
    dispatcher = TaskDispatcher(max_rounds=2)
    state = dispatcher.run(task)

    items = len((state.get("aggregated") or {}).get("items", []))
    print(f"Done. items={items} | plan_id={state.get('plan_id')} | queries_now={len(state.get('queries') or [])}")
