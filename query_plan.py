#!/usr/bin/env python3
"""
Query Firestore for plan information and update last_run_artifacts.json
"""
import json
from firebase import db
from datetime import datetime

def query_plan_info(plan_id):
    """Query Firestore for all information related to a plan_id"""
    print(f"Querying plan: {plan_id}")
    
    # Get the plan document from research_plans collection
    plan_ref = db.collection('research_plans').document(plan_id)
    plan_doc = plan_ref.get()
    
    if not plan_doc.exists:
        print(f"Plan {plan_id} not found in research_plans")
        return None
    
    plan_data = plan_doc.to_dict()
    print(f"Plan data: {json.dumps(plan_data, indent=2, default=str)}")
    
    # Get all runs for this plan
    runs_ref = plan_ref.collection('runs').stream()
    runs = []
    for run_doc in runs_ref:
        run_data = run_doc.to_dict()
        run_data['run_id'] = run_doc.id
        runs.append(run_data)
    
    print(f"\nFound {len(runs)} runs")
    
    # Get the latest run
    if not runs:
        return {
            'plan': plan_data,
            'runs': [],
            'total_items': 0,
            'aggregated_items': []
        }
    
    latest_run = max(runs, key=lambda r: r.get('ts', ''))
    run_id = latest_run['run_id']
    print(f"Latest run ID: {run_id}")
    print(f"  Status: {latest_run.get('status', 'unknown')}")
    print(f"  TS: {latest_run.get('ts', 'N/A')}")
    
    # Get all rounds for the latest run
    rounds_ref = plan_ref.collection('runs').document(run_id).collection('rounds').stream()
    rounds = []
    for round_doc in rounds_ref:
        round_data = round_doc.to_dict()
        round_data['round_id'] = round_doc.id
        rounds.append(round_data)
    
    print(f"\nFound {len(rounds)} rounds")
    
    # Get aggregated items from all rounds
    all_items = []
    for round_data in rounds:
        round_id = round_data['round_id']
        aggregate_ref = plan_ref.collection('runs').document(run_id).collection('rounds').document(round_id).collection('aggregate').stream()
        
        for agg_doc in aggregate_ref:
            agg_data = agg_doc.to_dict()
            items = agg_data.get('items', [])
            print(f"  Round {round_id}, Doc {agg_doc.id}: {len(items)} items")
            all_items.extend(items)
    
    print(f"\nTotal items across all rounds: {len(all_items)}")
    
    return {
        'plan': plan_data,
        'runs': runs,
        'latest_run': latest_run,
        'rounds': rounds,
        'total_items': len(all_items),
        'aggregated_items': all_items
    }

def build_artifacts(plan_info):
    """Build last_run_artifacts.json structure from plan info"""
    plan = plan_info['plan']
    latest_run = plan_info.get('latest_run', {})
    
    # Extract queries from the plan
    queries = plan.get('queries', [])
    
    artifacts = {
        'plan_id': plan.get('plan_id', ''),
        'run_id': latest_run.get('run_id', ''),
        'goal': plan.get('goal', ''),
        'ts': str(latest_run.get('ts', '')),
        'status': latest_run.get('status', 'unknown'),
        'queries': queries,
        'queries_count': len(queries),
        'total_items': plan_info['total_items'],
        'sample_items': plan_info['aggregated_items'][:10] if plan_info['aggregated_items'] else []
    }
    
    return artifacts

if __name__ == '__main__':
    import sys
    
    plan_id = '1d6a0e53-cf1f-42db-876c-d902e6191a33'
    if len(sys.argv) > 1:
        plan_id = sys.argv[1]
    
    plan_info = query_plan_info(plan_id)
    
    if plan_info:
        artifacts = build_artifacts(plan_info)
        
        # Write to last_run_artifacts.json
        output_file = 'last_run_artifacts.json'
        with open(output_file, 'w') as f:
            json.dump(artifacts, f, indent=2, default=str)
        
        print(f"\n✓ Updated {output_file}")
        print(f"  Plan: {artifacts['plan_id']}")
        print(f"  Run: {artifacts['run_id']}")
        print(f"  Goal: {artifacts['goal']}")
        print(f"  Queries: {artifacts['queries_count']}")
        print(f"  Total Items: {artifacts['total_items']}")
        print(f"  Status: {artifacts['status']}")
