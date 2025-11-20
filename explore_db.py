#!/usr/bin/env python3
"""
Explore Firestore collections and find the latest run
"""
from firebase import db

def explore_collections():
    """Explore what's in the database"""
    
    # Check runs collection
    print("=== RUNS ===")
    runs_ref = db.collection('runs').order_by('ts', direction='DESCENDING').limit(10).stream()
    
    for i, run_doc in enumerate(runs_ref, 1):
        run_data = run_doc.to_dict()
        print(f"\nRun {i}:")
        print(f"  ID: {run_doc.id}")
        print(f"  Goal: {run_data.get('goal', 'N/A')}")
        print(f"  Plan ID: {run_data.get('plan_id', 'N/A')}")
        print(f"  TS: {run_data.get('ts', 'N/A')}")
        print(f"  Status: {run_data.get('status', 'N/A')}")
        
        # Check if this run has queries
        if 'queries' in run_data:
            print(f"  Queries: {len(run_data['queries'])}")
        if 'final_queries' in run_data:
            print(f"  Final Queries: {len(run_data['final_queries'])}")
    
    # Check items collection
    print("\n\n=== ITEMS (count by plan_id) ===")
    items_ref = db.collection('items').stream()
    
    plan_counts = {}
    for item_doc in items_ref:
        item_data = item_doc.to_dict()
        plan_id = item_data.get('plan_id', 'unknown')
        plan_counts[plan_id] = plan_counts.get(plan_id, 0) + 1
    
    for plan_id, count in sorted(plan_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {plan_id}: {count} items")
    
    # Check rounds collection
    print("\n\n=== ROUNDS (latest 5) ===")
    rounds_ref = db.collection('rounds').order_by('ts', direction='DESCENDING').limit(5).stream()
    
    for i, round_doc in enumerate(rounds_ref, 1):
        round_data = round_doc.to_dict()
        print(f"\nRound {i}:")
        print(f"  ID: {round_doc.id}")
        print(f"  Plan ID: {round_data.get('plan_id', 'N/A')}")
        print(f"  TS: {round_data.get('ts', 'N/A')}")

if __name__ == '__main__':
    explore_collections()
