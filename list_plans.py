#!/usr/bin/env python3
"""
List all plans in Firestore
"""
from firebase import db

def list_all_plans():
    """List all plans in Firestore"""
    plans_ref = db.collection('plans').stream()
    
    plans = []
    for plan_doc in plans_ref:
        plan_data = plan_doc.to_dict()
        plan_data['plan_id'] = plan_doc.id
        plans.append(plan_data)
    
    print(f"Found {len(plans)} plans:\n")
    for plan in sorted(plans, key=lambda p: p.get('ts', ''), reverse=True):
        print(f"Plan ID: {plan.get('plan_id')}")
        print(f"  Goal: {plan.get('goal', 'N/A')}")
        print(f"  TS: {plan.get('ts', 'N/A')}")
        print(f"  Queries: {len(plan.get('queries', []))}")
        print()

if __name__ == '__main__':
    list_all_plans()
