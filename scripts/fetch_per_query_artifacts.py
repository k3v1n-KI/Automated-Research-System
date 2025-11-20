import sys
import os
# ensure project root is importable so local firebase.py is found
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from firebase import db

PLAN_ID = "0314181e-418e-46cf-8a87-9332d0c4d7c2"

root = db.collection("research_plans").document(PLAN_ID)
if not root:
    print('No firestore root')
    raise SystemExit(1)

runs_col = root.collection('runs')
print('Runs under plan:', PLAN_ID)
for run_doc in runs_col.stream():
    run_id = run_doc.id
    print('\nRun:', run_id)
    run_ref = runs_col.document(run_id)
    artifacts_col = run_ref.collection('artifacts')
    docs = list(artifacts_col.stream())
    if not docs:
        print('  No artifacts in run')
        continue
    print(f'  Artifacts ({len(docs)}):')
    for d in docs:
        doc_id = d.id
        if doc_id.startswith('search_per_query_') or doc_id.startswith('validate_per_query_') or doc_id.startswith('search_query_'):
            print('   -', doc_id)
            try:
                data = d.to_dict()
                print('     ', data)
            except Exception as e:
                print('     (error reading doc):', e)
    # Also list round subcollections for search/results if present
    rounds_col = run_ref.collection('rounds')
    for rdoc in rounds_col.stream():
        r_id = rdoc.id
        sub_search = rounds_col.document(r_id).collection('search')
        subdocs = list(sub_search.stream())
        if subdocs:
            print(f'  round {r_id} search chunks: {[sd.id for sd in subdocs]}')

print('\nDone')
