#!/usr/bin/env python3
"""
Comprehensive run documentation generator for algorithm_comprehensive.md
Extracts all data from Firestore and creates detailed documentation with samples.
"""
import json
import sys
from datetime import datetime
from typing import Dict, List, Any
from firebase import db
from collections import defaultdict

def get_plan_details(plan_id: str) -> Dict[str, Any]:
    """Get all plan details including queries and metadata"""
    plan_ref = db.collection('research_plans').document(plan_id)
    plan_doc = plan_ref.get()
    
    if not plan_doc.exists:
        print(f"❌ Plan {plan_id} not found")
        return None
    
    return plan_doc.to_dict()

def get_run_details(plan_id: str, run_id: str) -> Dict[str, Any]:
    """Get all run details"""
    run_ref = db.collection('research_plans').document(plan_id).collection('runs').document(run_id)
    run_doc = run_ref.get()
    
    if not run_doc.exists:
        return None
    
    return run_doc.to_dict()

def get_round_data(plan_id: str, run_id: str, round_id: str) -> Dict[str, Any]:
    """Get all data for a specific round"""
    round_path = f'research_plans/{plan_id}/runs/{run_id}/rounds/{round_id}'
    round_ref = db.collection('research_plans').document(plan_id).collection('runs').document(run_id).collection('rounds').document(round_id)
    round_doc = round_ref.get()
    
    data = {
        'metadata': round_doc.to_dict() if round_doc.exists else {},
        'search': [],
        'validate': {'kept': [], 'rejected': []},
        'scrape': [],
        'extract': [],
        'aggregate': []
    }
    
    # Get search results
    search_ref = round_ref.collection('search').stream()
    for doc in search_ref:
        data['search'].append(doc.to_dict())
    
    # Get validation results (kept and rejected)
    validate_ref = round_ref.collection('validate').stream()
    for doc in validate_ref:
        doc_data = doc.to_dict()
        if 'urls' in doc_data:
            data['validate']['kept'].extend(doc_data.get('urls', []))
        if 'rejected' in doc_data:
            data['validate']['rejected'].extend(doc_data.get('rejected', []))
    
    # Get scrape results
    scrape_ref = round_ref.collection('scrape').stream()
    for doc in scrape_ref:
        doc_data = doc.to_dict()
        if 'scrapes' in doc_data:
            data['scrape'].extend(doc_data.get('scrapes', []))
    
    # Get extract results
    extract_ref = round_ref.collection('extract').stream()
    for doc in extract_ref:
        doc_data = doc.to_dict()
        if 'items' in doc_data:
            data['extract'].extend(doc_data.get('items', []))
    
    # Get aggregated items
    aggregate_ref = round_ref.collection('aggregate').stream()
    for doc in aggregate_ref:
        doc_data = doc.to_dict()
        if 'items' in doc_data:
            data['aggregate'].extend(doc_data.get('items', []))
    
    return data

def get_artifacts(plan_id: str, run_id: str) -> Dict[str, Any]:
    """Get all artifacts for the run"""
    artifacts_ref = db.collection('research_plans').document(plan_id).collection('runs').document(run_id).collection('artifacts').stream()
    
    artifacts = {}
    for doc in artifacts_ref:
        artifacts[doc.id] = doc.to_dict()
    
    return artifacts

def analyze_deduplication(rounds_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze deduplication across rounds"""
    analysis = {
        'url_deduplication': {},
        'item_deduplication': {},
        'cross_round_stats': []
    }
    
    all_search_urls = set()
    all_validated_urls = set()
    all_scraped_urls = set()
    all_items = []
    
    for idx, round_data in enumerate(rounds_data):
        round_num = idx + 1
        
        # Track URLs at each stage
        search_urls = set()
        for search_result in round_data.get('search', []):
            for result in search_result.get('results', []):
                search_urls.add(result.get('url', ''))
        
        validated_urls = set([url for url in round_data.get('validate', {}).get('kept', [])])
        scraped_urls = set([s.get('url', '') for s in round_data.get('scrape', [])])
        
        # Calculate deduplication stats
        new_search = len(search_urls - all_search_urls)
        new_validated = len(validated_urls - all_validated_urls)
        new_scraped = len(scraped_urls - all_scraped_urls)
        
        analysis['cross_round_stats'].append({
            'round': round_num,
            'search': {
                'total': len(search_urls),
                'new': new_search,
                'duplicates': len(search_urls) - new_search
            },
            'validated': {
                'total': len(validated_urls),
                'new': new_validated,
                'duplicates': len(validated_urls) - new_validated
            },
            'scraped': {
                'total': len(scraped_urls),
                'new': new_scraped,
                'duplicates': len(scraped_urls) - new_scraped
            }
        })
        
        all_search_urls.update(search_urls)
        all_validated_urls.update(validated_urls)
        all_scraped_urls.update(scraped_urls)
        all_items.extend(round_data.get('aggregate', []))
    
    analysis['url_deduplication'] = {
        'total_search_urls': len(all_search_urls),
        'total_validated_urls': len(all_validated_urls),
        'total_scraped_urls': len(all_scraped_urls),
        'validation_filter_rate': 1 - (len(all_validated_urls) / len(all_search_urls)) if all_search_urls else 0,
        'scrape_success_rate': len(all_scraped_urls) / len(all_validated_urls) if all_validated_urls else 0
    }
    
    return analysis

def generate_documentation(plan_id: str) -> str:
    """Generate comprehensive documentation for a plan"""
    print(f"📝 Generating documentation for plan: {plan_id}")
    
    # Get plan details
    plan_data = get_plan_details(plan_id)
    if not plan_data:
        return None
    
    # Get all runs
    runs_ref = db.collection('research_plans').document(plan_id).collection('runs').stream()
    runs = []
    for run_doc in runs_ref:
        run_data = run_doc.to_dict()
        run_data['run_id'] = run_doc.id
        runs.append(run_data)
    
    if not runs:
        print("❌ No runs found")
        return None
    
    # Get latest run
    latest_run = max(runs, key=lambda r: r.get('ts', ''))
    run_id = latest_run['run_id']
    
    print(f"📊 Latest run: {run_id}")
    print(f"   Goal: {plan_data.get('goal', 'N/A')}")
    print(f"   Status: {latest_run.get('status', 'unknown')}")
    
    # Get all rounds
    rounds_ref = db.collection('research_plans').document(plan_id).collection('runs').document(run_id).collection('rounds').stream()
    rounds = []
    for round_doc in rounds_ref:
        round_id = round_doc.id
        print(f"   Fetching round {round_id}...")
        round_data = get_round_data(plan_id, run_id, round_id)
        round_data['round_id'] = round_id
        rounds.append(round_data)
    
    rounds.sort(key=lambda r: r['round_id'])
    
    # Get artifacts
    artifacts = get_artifacts(plan_id, run_id)
    
    # Analyze deduplication
    dedup_analysis = analyze_deduplication(rounds)
    
    # Build documentation
    doc = []
    doc.append("# Automated Research System - Comprehensive Technical Analysis")
    doc.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    doc.append(f"**Plan ID**: `{plan_id}`")
    doc.append(f"**Run ID**: `{run_id}`")
    doc.append(f"**Status**: {latest_run.get('status', 'unknown')}")
    doc.append("")
    
    # Overview
    doc.append("## 1. Run Overview")
    doc.append(f"- **Goal**: {plan_data.get('goal', 'N/A')}")
    doc.append(f"- **Total Rounds**: {len(rounds)}")
    doc.append(f"- **Start Time**: {latest_run.get('ts', 'N/A')}")
    doc.append(f"- **Queries**: {len(plan_data.get('queries', []))}")
    doc.append("")
    
    # Queries
    doc.append("### 1.1 Generated Queries")
    doc.append("")
    doc.append("The system generated the following queries for search:")
    doc.append("```json")
    doc.append(json.dumps(plan_data.get('queries', []), indent=2))
    doc.append("```")
    doc.append("")
    
    # Deduplication Analysis
    doc.append("## 2. Deduplication Analysis")
    doc.append("")
    doc.append("### 2.1 Overall Statistics")
    doc.append(f"- **Total unique search URLs**: {dedup_analysis['url_deduplication']['total_search_urls']}")
    doc.append(f"- **Total unique validated URLs**: {dedup_analysis['url_deduplication']['total_validated_urls']}")
    doc.append(f"- **Total unique scraped URLs**: {dedup_analysis['url_deduplication']['total_scraped_urls']}")
    doc.append(f"- **Validation filter rate**: {dedup_analysis['url_deduplication']['validation_filter_rate']:.1%}")
    doc.append(f"- **Scrape success rate**: {dedup_analysis['url_deduplication']['scrape_success_rate']:.1%}")
    doc.append("")
    
    doc.append("### 2.2 Per-Round Deduplication")
    doc.append("")
    for round_stats in dedup_analysis['cross_round_stats']:
        doc.append(f"**Round {round_stats['round']}:**")
        doc.append(f"- Search: {round_stats['search']['new']} new / {round_stats['search']['duplicates']} duplicates (total: {round_stats['search']['total']})")
        doc.append(f"- Validated: {round_stats['validated']['new']} new / {round_stats['validated']['duplicates']} duplicates (total: {round_stats['validated']['total']})")
        doc.append(f"- Scraped: {round_stats['scraped']['new']} new / {round_stats['scraped']['duplicates']} duplicates (total: {round_stats['scraped']['total']})")
        doc.append("")
    
    # Per-Round Details
    doc.append("## 3. Node-by-Node Execution Details")
    doc.append("")
    
    for round_data in rounds:
        round_num = round_data['round_id']
        doc.append(f"### Round {round_num}")
        doc.append("")
        
        # Search Node
        doc.append("#### 3.1 Search Node")
        doc.append("")
        search_results = round_data.get('search', [])
        if search_results:
            total_hits = sum(len(sr.get('results', [])) for sr in search_results)
            doc.append(f"- **Total raw search results**: {total_hits}")
            doc.append("")
            doc.append("**Sample search results** (first 3):")
            doc.append("```json")
            sample_results = []
            for sr in search_results[:1]:
                for result in sr.get('results', [])[:3]:
                    sample_results.append({
                        'url': result.get('url', ''),
                        'title': result.get('title', ''),
                        'snippet': result.get('snippet', '')[:100] + '...' if len(result.get('snippet', '')) > 100 else result.get('snippet', '')
                    })
            doc.append(json.dumps(sample_results, indent=2))
            doc.append("```")
        else:
            doc.append("- No search results found")
        doc.append("")
        
        # Validate Node
        doc.append("#### 3.2 Validate Node")
        doc.append("")
        validate_data = round_data.get('validate', {})
        kept_urls = validate_data.get('kept', [])
        rejected_urls = validate_data.get('rejected', [])
        
        doc.append(f"- **Accepted URLs**: {len(kept_urls)}")
        doc.append(f"- **Rejected URLs**: {len(rejected_urls)}")
        doc.append("")
        
        if kept_urls:
            doc.append("**Sample accepted URLs** (first 5):")
            doc.append("```")
            for url in kept_urls[:5]:
                doc.append(f"  - {url}")
            doc.append("```")
            doc.append("")
        
        if rejected_urls:
            doc.append("**Sample rejected URLs with scores** (first 5):")
            doc.append("```json")
            doc.append(json.dumps(rejected_urls[:5], indent=2))
            doc.append("```")
            doc.append("")
        
        # Scrape Node
        doc.append("#### 3.3 Scrape Node")
        doc.append("")
        scrape_data = round_data.get('scrape', [])
        
        if scrape_data:
            successful = sum(1 for s in scrape_data if s.get('clean_text') or s.get('md') or s.get('html'))
            doc.append(f"- **Total scrapes attempted**: {len(scrape_data)}")
            doc.append(f"- **Successful scrapes**: {successful}")
            doc.append(f"- **Success rate**: {successful/len(scrape_data):.1%}")
            doc.append("")
            
            doc.append("**Sample scraped content** (first 2):")
            doc.append("```json")
            sample_scrapes = []
            for scrape in scrape_data[:2]:
                text = scrape.get('clean_text', '') or scrape.get('md', '') or scrape.get('html', '')
                sample_scrapes.append({
                    'url': scrape.get('url', ''),
                    'content_length': len(text),
                    'content_preview': text[:200] + '...' if len(text) > 200 else text
                })
            doc.append(json.dumps(sample_scrapes, indent=2))
            doc.append("```")
        else:
            doc.append("- No scrape data found")
        doc.append("")
        
        # Extract Node
        doc.append("#### 3.4 Extract Node")
        doc.append("")
        extract_data = round_data.get('extract', [])
        
        if extract_data:
            doc.append(f"- **Total items extracted**: {len(extract_data)}")
            doc.append("")
            
            doc.append("**Sample extracted items** (first 5):")
            doc.append("```json")
            doc.append(json.dumps(extract_data[:5], indent=2))
            doc.append("```")
        else:
            doc.append("- No extracted items found")
        doc.append("")
        
        # Aggregate Node
        doc.append("#### 3.5 Aggregate Node")
        doc.append("")
        aggregate_data = round_data.get('aggregate', [])
        
        if aggregate_data:
            doc.append(f"- **Total aggregated items**: {len(aggregate_data)}")
            doc.append("")
            
            doc.append("**Sample aggregated items** (first 5):")
            doc.append("```json")
            doc.append(json.dumps(aggregate_data[:5], indent=2))
            doc.append("```")
        else:
            doc.append("- No aggregated items found")
        doc.append("")
    
    # Artifacts
    doc.append("## 4. LLM Prompts and Outputs")
    doc.append("")
    
    if artifacts:
        doc.append(f"**Total artifacts**: {len(artifacts)}")
        doc.append("")
        
        # Sample plan artifact
        plan_artifacts = [k for k in artifacts.keys() if 'plan' in k.lower()]
        if plan_artifacts:
            doc.append("### 4.1 Plan Node")
            doc.append("**Sample prompt and output**:")
            doc.append("```json")
            doc.append(json.dumps(artifacts[plan_artifacts[0]], indent=2, default=str))
            doc.append("```")
            doc.append("")
        
        # Sample extract artifact
        extract_artifacts = [k for k in artifacts.keys() if 'extract' in k.lower()]
        if extract_artifacts:
            doc.append("### 4.2 Extract Node")
            doc.append(f"**Total extract artifacts**: {len(extract_artifacts)}")
            doc.append("")
            doc.append("**Sample prompt and output**:")
            doc.append("```json")
            doc.append(json.dumps(artifacts[extract_artifacts[0]], indent=2, default=str))
            doc.append("```")
            doc.append("")
    
    # Summary
    doc.append("## 5. Final Summary")
    doc.append("")
    
    total_aggregated = sum(len(r.get('aggregate', [])) for r in rounds)
    doc.append(f"- **Total items across all rounds**: {total_aggregated}")
    doc.append(f"- **Average items per round**: {total_aggregated / len(rounds):.1f}")
    doc.append(f"- **Pipeline status**: {latest_run.get('status', 'unknown')}")
    doc.append("")
    
    return '\n'.join(doc)

def main():
    if len(sys.argv) < 2:
        print("Usage: python document_run.py <plan_id>")
        sys.exit(1)
    
    plan_id = sys.argv[1]
    
    print(f"🔍 Starting documentation generation...")
    print(f"   Plan ID: {plan_id}")
    print()
    
    documentation = generate_documentation(plan_id)
    
    if documentation:
        # Write to file
        output_file = 'algorithm_comprehensive.md'
        with open(output_file, 'w') as f:
            f.write(documentation)
        
        print()
        print(f"✅ Documentation generated successfully!")
        print(f"   Output: {output_file}")
        print(f"   Size: {len(documentation)} characters")
        
        # Also update last_run_artifacts.json
        plan_data = get_plan_details(plan_id)
        runs_ref = db.collection('research_plans').document(plan_id).collection('runs').stream()
        runs = []
        for run_doc in runs_ref:
            run_data = run_doc.to_dict()
            run_data['run_id'] = run_doc.id
            runs.append(run_data)
        
        latest_run = max(runs, key=lambda r: r.get('ts', ''))
        
        artifacts_data = {
            'plan_id': plan_id,
            'run_id': latest_run['run_id'],
            'goal': plan_data.get('goal', ''),
            'queries': plan_data.get('queries', []),
            'status': latest_run.get('status', 'unknown'),
            'ts': str(latest_run.get('ts', ''))
        }
        
        with open('last_run_artifacts.json', 'w') as f:
            json.dump(artifacts_data, f, indent=2, default=str)
        
        print(f"   Artifacts: last_run_artifacts.json")
    else:
        print("❌ Failed to generate documentation")
        sys.exit(1)

if __name__ == '__main__':
    main()
