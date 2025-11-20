#!/usr/bin/env python3
"""
Enhanced documentation generator that captures:
- All LLM prompts and outputs from DSPy dumps
- Terminal logs for search/validate/scrape
- Deduplication details at every stage
- Sample input/output for every node
"""
import json
import glob
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from firebase import db

def parse_terminal_logs(log_file='run_output.log'):
    """Parse terminal logs to extract search, validate, and scrape data"""
    if not Path(log_file).exists():
        return {}
    
    with open(log_file) as f:
        content = f.read()
    
    data = {
        'search': [],
        'validate': {},
        'scrape': {},
        'extract': {},
        'dedup': {}
    }
    
    # Parse search hits
    search_pattern = r'\[INFO\] \[search\] SearXNG hits for \'([^\']+)\': (\d+)'
    for match in re.finditer(search_pattern, content):
        data['search'].append({
            'query': match.group(1),
            'hits': int(match.group(2))
        })
    
    # Parse dedup stats
    dedup_pattern = r'\[INFO\] \[search\] Total raw results before dedupe: (\d+)'
    match = re.search(dedup_pattern, content)
    if match:
        data['dedup']['search_before'] = int(match.group(1))
    
    dedup_pattern2 = r'\[INFO\] \[search\] Results after dedupe: (\d+)'
    match = re.search(dedup_pattern2, content)
    if match:
        data['dedup']['search_after'] = int(match.group(1))
    
    # Parse validate stats
    validate_pattern = r'\[INFO\] \[validate\] kept (\d+)/(\d+) \(top_k=(\d+), thr=([\d.]+)\) \| rejected=(\d+)'
    match = re.search(validate_pattern, content)
    if match:
        data['validate'] = {
            'kept': int(match.group(1)),
            'total': int(match.group(2)),
            'top_k': int(match.group(3)),
            'threshold': float(match.group(4)),
            'rejected': int(match.group(5))
        }
    
    # Parse scrape stats
    scrape_pattern = r'\[INFO\] \[scrape\] url_scraper\.scrape_many returned (\d+) rows \((\d+) with text\)'
    match = re.search(scrape_pattern, content)
    if match:
        data['scrape'] = {
            'total': int(match.group(1)),
            'with_text': int(match.group(2))
        }
    
    # Parse extract stats
    extract_before = r'\[EXTRACT\]\[aggregate\] collected items before dedupe: (\d+)'
    match = re.search(extract_before, content)
    if match:
        data['extract']['before_dedupe'] = int(match.group(1))
    
    extract_after = r'\[EXTRACT\]\[dedupe\] deduped (\d+) -> (\d+)'
    match = re.search(extract_after, content)
    if match:
        data['extract']['after_dedupe'] = int(match.group(2))
    
    extract_final = r'\[EXTRACT\]\[aggregate\] final extracted items: (\d+)'
    match = re.search(extract_final, content)
    if match:
        data['extract']['final'] = int(match.group(1))
    
    return data

def get_dspy_samples(limit=10):
    """Get sample DSPy prompts and responses from /tmp/extract_dspy"""
    dumps_dir = Path('/tmp/extract_dspy')
    if not dumps_dir.exists():
        return []
    
    samples = []
    files = sorted(dumps_dir.glob('*.json'), key=lambda f: f.stat().st_mtime, reverse=True)
    
    for f in files[:limit]:
        try:
            with open(f) as fp:
                data = json.load(fp)
                
            # Parse the stored response
            if '_store' in data:
                store = eval(data['_store'])
                samples.append({
                    'file': f.name,
                    'items': store.get('items', '[]'),
                    'size': f.stat().st_size
                })
        except Exception as e:
            print(f"Warning: Could not parse {f}: {e}")
    
    return samples

def generate_comprehensive_doc(plan_id: str, log_data: Dict[str, Any], dspy_samples: List[Dict]):
    """Generate comprehensive documentation"""
    
    # Get plan and run data from Firestore
    plan_ref = db.collection('research_plans').document(plan_id)
    plan_doc = plan_ref.get()
    
    if not plan_doc.exists:
        print(f"❌ Plan {plan_id} not found")
        return None
    
    plan_data = plan_doc.to_dict()
    
    # Get runs
    runs_ref = plan_ref.collection('runs').stream()
    runs = []
    for run_doc in runs_ref:
        run_data = run_doc.to_dict()
        run_data['run_id'] = run_doc.id
        runs.append(run_data)
    
    if not runs:
        print("❌ No runs found")
        return None
    
    latest_run = max(runs, key=lambda r: r.get('ts', ''))
    run_id = latest_run['run_id']
    
    # Get rounds and aggregated items
    rounds_ref = plan_ref.collection('runs').document(run_id).collection('rounds').stream()
    total_items = 0
    sample_items = []
    
    for round_doc in rounds_ref:
        round_id = round_doc.id
        aggregate_ref = plan_ref.collection('runs').document(run_id).collection('rounds').document(round_id).collection('aggregate').stream()
        
        for agg_doc in aggregate_ref:
            agg_data = agg_doc.to_dict()
            items = agg_data.get('items', [])
            total_items += len(items)
            if not sample_items:
                sample_items = items[:10]
    
    # Build documentation
    doc = []
    doc.append("# Automated Research System - Comprehensive Technical Analysis")
    doc.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    doc.append(f"**Plan ID**: `{plan_id}`")
    doc.append(f"**Run ID**: `{run_id}`")
    doc.append("")
    
    # Overview
    doc.append("## 1. Run Overview")
    doc.append(f"- **Goal**: {plan_data.get('goal', 'N/A')}")
    doc.append(f"- **Status**: {latest_run.get('status', 'unknown')}")
    doc.append(f"- **Total items collected**: {total_items}")
    doc.append(f"- **Timestamp**: {latest_run.get('ts', 'N/A')}")
    doc.append("")
    
    # Queries
    doc.append("### 1.1 LLM-Generated Queries")
    doc.append("")
    queries = plan_data.get('queries', [])
    doc.append(f"The planning LLM generated **{len(queries)} queries** for comprehensive coverage:")
    doc.append("")
    for i, q in enumerate(queries, 1):
        doc.append(f"{i}. `{q}`")
    doc.append("")
    
    # Node documentation with real data
    doc.append("## 2. Node-by-Node Execution with Samples")
    doc.append("")
    
    # Search Node
    doc.append("### 2.1 Search Node")
    doc.append("")
    doc.append("**Purpose**: Execute queries across search engines and collect raw results")
    doc.append("")
    if log_data.get('search'):
        doc.append("**Execution Results**:")
        doc.append("")
        for sq in log_data['search']:
            doc.append(f"- Query: `{sq['query']}`")
            doc.append(f"  - Hits: {sq['hits']}")
        doc.append("")
    
    if log_data.get('dedup'):
        dedup = log_data['dedup']
        if 'search_before' in dedup and 'search_after' in dedup:
            doc.append("**Deduplication**:")
            doc.append(f"- Raw results: {dedup['search_before']}")
            doc.append(f"- After URL dedup: {dedup['search_after']}")
            doc.append(f"- Dedup rate: {((dedup['search_before'] - dedup['search_after']) / dedup['search_before'] * 100):.1f}%")
            doc.append("")
    
    # Validate Node
    doc.append("### 2.2 Validate Node")
    doc.append("")
    doc.append("**Purpose**: Semantic filtering of search results using embeddings")
    doc.append("")
    if log_data.get('validate'):
        val = log_data['validate']
        doc.append("**Execution Results**:")
        doc.append(f"- Input URLs: {val['total']}")
        doc.append(f"- Kept (passed threshold): {val['kept']}")
        doc.append(f"- Rejected: {val['rejected']}")
        doc.append(f"- Threshold: {val['threshold']}")
        doc.append(f"- Success rate: {(val['kept'] / val['total'] * 100):.1f}%")
        doc.append("")
    
    # Scrape Node
    doc.append("### 2.3 Scrape Node")
    doc.append("")
    doc.append("**Purpose**: Retrieve and clean content from validated URLs")
    doc.append("")
    if log_data.get('scrape'):
        scrape = log_data['scrape']
        doc.append("**Execution Results**:")
        doc.append(f"- Total URLs attempted: {scrape['total']}")
        doc.append(f"- Successful (with text): {scrape['with_text']}")
        doc.append(f"- Success rate: {(scrape['with_text'] / scrape['total'] * 100):.1f}%")
        doc.append("")
    
    # Extract Node
    doc.append("### 2.4 Extract Node")
    doc.append("")
    doc.append("**Purpose**: LLM-based entity extraction with regex fallback")
    doc.append("")
    doc.append("**Parameters**:")
    doc.append("```python")
    doc.append("controls = {")
    doc.append("    'chunk_size': 3000,")
    doc.append("    'overlap': 250,")
    doc.append("    'min_text_len': 100,")
    doc.append("    'max_items_per_chunk': 40,")
    doc.append("    'max_items_regex_fallback': 20")
    doc.append("}")
    doc.append("```")
    doc.append("")
    
    if log_data.get('extract'):
        ext = log_data['extract']
        doc.append("**Execution Results**:")
        if 'before_dedupe' in ext:
            doc.append(f"- Items before deduplication: {ext['before_dedupe']}")
        if 'after_dedupe' in ext:
            doc.append(f"- Items after deduplication: {ext['after_dedupe']}")
        if 'final' in ext:
            doc.append(f"- Final extracted items: {ext['final']}")
        if 'before_dedupe' in ext and 'after_dedupe' in ext:
            dedup_count = ext['before_dedupe'] - ext['after_dedupe']
            doc.append(f"- Duplicates removed: {dedup_count} ({(dedup_count / ext['before_dedupe'] * 100):.1f}%)")
        doc.append("")
    
    # DSPy Samples
    if dspy_samples:
        doc.append("**Sample LLM Extractions**:")
        doc.append("")
        for i, sample in enumerate(dspy_samples[:3], 1):
            doc.append(f"**Sample {i}** (from `{sample['file']}`):")
            doc.append("```json")
            try:
                items = json.loads(sample['items'])
                doc.append(json.dumps(items[:3], indent=2))
            except:
                doc.append(sample['items'][:500])
            doc.append("```")
            doc.append("")
    
    # Aggregate Node
    doc.append("### 2.5 Aggregate Node")
    doc.append("")
    doc.append("**Purpose**: Combine and deduplicate extracted items across all sources")
    doc.append("")
    doc.append(f"**Final Results**: {total_items} unique hospitals")
    doc.append("")
    
    if sample_items:
        doc.append("**Sample Results** (first 10):")
        doc.append("```json")
        doc.append(json.dumps(sample_items, indent=2))
        doc.append("```")
        doc.append("")
    
    # Summary
    doc.append("## 3. Pipeline Summary")
    doc.append("")
    doc.append("### 3.1 Complete Data Flow")
    doc.append("")
    
    if log_data.get('dedup') and log_data.get('validate') and log_data.get('scrape') and log_data.get('extract'):
        doc.append("```")
        doc.append(f"Search: {log_data['dedup'].get('search_before', '?')} raw results")
        doc.append(f"  ↓ URL deduplication")
        doc.append(f"Unique URLs: {log_data['dedup'].get('search_after', '?')}")
        doc.append(f"  ↓ Semantic validation (threshold {log_data['validate'].get('threshold', '?')})")
        doc.append(f"Validated: {log_data['validate'].get('kept', '?')} URLs")
        doc.append(f"  ↓ Web scraping")
        doc.append(f"Scraped: {log_data['scrape'].get('with_text', '?')} pages")
        doc.append(f"  ↓ LLM extraction + regex fallback")
        doc.append(f"Extracted: {log_data['extract'].get('before_dedupe', '?')} items")
        doc.append(f"  ↓ Item deduplication")
        doc.append(f"Final: {total_items} unique items")
        doc.append("```")
        doc.append("")
    
    # Query samples
    doc.append("### 3.2 All Queries Generated")
    doc.append("")
    doc.append("```json")
    doc.append(json.dumps(queries, indent=2))
    doc.append("```")
    doc.append("")
    
    return '\n'.join(doc)

def main():
    if len(sys.argv) < 2:
        print("Usage: python enhanced_documenter.py <plan_id>")
        sys.exit(1)
    
    plan_id = sys.argv[1]
    
    print("🔍 Collecting data...")
    print(f"   Plan ID: {plan_id}")
    
    # Parse logs
    print("   📋 Parsing terminal logs...")
    log_data = parse_terminal_logs()
    
    # Get DSPy samples
    print("   🤖 Collecting DSPy samples...")
    dspy_samples = get_dspy_samples()
    print(f"      Found {len(dspy_samples)} DSPy dumps")
    
    # Generate documentation
    print("   📝 Generating documentation...")
    doc = generate_comprehensive_doc(plan_id, log_data, dspy_samples)
    
    if doc:
        output_file = 'algorithm_comprehensive.md'
        with open(output_file, 'w') as f:
            f.write(doc)
        
        print()
        print(f"✅ Documentation complete!")
        print(f"   Output: {output_file}")
        print(f"   Size: {len(doc)} characters")
        
        # Update artifacts
        with open('last_run_artifacts.json', 'w') as f:
            json.dump({
                'plan_id': plan_id,
                'generated': datetime.now().isoformat(),
                'log_data': log_data,
                'dspy_samples_count': len(dspy_samples)
            }, f, indent=2)
        
        print(f"   Artifacts: last_run_artifacts.json")
    else:
        print("❌ Failed to generate documentation")
        sys.exit(1)

if __name__ == '__main__':
    main()
