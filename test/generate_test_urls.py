#!/usr/bin/env python3
"""
Generate real URLs from the search node to test scraping
Runs the algorithm up to the validated URLs stage
"""

import asyncio
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from algorithm import ResearchState, build_research_algorithm


async def generate_test_urls():
    """Run algorithm to get real validated URLs for testing"""
    
    # Create a simple test state
    state: ResearchState = {
        "initial_prompt": "Find hospitals in Toronto",
        "column_specs": ["Name", "Address", "Phone", "Website"],
        "queries": [],
        "search_results": [],
        "validated_urls": [],
        "scraped_content": [],
        "extracted_items": [],
        "final_dataset": [],
        "session_id": "test_scrape_urls",
        "round": 0,
        "error": None,
        "previous_session_id": None,
        "tweak_instructions": None,
        "previous_queries": [],
        "previous_items": [],
        "columns": [
            {"name": "Name", "type": "string"},
            {"name": "Address", "type": "string"},
            {"name": "Phone", "type": "string"},
            {"name": "Website", "type": "url"}
        ]
    }
    
    print("🚀 Building research graph...")
    graph = build_research_algorithm()
    
    print("\n📋 Running algorithm up to validated URLs...")
    print(f"   Prompt: {state['initial_prompt']}")
    print(f"   Columns: {', '.join(s for s in state['column_specs'])}")
    
    try:
        # Run the graph asynchronously
        final_state = None
        async for output in graph.astream(state):
            for node_name, node_state in output.items():
                if node_name == "Query Generation":
                    print(f"\n✓ Node 1: Query Generation")
                    print(f"  Generated {len(node_state.get('queries', []))} queries")
                    if len(node_state.get('queries', [])) > 0:
                        print(f"  Sample: {node_state['queries'][0]}")
                
                elif node_name == "Search":
                    print(f"\n✓ Node 2: Search")
                    results = node_state.get('search_results', [])
                    print(f"  Found {len(results)} results")
                
                elif node_name == "Validate":
                    print(f"\n✓ Node 3: Validate URLs")
                    urls = node_state.get('validated_urls', [])
                    print(f"  Validated {len(urls)} URLs")
                    final_state = node_state
                    break
            
            if final_state:
                break
        
        if final_state:
            urls = final_state.get('validated_urls', [])
            if urls:
                print(f"\n✅ SUCCESS: Generated {len(urls)} validated URLs")
                
                # Save URLs to file for scrape testing
                output_file = Path(__file__).parent / "test_urls.json"
                with open(output_file, 'w') as f:
                    json.dump({
                        "timestamp": str(Path.__mro__),
                        "prompt": state['initial_prompt'],
                        "total_urls": len(urls),
                        "urls": urls
                    }, f, indent=2)
                
                print(f"\n📁 Saved to: {output_file}")
                print("\n📍 Sample URLs:")
                for i, url in enumerate(urls[:5], 1):
                    print(f"   {i}. {url}")
                
                return urls
            else:
                print("❌ No URLs generated")
                return []
        else:
            print("❌ Algorithm did not complete")
            return []
    
    except Exception as e:
        print(f"❌ Error running algorithm: {e}")
        import traceback
        traceback.print_exc()
        return []


if __name__ == "__main__":
    urls = asyncio.run(generate_test_urls())
    print(f"\n🎯 Ready to test {len(urls)} URLs for scraping failures")
