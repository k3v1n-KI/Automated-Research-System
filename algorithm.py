"""
Research Algorithm - LangGraph based dataset generation pipeline.
6-node architecture: Query Generation -> Search -> Validate -> Scrape -> Extract -> Deduplicate
No Firebase, uses in-memory state management.
"""

import json
from datetime import datetime
from typing import Dict, Optional, Callable, TypedDict, List

from langgraph.graph import StateGraph, START, END
from dotenv import find_dotenv, load_dotenv
import os

# Import node implementations
from nodes.query_generation import QueryGenerationNode
from nodes.search import SearchNode
from nodes.validate import ValidateNode
from nodes.scrape import ScrapeNode
from nodes.extract import ExtractNode
from nodes.deduplicate import DeduplicateNode

# Load environment
dotenv_path = find_dotenv()
if dotenv_path:
    load_dotenv(dotenv_path)


# ============================================================================
# State Definition
# ============================================================================

class ResearchState(TypedDict):
    """Algorithm state - tracks progress through pipeline"""
    initial_prompt: str
    column_specs: List[str]
    queries: List[str]
    search_results: List[Dict]
    validated_urls: List[str]
    scraped_content: List[Dict]
    extracted_items: List[Dict]
    final_dataset: List[Dict]
    session_id: str
    round: int
    error: Optional[str]
    previous_session_id: Optional[str]
    tweak_instructions: Optional[str]
    previous_queries: List[str]
    previous_items: List[Dict]
    columns: List[Dict]
    priority_columns: List[str]


# ============================================================================
# Logging & Progress Tracking
# ============================================================================

class ProgressTracker:
    """Tracks progress and sends updates to frontend"""
    
    def __init__(self, emit_fn: Optional[Callable] = None):
        self.emit_fn = emit_fn
        self.current_step = ""
        self.current_detail = ""
    
    def update(self, step: str, detail: str = "", data: Optional[Dict] = None):
        """Update progress"""
        self.current_step = step
        self.current_detail = detail
        
        # Log to terminal
        print(f"\n{'='*70}")
        print(f"📍 {step}")
        print(f"   {detail}")
        if data:
            print(f"   Data: {json.dumps(data, default=str)[:200]}")
        print(f"{'='*70}")
        
        # Emit to frontend
        if self.emit_fn:
            try:
                self.emit_fn("progress", {
                    "step": step,
                    "detail": detail,
                    "data": data,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                print(f"❌ Failed to emit progress: {e}")



# ============================================================================
# Algorithm Builder
# ============================================================================

def build_research_algorithm(emit_fn: Optional[Callable] = None):
    """Build the complete research algorithm LangGraph"""
    
    progress = ProgressTracker(emit_fn)
    
    # Initialize node instances
    query_gen_node = QueryGenerationNode()
    search_node = SearchNode()
    validate_node = ValidateNode(threshold=0.5)
    scrape_node = ScrapeNode(timeout_ms=15000)
    extract_node = ExtractNode(char_limit=3000)
    deduplicate_node = DeduplicateNode()
    
    # Create graph
    graph = StateGraph(ResearchState)
    
    # Wrap nodes to inject progress tracker
    def make_node(node_instance):
        async def wrapped(state):
            return await node_instance.execute(state, progress)
        return wrapped
    
    # Add nodes
    graph.add_node("query_generation", make_node(query_gen_node))
    graph.add_node("search", make_node(search_node))
    graph.add_node("validate", make_node(validate_node))
    graph.add_node("scrape", make_node(scrape_node))
    graph.add_node("extract", make_node(extract_node))
    graph.add_node("deduplicate", make_node(deduplicate_node))
    
    # Add edges
    graph.add_edge(START, "query_generation")
    graph.add_edge("query_generation", "search")
    graph.add_edge("search", "validate")
    graph.add_edge("validate", "scrape")
    graph.add_edge("scrape", "extract")
    graph.add_edge("extract", "deduplicate")
    graph.add_edge("deduplicate", END)
    
    return graph.compile()


if __name__ == "__main__":
    # Test
    print("Algorithm module loaded. Use build_research_algorithm() to create graph.")
