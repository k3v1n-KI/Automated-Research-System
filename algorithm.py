"""
Research Algorithm - LangGraph based dataset generation pipeline.
6-node architecture: Query Generation -> Search -> Validate -> Scrape -> Extract -> Deduplicate
"""

import json
from datetime import datetime
from typing import Dict, Optional, Callable, TypedDict, List

from langgraph.graph import StateGraph, START, END
from dotenv import find_dotenv, load_dotenv
import os

# Import node implementations
from nodes.query_generation import QueryGenerationNode
from nodes.query_expansion_matrix import QueryExpansionMatrixNode
from nodes.search import SearchNode
from nodes.validate import ValidateNode
from nodes.Crawl4AI_scrape import Crawl4AIScrapeNode
from nodes.Crawl4AI_extract import Crawl4AIExtractNode
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
    queries: List[Dict | str]
    search_results: List[Dict]
    validated_urls: List[str]
    validated_results: List[Dict]
    scraped_content: List[Dict]
    extracted_items: List[Dict]
    final_dataset: List[Dict]
    session_id: str
    round: int
    error: Optional[str]
    previous_session_id: Optional[str]
    tweak_instructions: Optional[str]
    previous_queries: List[Dict | str]
    previous_items: List[Dict]
    columns: List[Dict]
    priority_columns: List[str]
    hard_identifier_columns: List[str]
    soft_identifier_columns: List[str]


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
    strategy = os.getenv("QUERY_GENERATION_STRATEGY", "qem").lower()
    if strategy in {"simple", "baseline", "basic"}:
        query_gen_node = QueryGenerationNode()
    else:
        query_gen_node = QueryExpansionMatrixNode()
    search_node = SearchNode()
    validate_node = ValidateNode(threshold=0.5)
    scrape_node = Crawl4AIScrapeNode(timeout_ms=15000)
    extract_node = Crawl4AIExtractNode(char_limit=12000)
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


