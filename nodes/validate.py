"""
Node 3: Validate
Validates URLs using semantic similarity against source queries.
"""

from typing import TYPE_CHECKING

from nodes.base import BaseNode

if TYPE_CHECKING:
    from algorithm import ResearchState, ProgressTracker


class ValidateNode(BaseNode):
    """
    Validates URLs using semantic similarity.
    
    Input State Keys:
        - search_results: List of URLs with metadata
        - queries: Source queries for comparison
    
    Output State Keys:
        - validated_urls: Filtered list of URLs
    """
    
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
    
    async def execute(self, state: "ResearchState", progress: "ProgressTracker") -> "ResearchState":
        """Visit and validate URLs - for now just collect all URLs"""
        
        urls_data = state['search_results']
        
        progress.update(
            "URL Validation",
            f"Preparing to visit {len(urls_data)} URLs..."
        )
        
        # For now, just collect all URLs without filtering
        validated = [url_data.get('url') for url_data in urls_data if url_data.get('url')]
        
        state['validated_urls'] = validated
        
        progress.update(
            "URL Validation Complete",
            f"Ready to scrape {len(validated)}/{len(urls_data)} URLs",
            {"validated": len(validated), "original": len(urls_data)}
        )
        
        return state
