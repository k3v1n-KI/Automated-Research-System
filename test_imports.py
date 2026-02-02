#!/usr/bin/env python3
"""Quick test to verify all imports work correctly"""

try:
    from algorithm import build_research_algorithm, ResearchState, ProgressTracker
    from nodes import (
        QueryGenerationNode,
        SearchNode,
        ValidateNode,
        ScrapeNode,
        ExtractNode,
        DeduplicateNode,
    )
    print("✅ All imports successful!")
    print("✅ ResearchState imported")
    print("✅ ProgressTracker imported")
    print("✅ build_research_algorithm imported")
    print("✅ All 6 node classes imported")
    print("\n✨ Modular structure is working correctly!")
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
