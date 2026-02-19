#!/usr/bin/env python3
"""
Test the improved decomposition prompt with column context
Demonstrates how providing column specs improves axis discovery
"""

import asyncio
from dotenv import find_dotenv, load_dotenv
from query_expansion_matrix import QueryExpansionMatrix

async def test_with_columns():
    """Test decomposition with column context"""
    load_dotenv(find_dotenv())
    
    print("=" * 70)
    print("TESTING DECOMPOSITION WITH COLUMN CONTEXT")
    print("=" * 70)
    print()
    
    qem = QueryExpansionMatrix()
    
    # Test 1: Without column context (baseline)
    print("TEST 1: WITHOUT COLUMN CONTEXT")
    print("-" * 70)
    goal = "Find hospitals in Ontario"
    result = await qem.execute(goal, strategy="corner_only")
    print()
    
    # Test 2: With column context (improved)
    print("\nTEST 2: WITH COLUMN CONTEXT")
    print("-" * 70)
    column_specs = [
        {"name": "Name", "description": "Hospital name or organization"},
        {"name": "Address", "description": "Street address"},
        {"name": "City", "description": "City or municipality"},
        {"name": "Phone", "description": "Contact phone number"},
        {"name": "Bed Count", "description": "Number of hospital beds"},
        {"name": "Specializations", "description": "Medical specialties offered"}
    ]
    
    result = await qem.execute(goal, strategy="corner_only", column_specs=column_specs)
    print()
    
    # Test 3: Different domain without columns
    print("\nTEST 3: DIFFERENT DOMAIN - Software Libraries (No Columns)")
    print("-" * 70)
    goal2 = "Find popular Python machine learning libraries"
    result2 = await qem.execute(goal2, strategy="corner_only")
    print()
    
    # Test 4: Different domain with columns
    print("\nTEST 4: DIFFERENT DOMAIN - Software Libraries (With Columns)")
    print("-" * 70)
    ml_columns = [
        {"name": "Name", "description": "Library or package name"},
        {"name": "GitHub URL", "description": "Repository link"},
        {"name": "Latest Version", "description": "Current stable version"},
        {"name": "Documentation", "description": "URL to documentation"},
        {"name": "Use Case", "description": "Primary intended use"},
        {"name": "Installation", "description": "How to install (pip, conda, etc)"}
    ]
    
    result3 = await qem.execute(goal2, strategy="corner_only", column_specs=ml_columns)
    
    print("\n" + "=" * 70)
    print("TESTS COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(test_with_columns())
