#!/usr/bin/env python3
"""
Example: Using Column Context for Better Query Expansion

This example shows how to leverage the new column_specs parameter
to guide the query expansion decomposition toward covering specific data fields.
"""

import asyncio
from dotenv import find_dotenv, load_dotenv
from query_expansion_matrix import QueryExpansionMatrix


async def example_hospital_search():
    """Example 1: Hospital search with column guidance"""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Hospital Search with Column Guidance")
    print("=" * 70)
    
    qem = QueryExpansionMatrix()
    
    # Define what fields we want to extract
    hospital_columns = [
        {"name": "Name", "description": "Hospital name or organization"},
        {"name": "Address", "description": "Street address"},
        {"name": "City", "description": "City or municipality"},
        {"name": "Phone", "description": "Contact phone number"},
        {"name": "Bed Count", "description": "Number of hospital beds"},
        {"name": "Specializations", "description": "Medical specialties offered"}
    ]
    
    # The decomposition will now consider: "What axes of variation would help us
    # find sources containing or informing these 6 fields?"
    result = await qem.execute(
        goal="Find hospitals in Ontario",
        strategy="corner_only",
        column_specs=hospital_columns
    )
    
    print("\n✅ Generated queries optimized for these fields:")
    for col in hospital_columns:
        print(f"  • {col['name']}: {col['description']}")


async def example_software_library_search():
    """Example 2: Software library search with column guidance"""
    print("\n\n" + "=" * 70)
    print("EXAMPLE 2: Software Library Search with Column Guidance")
    print("=" * 70)
    
    qem = QueryExpansionMatrix()
    
    # Define what fields we want for each library
    library_columns = [
        {"name": "Name", "description": "Library or package name"},
        {"name": "GitHub URL", "description": "Repository link"},
        {"name": "Latest Version", "description": "Current stable version"},
        {"name": "Documentation URL", "description": "Link to documentation"},
        {"name": "Use Case", "description": "Primary purpose (NLP, CV, etc)"},
        {"name": "Installation Method", "description": "How to install (pip, conda, etc)"}
    ]
    
    result = await qem.execute(
        goal="Find popular Python deep learning libraries",
        strategy="corner_only",
        column_specs=library_columns
    )
    
    print("\n✅ Generated queries optimized for these fields:")
    for col in library_columns:
        print(f"  • {col['name']}: {col['description']}")


async def example_no_columns():
    """Example 3: Generic search without specific field guidance"""
    print("\n\n" + "=" * 70)
    print("EXAMPLE 3: Generic Search (No Column Guidance)")
    print("=" * 70)
    
    qem = QueryExpansionMatrix()
    
    # When you don't specify columns, the decomposition uses the generic prompt
    # that focuses on general coverage of the topic
    result = await qem.execute(
        goal="Find climate change research institutions",
        strategy="corner_only"
        # No column_specs - uses generic decomposition
    )
    
    print("\n✅ Generated queries for broad topic coverage")


async def main():
    """Run all examples"""
    load_dotenv(find_dotenv())
    
    print("\n" + "█" * 70)
    print("QUERY EXPANSION MATRIX - COLUMN CONTEXT EXAMPLES")
    print("█" * 70)
    
    try:
        await example_hospital_search()
        await example_software_library_search()
        await example_no_columns()
        
        print("\n\n" + "=" * 70)
        print("KEY TAKEAWAYS")
        print("=" * 70)
        print("""
1. COLUMN GUIDANCE IMPROVES COVERAGE
   - When you specify columns, the LLM identifies axes that help find those fields
   - Generates more targeted queries for your specific data needs

2. BACKWARDS COMPATIBLE
   - Old code without column_specs still works perfectly
   - column_specs is completely optional

3. WORKS ACROSS DOMAINS
   - Hospitals, software libraries, research institutions, etc.
   - Same flexible decomposition approach

4. USE CASES
   ✓ You know exactly what fields you need to extract
   ✓ You want the system to focus on diversity that matters to you
   ✓ You're building a structured data scraper with known schema
        """)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("(This likely means OPENAI_API_KEY is not set in environment)")


if __name__ == "__main__":
    asyncio.run(main())
