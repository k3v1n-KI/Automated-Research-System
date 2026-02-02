# Automated Research System - High-Level Flowchart

## System Overview (Presentation)

```mermaid
flowchart TD
    Start([Start]) --> Input[/User Defines:<br/>• Research Goal<br/>• Data Columns<br/>• Priority Columns/]
    
    Input --> AutoResearch[Automated Research Pipeline<br/>━━━━━━━━━━━━━━━━━━━━<br/>1. Generate Search Queries<br/>2. Find & Validate URLs<br/>3. Scrape Web Content<br/>4. Extract Structured Data]
    
    AutoResearch --> Quality{Quality<br/>Threshold<br/>Met?}
    
    Quality -->|No| AutoResearch
    Quality -->|Yes| Diversity[Calculate Diversity Index<br/>Gini-Simpson Formula]
    
    Diversity --> Database[(Save Results<br/>to CSV)]
    
    Database --> Display[/Display Results:<br/>• Dataset Records<br/>• Diversity Metrics<br/>• Value Breakdown/]
    
    Display --> UserChoice{Next<br/>Action?}
    
    UserChoice -->|Continue Collecting| AutoResearch
    UserChoice -->|New Research| Input
    UserChoice -->|Complete| End([End])
    
    style Start fill:#90EE90,stroke:#333,stroke-width:3px
    style End fill:#FFB6C1,stroke:#333,stroke-width:3px
    style Quality fill:#FFD700,stroke:#333,stroke-width:2px
    style UserChoice fill:#FFD700,stroke:#333,stroke-width:2px
    style AutoResearch fill:#87CEEB,stroke:#333,stroke-width:2px
    style Diversity fill:#F0E68C,stroke:#333,stroke-width:2px
    style Database fill:#98FB98,stroke:#333,stroke-width:2px
    style Input fill:#E6E6FA,stroke:#333,stroke-width:2px
    style Display fill:#E6E6FA,stroke:#333,stroke-width:2px
```

## Key Components

### 🎯 User Input
- Define research goal and objectives
- Specify data columns to extract
- Mark priority columns for diversity tracking

### 🤖 Automated Research Pipeline
**4-Step Process:**
1. **Query Generation** - AI creates optimized search queries
2. **URL Discovery** - Finds and validates relevant web sources
3. **Content Scraping** - Extracts HTML from validated URLs
4. **Data Extraction** - Uses LLM to parse structured information

### 📊 Quality & Diversity Analysis
- **Quality Check** - Ensures threshold requirements are met
- **Diversity Index** - Gini-Simpson formula: `1 - Σ(n_i/N)²`
- **Value Breakdown** - Shows distribution across priority columns

### 💾 Results & Output
- Saves dataset to CSV format
- Displays diversity metrics and statistics
- Interactive loop for continued research

## Example Use Case

**Research Goal:** "Find hospitals in Toronto"

**User Input:**
- Columns: Name, Address, City, Website
- Priority Column: City (for diversity tracking)

**System Output:**
- 15 hospital records extracted
- Diversity Index: 0.42 (cities include Toronto, Vaughan, Mississauga)
- CSV file with complete dataset
- Visual breakdown of city distribution

---

## Technical Stack

- **Frontend:** HTML5, JavaScript, Socket.IO
- **Backend:** Python, Flask, LangGraph
- **AI/LLM:** OpenAI GPT-4o-mini
- **Search:** SearXNG API
- **Database:** PostgreSQL + CSV Export
