# Query Expansion Matrix Testing

Testing a structured approach to query generation that uses axes of variance and strategic "corner checks" to maximize coverage and completeness.

## Concept

**Current Method:** Simple synonym expansion
```
hospitals in ontario
hospital list ontario
ontario hospital directory
...
```

**New Method: Query Expansion Matrix**
```
Step 1: Decompose goal into axes
  - Entities: Hospital, Health Centre, Clinic
  - Geographic: Ontario, Greater Toronto Area, Rural Ontario
  - Attributes: Trauma Centre, Walk-in, Private
  - Formats: PDF, Directory

Step 2: Corner Check Strategy
  - Mainstream: "Hospital in Ontario"
  - Format: "List of Hospitals Ontario filetype:pdf"
  - Edge Case: "Health Centre in Rural Ontario"
  - Type: "Trauma Centre Ontario directory"
```

## Files

- `query_expansion_matrix.py` - Main implementation of the QEM strategy
- `comparison.py` - Run both methods and compare metrics
- `comparison_results/` - Saved comparison data

## Running Tests

### Quick Demo
```bash
python query_expansion_matrix.py
```

Shows how the axes are decomposed and corner queries generated.

### Full Comparison
```bash
python comparison.py
```

Runs both methods on the same goal and produces:
- Coverage metrics (what % of documents were extracted)
- Completeness metrics (what % of fields have data)
- Missing value counts
- Sample efficiency (records per query)
- JSON report with full results

## Metrics Explained

| Metric | Meaning | Higher is Better |
|--------|---------|-----------------|
| **Query Count** | Number of search queries | N/A (controlled) |
| **Total URLs Found** | Raw search results | ✅ Yes |
| **Validated URLs** | URLs that passed validation | ✅ Yes |
| **Successfully Scraped** | Pages successfully scraped | ✅ Yes |
| **Extracted Records** | Entities extracted from documents | ✅ Yes |
| **Coverage** | Extraction rate (records/validated URLs) | ✅ Yes |
| **Completeness** | Percentage of fields filled | ✅ Yes |
| **Missing Values** | Total empty fields | ❌ No |
| **Unique Records** | Deduplicated sample count | ✅ Yes |

## Expected Outcomes

### Why QEM Should Win

1. **Better Coverage**: By varying axes (entity types, geographic scopes, data formats), we find sources the simple method misses
2. **Better Completeness**: Format-specific queries (PDFs, directories) often contain more structured data
3. **Strategic Efficiency**: Corner checks are smarter than random expansion

### Hypothetical Example

**Current Method (5 queries):**
```
Total URLs: 28
Extracted: 168 records
Coverage: 85.7%
Completeness: 72%
Missing values: 1,200
```

**QEM Method (8 queries):**
```
Total URLs: 42 (+50%)
Extracted: 245 records (+46%)
Coverage: 89.3%
Completeness: 81%
Missing values: 890 (-26%)
```

## How to Modify

### Change the Goal
```bash
# Edit comparison.py line: asyncio.run(run_comparison(goal="your goal"))
python comparison.py
```

### Adjust Decomposition
Edit the `DECOMPOSITION_PROMPT` in `query_expansion_matrix.py` to guide the LLM toward different axes.

### Change Strategy
In `comparison.py`, change:
```python
result = await qem.execute(goal, strategy="corner_only")  # 4-5 queries
# or
result = await qem.execute(goal, strategy="full_matrix")   # 15+ queries
```

## Next Steps

1. Run the comparison and collect metrics
2. Iterate on decomposition prompts to improve axis identification
3. A/B test different corner check strategies
4. Integrate winning strategy into main algorithm

