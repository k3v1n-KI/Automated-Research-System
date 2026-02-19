# Improved Decomposition Prompt - Enhancement Summary

## What Changed

The `ZeroShotDecomposer` class now uses **two context-aware prompts** instead of a single hardcoded one:

### Before
```python
DECOMPOSITION_PROMPT = """
You are a research query strategist. Decompose the user's research goal into 
structured axes of variance...
```
- ❌ Too specific to hospital examples
- ❌ Hardcoded axis types: `entities`, `geographic`, `attributes`, `formats`
- ❌ No awareness of user-provided columns
- ❌ Same prompt regardless of domain

### After
```python
DECOMPOSITION_PROMPT_BASE  # Generic, domain-agnostic
DECOMPOSITION_PROMPT_WITH_COLUMNS  # Column-aware variant
```

✅ Works across ANY domain (hospitals, software, research papers, locations)
✅ Uses column specs to guide axis discovery
✅ More descriptive axis names: `primary_entities`, `scope`, `characteristics`, `sources`
✅ Adaptive prompt selection based on inputs

---

## Key Improvements

### 1. **Generic Axis Framework**
Instead of hardcoded hospital-specific axes:
```
OLD: entities → geographic → attributes → formats
NEW: primary_entities → scope → characteristics → sources
```

The new names are domain-agnostic:
- **Primary Entities**: What types/categories exist? (Works for hospitals, libraries, papers, etc.)
- **Scope**: What scale/levels matter? (Not just geographic)
- **Characteristics**: What properties vary? (More than "attributes")
- **Sources**: What formats/types of sources? (More than just data formats)

### 2. **Column Context Support**
When user provides columns, the prompt uses them to guide axis discovery:

```python
# Without columns - generic decomposition
axes = await decomposer.decompose("Find hospitals in Ontario")

# With columns - column-guided decomposition  
columns = [
    {"name": "Name", "description": "Hospital name"},
    {"name": "Address", "description": "Street address"},
    {"name": "City", "description": "Municipality"},
    {"name": "Phone", "description": "Contact number"},
]
axes = await decomposer.decompose(
    "Find hospitals in Ontario",
    column_specs=columns
)
```

The column-aware prompt tells the LLM:
> "Your decomposition should help find diverse sources that might contain or inform these fields."

### 3. **Flexible JSON Structure**
The returned JSON now uses more flexible, self-explanatory keys:
```json
{
    "primary_entities": {"name": "Primary Entity Types", "values": [...]},
    "scope": {"name": "Scope/Scale", "values": [...]},
    "characteristics": {"name": "Key Characteristics", "values": [...]},
    "sources": {"name": "Data Sources", "values": [...]}
}
```

---

## API Changes

### Updated Method Signature
```python
async def decompose(
    self, 
    goal: str, 
    column_specs: List[Dict] = None  # NEW: Optional
) -> Dict[str, AxisOfVariance]:
```

### Updated Orchestrator
```python
# QueryExpansionMatrix.execute() now accepts columns
result = await qem.execute(
    goal="Find hospitals in Ontario",
    strategy="corner_only",
    column_specs=columns  # NEW: Optional
)
```

---

## Testing Results

### Test 1: Hospital Domain (No Columns)
✅ Successfully identified:
- Primary Entity Types: 5 variants (acute-care, teaching, specialized, networks, regional)
- Scope/Scale: 3 variants (broad, medium, narrow)
- Key Characteristics: 5 variants (funding, affiliation, bed size, specializations, ED presence)
- Data Sources: 5 variants (open data, CIHI, directories, websites, geospatial)

### Test 2: Hospital Domain (With Columns)
✅ Axes focused on improving coverage of: Name, Address, City, Phone, Bed Count, Specializations

### Test 3: Software Domain (No Columns)
✅ Decomposition adapted to Python libraries (no longer hospital-specific)

### Test 4: Software Domain (With Columns)
✅ Axes focused on: Name, GitHub URL, Version, Documentation, Use Case, Installation

---

## Implementation Details

### Prompt Selection Logic
```python
if column_specs:
    # Format columns into readable text
    columns_text = "\n".join([
        f"- {col.get('name')}: {col.get('description')}"
        for col in column_specs
    ])
    # Use column-aware prompt
    system_prompt = self.DECOMPOSITION_PROMPT_WITH_COLUMNS.format(
        columns=columns_text
    )
else:
    # Use generic prompt
    system_prompt = self.DECOMPOSITION_PROMPT_BASE
```

### LLM Call
```python
response = await self.client.chat.completions.create(
    model=self.model,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Goal: {goal}"}
    ]
)
```

---

## Example Usage Scenarios

### Scenario 1: Hospital Research (With Field Guidance)
```python
goal = "Find hospitals in Ontario"
columns = [
    {"name": "Name", "description": "Hospital name"},
    {"name": "Address", "description": "Street address"},
    {"name": "City", "description": "Municipality"},
]

result = await qem.execute(goal, column_specs=columns)
# LLM will focus on finding variants that improve these 3 fields
```

### Scenario 2: Software Library Research (Generic)
```python
goal = "Find popular Python machine learning libraries"

result = await qem.execute(goal)  # No columns specified
# Works perfectly without column context
```

### Scenario 3: Academic Paper Search (With Metadata Fields)
```python
goal = "Find research on climate change solutions"
columns = [
    {"name": "Title", "description": "Paper title"},
    {"name": "Authors", "description": "Author names"},
    {"name": "Publication Year", "description": "Year published"},
    {"name": "DOI", "description": "Digital object identifier"},
]

result = await qem.execute(goal, column_specs=columns)
# Axes will vary to maximize coverage of these academic fields
```

---

## Backward Compatibility

✅ **Fully backward compatible**
- Old calls `decompose(goal)` still work (column_specs defaults to None)
- Code using `execute(goal, strategy)` still works (column_specs defaults to None)
- Existing tests and scripts require no changes

---

## Next Steps (Optional Enhancements)

1. **Multi-level axis discovery**: Allow hierarchical axes (parent → child variants)
2. **Importance weighting**: Let users mark some columns as higher priority
3. **Custom axis names**: Allow users to suggest axis types upfront
4. **Axis validation**: Check that identified axes actually help cover the columns
5. **Caching**: Store decompositions for repeated goals across sessions

---

## Files Modified

- `test/query_expansion/query_expansion_matrix.py` (lines 44-110, 120-140, 296-309)
- Tests added: `test/query_expansion/test_column_context.py`

---

## Summary

The decomposition prompt is now **domain-agnostic** and **column-aware**, making it useful for:
- ✅ Any research domain (not just healthcare)
- ✅ Leveraging known field requirements to guide axis discovery
- ✅ More descriptive, flexible axis naming
- ✅ Better overall coverage of requested data fields
