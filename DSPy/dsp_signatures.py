# dsp_signatures.py
import dspy
import json

class SeedQueryGenSig(dspy.Signature):
    """Generate N diverse, precise search queries for an info-gathering goal."""
    goal = dspy.InputField()
    n = dspy.InputField()
    queries = dspy.OutputField(desc="list[str]")

class CriticActionsSig(dspy.Signature):
    """Propose compact actions JSON to improve next-round coverage and fill gaps."""
    goal = dspy.InputField()
    round_profile = dspy.InputField(desc="stats: item_count, field_coverage, new_items, top_domains")
    constraints = dspy.InputField(desc="deny_terms, max_queries, budgets")
    actions_json = dspy.OutputField(desc="JSON with keys: expand_sources, enrich_missing, deny_terms, max_queries")

import re

class ExtractNames(dspy.Signature):
    """Extract Ontario hospital names from text.

    Find all unique hospital names in the text that are located in Ontario, Canada.
    Only include actual hospital names (e.g. "Toronto General Hospital"), not generic mentions.
    Skip any entries that don't seem to be actual hospital names.

    Example output: ["Toronto General Hospital", "Mount Sinai Hospital"]
    """

    page_text = dspy.InputField(desc="Text to extract hospital names from")
    names = dspy.OutputField(desc="List of Ontario hospital names found in text")

class ExtractHospitalDetails(dspy.Signature):
    """Extract details for a specific hospital from text.

    Given a hospital name and text, extract the following fields:
    - address: Full Ontario address (e.g. "200 Elizabeth St, Toronto, ON M5G 2C4")
    - phone: Phone number in (XXX) XXX-XXXX format
    - website: Full URL starting with http:// or https://
    
    Skip any fields that aren't found or aren't properly formatted.
    """

    hospital_name = dspy.InputField(desc="Name of hospital to find details for")
    page_text = dspy.InputField(desc="Text to search for hospital details")
    details = dspy.OutputField(desc="Dictionary with address, phone, website fields")

class ExtractHospitalsSig(dspy.Signature):
    """Extract Ontario hospitals from text and return as JSON list.

    First extract all hospital names, then find details for each hospital.
    Output format is a list of hospital objects with these fields:
    - name: Full hospital name 
    - address: Full Ontario address
    - phone: Phone number in (XXX) XXX-XXXX format
    - website: Full URL
    - source_url: URL where information was found

    Skip any entries missing required fields.
    """

    page_text = dspy.InputField(desc="Text containing hospital information")
    goal = dspy.InputField(desc="Description of extraction goal")
    schema = dspy.InputField(desc="List of required fields to extract")
    items = dspy.OutputField(desc="List of hospital objects")

    def __init__(self):
        super().__init__()
        # Load examples
        from .hospital_examples import EXAMPLES
        self.examples = EXAMPLES
        
        # Initialize sub-predictors
        self.name_finder = dspy.Predict(ExtractNames)
        self.details_finder = dspy.Predict(ExtractHospitalDetails)
        self._compiled = False

    def forward(self, goal: str, page_text: str, schema: list) -> dict:
        """Extract hospitals using two-stage process"""
        # First get all hospital names
        print(f"[EXTRACT] Finding hospital names in text len={len(page_text)}")
        resp = self.name_finder(page_text=page_text)
        if not hasattr(resp, "names") or not isinstance(resp.names, list):
            print("[EXTRACT] No names found")
            return {"items": []}
        
        print(f"[EXTRACT] Found {len(resp.names)} hospital names")

        # Extract source URL
        source_url = ""
        if isinstance(page_text, str) and "http" in page_text:
            # Try to find source URL in the text
            match = re.search(r"(https?://[^\s]+)", page_text)
            if match:
                source_url = match.group(1)

        # Then get details for each name
        items = []
        for name in resp.names:
            print(f"[EXTRACT] Finding details for {name}")
            details = self.details_finder(hospital_name=name, page_text=page_text)
            if not hasattr(details, "details") or not isinstance(details.details, dict):
                continue

            # Create hospital entry with basic validation
            hospital = {"name": name, "source_url": source_url}
            
            # Add validated fields
            for field in ["address", "phone", "website"]:
                if field in details.details and details.details[field]:
                    # Basic format validation
                    if field == "address" and "toronto" in details.details[field].lower():
                        hospital[field] = details.details[field]
                    elif field == "phone" and re.match(r'\(\d{3}\) \d{3}-\d{4}', details.details[field]):
                        hospital[field] = details.details[field]
                    elif field == "website" and details.details[field].startswith(('http://', 'https://')):
                        hospital[field] = details.details[field]

            # Check required fields
            missing_required = False
            for field in schema:
                if field not in hospital:
                    print(f"[EXTRACT] Missing required field {field}")
                    missing_required = True
                    break

            if not missing_required:
                print(f"[EXTRACT] Added hospital {name}")
                items.append(hospital)

        print(f"[EXTRACT] Returning {len(items)} validated hospitals")
        return {"items": items}


class ExtractEntitiesSig(dspy.Signature):
    """Extract goal-specific entities from page text into the required schema."""
    goal = dspy.InputField()
    page_text = dspy.InputField()
    schema = dspy.InputField(desc="required fields, e.g., ['name','address','phone','website','source_url']")
    items = dspy.OutputField(desc="list[dict] rows with required fields present")
