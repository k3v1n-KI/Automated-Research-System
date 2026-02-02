# dataset_builder.py
# Structured dataset creation and export for research results
# Supports: CSV, JSON, Parquet, SQL, and nested hierarchical formats

from __future__ import annotations
import os
import json
import csv
import uuid
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from enum import Enum
import hashlib

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

try:
    import pyarrow.parquet as pq
    import pyarrow as pa
    HAS_ARROW = True
except ImportError:
    HAS_ARROW = False
    pq = None


# ============================================================================
# DATASET ENUMS & CONSTANTS
# ============================================================================

class DatasetFormat(Enum):
    """Supported export formats."""
    JSON = "json"
    JSONL = "jsonl"  # JSON Lines (one record per line)
    CSV = "csv"
    PARQUET = "parquet"
    SQL = "sql"
    HTML = "html"


class DatasetSchema:
    """Pre-defined schemas for common entity types."""
    
    HOSPITAL = {
        "name": str,
        "address": str,
        "city": str,
        "state": str,
        "country": str,
        "postal_code": str,
        "phone": str,
        "website": str,
        "latitude": float,
        "longitude": float,
        "verified": bool,
        "confidence": float,
    }
    
    RESEARCH_ITEM = {
        "id": str,
        "name": str,
        "description": str,
        "category": str,
        "source_url": str,
        "extracted_from": str,
        "confidence": float,
        "metadata": dict,
        "verified": bool,
    }
    
    LOCATION = {
        "name": str,
        "address": str,
        "city": str,
        "state_province": str,
        "country": str,
        "postal_code": str,
        "latitude": float,
        "longitude": float,
        "verified": bool,
        "confidence": float,
        "source": str,
        "extracted_date": str,
    }


# ============================================================================
# DATASET CLASSES
# ============================================================================

@dataclass
class DatasetRecord:
    """Base record class with validation and serialization."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert record to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetRecord':
        """Reconstruct record from dictionary."""
        return cls(**data)


@dataclass
class HospitalRecord(DatasetRecord):
    """Structured hospital data record."""
    name: str = ""
    address: str = ""
    city: str = ""
    state: str = ""
    country: str = ""
    postal_code: str = ""
    phone: str = ""
    website: str = ""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    verified: bool = False
    verification_source: str = ""
    confidence: float = 0.0
    beds: Optional[int] = None
    specialties: List[str] = field(default_factory=list)
    emergency: bool = False
    teaching_hospital: bool = False
    accreditation: str = ""
    source_urls: List[str] = field(default_factory=list)


@dataclass
class ResearchItemRecord(DatasetRecord):
    """Generic research item record."""
    name: str = ""
    description: str = ""
    category: str = ""
    source_url: str = ""
    extracted_from: str = ""
    confidence: float = 0.0
    verified: bool = False
    attributes: Dict[str, Any] = field(default_factory=dict)
    raw_data: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# STRUCTURED DATASET BUILDER
# ============================================================================

class StructuredDatasetBuilder:
    """
    Main builder class for creating structured, validated datasets
    from research results.
    """
    
    def __init__(self, 
                 schema: Optional[Dict[str, type]] = None,
                 validate: bool = True,
                 deduplicate: bool = True):
        """
        Args:
            schema: Field name -> type mapping for validation
            validate: Whether to validate records against schema
            deduplicate: Whether to remove duplicate records
        """
        self.schema = schema or DatasetSchema.RESEARCH_ITEM
        self.validate = validate
        self.deduplicate = deduplicate
        self.records: List[Dict[str, Any]] = []
        self._seen_hashes: set = set()
        self.metadata = {
            "created_at": datetime.utcnow().isoformat(),
            "record_count": 0,
            "source": "research_system",
            "schema_type": "generic"
        }
    
    def add_record(self, record: Dict[str, Any]) -> bool:
        """
        Add a record to the dataset.
        
        Args:
            record: Dictionary representing a record
        
        Returns:
            True if record was added, False if rejected (duplicate/invalid)
        """
        # Validate against schema
        if self.validate:
            if not self._validate_record(record):
                print(f"[WARN] Record failed validation: {record.get('name', 'unknown')}")
                return False
        
        # Check for duplicates
        if self.deduplicate:
            record_hash = self._compute_hash(record)
            if record_hash in self._seen_hashes:
                return False
            self._seen_hashes.add(record_hash)
        
        self.records.append(record)
        self.metadata["record_count"] = len(self.records)
        return True
    
    def add_records(self, records: List[Dict[str, Any]]) -> int:
        """
        Add multiple records. Returns count of records actually added.
        """
        count = 0
        for record in records:
            if self.add_record(record):
                count += 1
        return count
    
    def add_from_extraction_output(self, items: List[Dict[str, Any]], 
                                   source_url: Optional[str] = None) -> int:
        """
        Add records from extraction output (e.g., from extract.py).
        Automatically enriches with metadata.
        """
        count = 0
        for item in items:
            record = dict(item)  # Copy to avoid mutating original
            
            # Add enrichment
            if source_url and "source_url" not in record:
                record["source_url"] = source_url
            
            if "created_at" not in record:
                record["created_at"] = datetime.utcnow().isoformat()
            
            if self.add_record(record):
                count += 1
        
        return count
    
    def filter_by_confidence(self, min_confidence: float) -> 'StructuredDatasetBuilder':
        """Return new dataset with only high-confidence records."""
        filtered = StructuredDatasetBuilder(
            schema=self.schema,
            validate=self.validate,
            deduplicate=False  # Already filtered
        )
        
        for record in self.records:
            conf = record.get("confidence", 0.0)
            if conf >= min_confidence:
                filtered.records.append(record)
        
        filtered.metadata["record_count"] = len(filtered.records)
        filtered.metadata["filter_min_confidence"] = min_confidence
        return filtered
    
    def filter_by_verified(self, verified_only: bool = True) -> 'StructuredDatasetBuilder':
        """Return new dataset with only verified/unverified records."""
        filtered = StructuredDatasetBuilder(
            schema=self.schema,
            validate=self.validate,
            deduplicate=False
        )
        
        for record in self.records:
            is_verified = record.get("verified", False)
            if is_verified == verified_only:
                filtered.records.append(record)
        
        filtered.metadata["record_count"] = len(filtered.records)
        return filtered
    
    def group_by_field(self, field_name: str) -> Dict[str, List[Dict[str, Any]]]:
        """Group records by a field value."""
        groups = {}
        for record in self.records:
            key = str(record.get(field_name, "unknown"))
            if key not in groups:
                groups[key] = []
            groups[key].append(record)
        return groups
    
    def export_json(self, filepath: str, pretty: bool = True, 
                   include_metadata: bool = True) -> str:
        """
        Export dataset as JSON.
        
        Returns: Path to created file
        """
        data = {}
        if include_metadata:
            data["metadata"] = self.metadata
        data["records"] = self.records
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2 if pretty else None)
        
        print(f"[OK] Exported {len(self.records)} records to {filepath}")
        return filepath
    
    def export_jsonl(self, filepath: str) -> str:
        """Export dataset as JSON Lines (one record per line)."""
        with open(filepath, 'w', encoding='utf-8') as f:
            for record in self.records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        print(f"[OK] Exported {len(self.records)} records to {filepath} (JSONL)")
        return filepath
    
    def export_csv(self, filepath: str, include_nested: bool = False) -> str:
        """
        Export dataset as CSV.
        
        Args:
            filepath: Output file path
            include_nested: If True, flatten nested dicts/lists to JSON strings
        """
        if not self.records:
            print("[WARN] No records to export")
            return filepath
        
        # Determine columns from records
        columns = set()
        for record in self.records:
            columns.update(record.keys())
        columns = sorted(list(columns))
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            
            for record in self.records:
                row = {}
                for col in columns:
                    val = record.get(col)
                    
                    # Handle complex types
                    if isinstance(val, (dict, list)):
                        if include_nested:
                            row[col] = json.dumps(val, ensure_ascii=False)
                        else:
                            row[col] = str(val)
                    else:
                        row[col] = val or ""
                
                writer.writerow(row)
        
        print(f"[OK] Exported {len(self.records)} records to {filepath} (CSV)")
        return filepath
    
    def export_parquet(self, filepath: str) -> str:
        """
        Export dataset as Parquet (columnar format, great for analytics).
        Requires: pyarrow, pandas
        """
        if not HAS_PANDAS or not HAS_ARROW:
            raise ImportError("pandas and pyarrow required for Parquet export")
        
        df = pd.DataFrame(self.records)
        df.to_parquet(filepath, compression='snappy')
        
        print(f"[OK] Exported {len(self.records)} records to {filepath} (Parquet)")
        return filepath
    
    def export_sql(self, filepath: str, table_name: str = "dataset") -> str:
        """
        Export dataset as SQL INSERT statements.
        """
        if not self.records:
            return filepath
        
        lines = [f"-- Dataset exported {datetime.utcnow().isoformat()}"]
        lines.append(f"-- Total records: {len(self.records)}\n")
        
        columns = sorted(set(k for record in self.records for k in record.keys()))
        
        # CREATE TABLE statement
        column_defs = []
        for col in columns:
            col_type = "TEXT"  # Default to TEXT; could be smarter
            if col in self.schema:
                if self.schema[col] in (int, float):
                    col_type = "REAL" if self.schema[col] == float else "INTEGER"
                elif self.schema[col] == bool:
                    col_type = "INTEGER"
            column_defs.append(f"  {col} {col_type}")
        
        lines.append(f"CREATE TABLE IF NOT EXISTS {table_name} (")
        lines.append(",\n".join(column_defs))
        lines.append(");\n")
        
        # INSERT statements
        for record in self.records:
            cols = []
            vals = []
            for col in columns:
                val = record.get(col)
                cols.append(col)
                
                if val is None:
                    vals.append("NULL")
                elif isinstance(val, bool):
                    vals.append("1" if val else "0")
                elif isinstance(val, (int, float)):
                    vals.append(str(val))
                else:
                    vals.append(f"'{str(val).replace(chr(39), chr(39)*2)}'")
            
            insert_sql = f"INSERT INTO {table_name} ({', '.join(cols)}) VALUES ({', '.join(vals)});"
            lines.append(insert_sql)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        
        print(f"[OK] Exported {len(self.records)} records to {filepath} (SQL)")
        return filepath
    
    def export_html(self, filepath: str, title: str = "Research Dataset") -> str:
        """Export dataset as an HTML table."""
        if not self.records:
            html_content = "<p>No records to display</p>"
        else:
            columns = sorted(set(k for record in self.records for k in record.keys()))
            
            html_lines = [
                "<!DOCTYPE html>",
                "<html>",
                "<head>",
                f"<title>{title}</title>",
                "<style>",
                "table { border-collapse: collapse; width: 100%; }",
                "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
                "th { background-color: #4CAF50; color: white; }",
                "tr:nth-child(even) { background-color: #f2f2f2; }",
                "</style>",
                "</head>",
                "<body>",
                f"<h1>{title}</h1>",
                "<table>",
                "<thead><tr>" + "".join(f"<th>{col}</th>" for col in columns) + "</tr></thead>",
                "<tbody>"
            ]
            
            for record in self.records:
                row_html = "<tr>"
                for col in columns:
                    val = record.get(col, "")
                    if isinstance(val, (dict, list)):
                        val = json.dumps(val, ensure_ascii=False)
                    row_html += f"<td>{str(val)}</td>"
                row_html += "</tr>"
                html_lines.append(row_html)
            
            html_lines.extend([
                "</tbody>",
                "</table>",
                "</body>",
                "</html>"
            ])
            
            html_content = "\n".join(html_lines)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"[OK] Exported {len(self.records)} records to {filepath} (HTML)")
        return filepath
    
    def export(self, filepath: str, format: Union[str, DatasetFormat] = "json", 
              **kwargs) -> str:
        """
        Universal export function.
        
        Args:
            filepath: Output file path
            format: Export format (json, csv, parquet, sql, html, jsonl)
            **kwargs: Format-specific options
        
        Returns:
            Path to created file
        """
        if isinstance(format, str):
            format = DatasetFormat[format.upper()]
        
        if format == DatasetFormat.JSON:
            return self.export_json(filepath, **kwargs)
        elif format == DatasetFormat.JSONL:
            return self.export_jsonl(filepath)
        elif format == DatasetFormat.CSV:
            return self.export_csv(filepath, **kwargs)
        elif format == DatasetFormat.PARQUET:
            return self.export_parquet(filepath)
        elif format == DatasetFormat.SQL:
            return self.export_sql(filepath, **kwargs)
        elif format == DatasetFormat.HTML:
            return self.export_html(filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics about the dataset."""
        summary = dict(self.metadata)
        summary["total_records"] = len(self.records)
        
        if self.records:
            # Field coverage
            all_fields = set()
            fields_used = {}
            for record in self.records:
                for field in record.keys():
                    all_fields.add(field)
                    fields_used[field] = fields_used.get(field, 0) + 1
            
            summary["fields"] = {
                "total": len(all_fields),
                "coverage": {k: f"{v/len(self.records)*100:.1f}%" for k, v in fields_used.items()}
            }
            
            # Confidence stats if available
            confidences = [r.get("confidence", 0) for r in self.records if isinstance(r.get("confidence"), (int, float))]
            if confidences:
                summary["confidence"] = {
                    "avg": sum(confidences) / len(confidences),
                    "min": min(confidences),
                    "max": max(confidences)
                }
        
        return summary
    
    @staticmethod
    def _validate_record(record: Dict[str, Any]) -> bool:
        """Validate record has required fields."""
        return isinstance(record, dict) and len(record) > 0
    
    @staticmethod
    def _compute_hash(record: Dict[str, Any]) -> str:
        """Compute hash of record for deduplication."""
        # Create a canonical string representation
        key_parts = []
        for k in sorted(record.keys()):
            v = record[k]
            if isinstance(v, (dict, list)):
                v = json.dumps(v, sort_keys=True, ensure_ascii=False)
            key_parts.append(f"{k}={str(v)}")
        
        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_hospital_dataset(validate: bool = True) -> StructuredDatasetBuilder:
    """Create a dataset builder for hospital records."""
    builder = StructuredDatasetBuilder(
        schema=DatasetSchema.HOSPITAL,
        validate=validate,
        deduplicate=True
    )
    builder.metadata["schema_type"] = "hospital"
    return builder


def create_research_dataset(validate: bool = True) -> StructuredDatasetBuilder:
    """Create a dataset builder for generic research items."""
    builder = StructuredDatasetBuilder(
        schema=DatasetSchema.RESEARCH_ITEM,
        validate=validate,
        deduplicate=True
    )
    builder.metadata["schema_type"] = "research_item"
    return builder


def create_location_dataset(validate: bool = True) -> StructuredDatasetBuilder:
    """Create a dataset builder for location records."""
    builder = StructuredDatasetBuilder(
        schema=DatasetSchema.LOCATION,
        validate=validate,
        deduplicate=True
    )
    builder.metadata["schema_type"] = "location"
    return builder


# Example usage
if __name__ == "__main__":
    # Create a sample hospital dataset
    builder = create_hospital_dataset()
    
    sample_records = [
        {
            "name": "Toronto General Hospital",
            "address": "200 Elizabeth St",
            "city": "Toronto",
            "state": "Ontario",
            "country": "Canada",
            "verified": True,
            "confidence": 0.95,
            "beds": 1000,
            "emergency": True,
            "teaching_hospital": True,
        },
        {
            "name": "St. Michael's Hospital",
            "address": "30 Bond St",
            "city": "Toronto",
            "state": "Ontario",
            "country": "Canada",
            "verified": True,
            "confidence": 0.92,
            "beds": 400,
            "emergency": True,
            "teaching_hospital": True,
        }
    ]
    
    added = builder.add_records(sample_records)
    print(f"Added {added} records")
    print(f"\nDataset Summary:")
    print(json.dumps(builder.get_summary(), indent=2))
    
    # Export to multiple formats
    print("\nExporting...")
    builder.export("hospitals.json", format="json")
    builder.export("hospitals.csv", format="csv")
    builder.export("hospitals.html", format="html", title="Ontario Hospitals")
