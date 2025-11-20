# logger.py
import time
from typing import Any, Dict
from firebase import db  

RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
RED = "\033[31m"
GRAY = "\033[90m"

def _ts(): return time.strftime("%Y-%m-%d %H:%M:%S")

class DualLogger:
    """Logger that writes to both stdout and Firestore."""
    
    def __init__(self, run_id: str = ""):
        """Initialize logger. Empty run_id disables Firestore logging."""
        self.run_id = run_id
        
    def log(self, kind: str, msg: str):
        """Log a message to stdout (and Firestore if run_id set)."""
        color = {"STEP":CYAN, "INFO":GRAY, "OK":GREEN, "WARN":YELLOW, "ERR":RED}.get(kind, GRAY)
        print(f"{BOLD if kind=='STEP' else ''}{color}[{_ts()}][{kind}]{RESET} {msg}")
        if self.run_id and db:
            try:
                db.collection("logs").document(f"{self.run_id}_{_ts()}").set(
                    {"ts": _ts(), "kind": kind, "msg": msg}
                )
            except Exception:
                pass

    def write_round_doc(self, round_idx: int, node: str, payload: Dict[str, Any]):
        """Write to runs/{run_id}/rounds/{round}/steps/{node}"""
        if not self.run_id or not db:
            return
            
        col = db.collection("runs").document(self.run_id)\
                    .collection("rounds").document(f"{round_idx:03d}")\
                    .collection("steps")
                    
        # Special handling for validation results
        if node == "validate":
            results_doc = {
                "ts": _ts(),
                "total_urls": payload.get("raw_count", 0),
                "kept_urls": payload.get("kept_count", 0),
                "discarded_urls": payload.get("raw_count", 0) - payload.get("kept_count", 0),
                "queries": payload.get("queries", []),
                "per_query": payload.get("per_query", {})
            }
            col.document("results_validate").set(results_doc)
            
        col.document(node).set({
            "ts": _ts(),
            **payload
        }, merge=True)

# Global logger instance for convenience
logger = DualLogger()

def write_round_doc(self, round_idx: int, node: str, payload: Dict[str, Any]):
    """Write to runs/{run_id}/rounds/{round}/steps/{node}"""
    if not self.run_id or not db:
        return
        
    col = db.collection("runs").document(self.run_id)\
                .collection("rounds").document(f"{round_idx:03d}")\
                .collection("steps")
                
    # Special handling for validation results
    if node == "validate":
        results_doc = {
            "ts": _ts(),
            "total_urls": payload.get("raw_count", 0),
            "kept_urls": payload.get("kept_count", 0),
            "discarded_urls": payload.get("raw_count", 0) - payload.get("kept_count", 0),
            "queries": payload.get("queries", []),
            "per_query": payload.get("per_query", {})
        }
        col.document("results_validate").set(results_doc)
        
    col.document(node).set({
        "ts": _ts(),
        **payload
    }, merge=True)

def write_round_root(self, round_idx: int, payload: Dict[str, Any]):
    """Write to runs/{run_id}/rounds/{round}"""
    if not self.run_id or not db:
        return
        
    col = db.collection("runs").document(self.run_id)\
                .collection("rounds")
    col.document(f"{round_idx:03d}").set(payload, merge=True)

def write_run_root(self, payload: Dict[str, Any]):
    """Write to runs/{run_id} root doc."""
    if db:
        db.collection("runs").document(self.run_id).set(payload, merge=True)
