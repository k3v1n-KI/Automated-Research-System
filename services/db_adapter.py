# services/db_adapter.py
import json
from typing import Any, Dict, List
from datetime import datetime, timezone
from firebase import db

FIRESTORE_DOC_MAX = 1_048_576          # 1 MiB hard cap
FIRESTORE_SAFE_BUDGET = 900_000        # leave headroom for metadata/overhead

def _ts_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _sanitize_for_firestore(val: Any) -> Any:
    if val is None: return None
    if isinstance(val, (str, int, float, bool)): return val
    if isinstance(val, tuple): return [_sanitize_for_firestore(x) for x in val]
    if isinstance(val, list):  return [_sanitize_for_firestore(x) for x in val]
    if isinstance(val, dict):
        return {str(k): _sanitize_for_firestore(v) for k, v in val.items()}
    return str(val)

def _json_bytes(obj: Any) -> int:
    try:
        return len(json.dumps(obj, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))
    except Exception:
        # best effort
        return len(str(obj).encode('utf-8'))

def _compact_scrape_item(it: Dict[str, Any], *, max_text_len: int = 20000) -> Dict[str, Any]:
    """
    Keep only what we need for extract; drop heavy fields.
    - ALWAYS drop 'html'
    - Cap 'clean_text' (or text/content) to max_text_len
    """
    keep = {}
    # Always carry URL + small metadata if present
    for k in ("url", "title"):
        v = it.get(k)
        if isinstance(v, str): keep[k] = v

    # Normalize a text field
    text = ""
    for k in ("clean_text", "text", "content", "cleaned", "body"):
        v = it.get(k)
        if isinstance(v, str) and v.strip():
            text = v
            break
    if text:
        if len(text) > max_text_len:
            text = text[:max_text_len]
        keep["clean_text"] = text

    # Optionally carry status_code / content_type if you store them and they’re small
    for k in ("status_code", "content_type"):
        v = it.get(k)
        if isinstance(v, (str, int)):
            keep[k] = v

    return keep

class PlanRunDbAdapter:
    def __init__(self, plan_id: str, run_id: str):
        self.plan_id = plan_id
        self.run_id = run_id

    def _root(self):
        return db.collection("research_plans").document(self.plan_id) if db else None

    def _runs(self):
        r = self._root()
        return r.collection("runs").document(self.run_id) if r else None

    def write_run_root(self, payload: Dict[str, Any]):
        ref = self._runs()
        if ref:
            ref.set({"ts": _ts_iso(), **_sanitize_for_firestore(payload)}, merge=True)

    def write_step(self, name: str, payload: Dict[str, Any]):
        ref = self._runs()
        if ref:
            ref.collection("steps").document(name).set(
                {"ts": _ts_iso(), **_sanitize_for_firestore(payload)}, merge=True
            )

    def write_round(self, round_idx: int, payload: Dict[str, Any]):
        ref = self._runs()
        if ref:
            ref.collection("rounds").document(f"{round_idx:03d}").set(
                {"ts": _ts_iso(), **_sanitize_for_firestore(payload)}, merge=True
            )

    def write_artifact(self, kind: str, payload: Dict[str, Any]):
        ref = self._runs()
        if ref:
            ref.collection("artifacts").document(kind).set(
                {"ts": _ts_iso(), **_sanitize_for_firestore(payload)}, merge=True
            )

    def remember_json(self, key: str, payload: Dict[str, Any]):
        ref = self._runs()
        if ref:
            ref.collection("memory").document(key).set(
                {"ts": _ts_iso(), **_sanitize_for_firestore(payload)}, merge=True
            )

    # ---------- BYTE-AWARE WRITER ----------
    def _write_list_chunks_bytes(self, round_idx: int, collection: str, doc_prefix: str, data_list: List[Dict[str, Any]]):
        """
        Build docs under ~900KB each. Writes:
        research_plans/{plan}/runs/{run}/rounds/{idx}/{collection}/{doc_prefix}_{NNN}
        """
        ref = self._runs()
        if not ref: return
        col = ref.collection("rounds").document(f"{round_idx:03d}").collection(collection)

        batch: List[Dict[str, Any]] = []
        i = 0

        def _flush():
            nonlocal i, batch
            if not batch: return
            payload = {"ts": _ts_iso(), "count": len(batch), "items": _sanitize_for_firestore(batch)}
            # Ensure payload stays under budget; if not, conservatively split one-by-one
            if _json_bytes(payload) > FIRESTORE_SAFE_BUDGET:
                # fallback: write individually
                for j, item in enumerate(batch):
                    single = {"ts": _ts_iso(), "count": 1, "items": [_sanitize_for_firestore(item)]}
                    col.document(f"{doc_prefix}_{i:03d}").set(single)
                    i += 1
            else:
                col.document(f"{doc_prefix}_{i:03d}").set(payload)
                i += 1
            batch = []

        # build batches under byte budget
        for item in data_list or []:
            candidate = batch + [item]
            payload = {"ts": "x", "count": len(candidate), "items": _sanitize_for_firestore(candidate)}
            if _json_bytes(payload) > FIRESTORE_SAFE_BUDGET:
                _flush()
                batch = [item]
            else:
                batch = candidate
        _flush()

    # ---------- PUBLIC SAVE APIS ----------
    def write_search_results(self, round_idx: int, raw_results: List[Dict[str, Any]]):
        self._write_list_chunks_bytes(round_idx, "search", "results", raw_results or [])

    def write_validated(self, round_idx: int, urls: List[str]):
        dict_list = [{"url": u} for u in (urls or []) if isinstance(u, str)]
        self._write_list_chunks_bytes(round_idx, "validate", "urls", dict_list)

    def write_validate_rejected(self, round_idx: int, items: List[Dict[str, Any]]):
        """
        Persist rejected/filtered validation candidates for auditing. Each item
        should be a dict with at least 'url' and optionally 'score'.
        Stored under rounds/{idx}/validate/rejected_{NNN}
        """
        dict_list = []
        for it in (items or []):
            u = it.get("url") if isinstance(it, dict) else None
            if not u or not isinstance(u, str):
                continue
            d = {"url": u}
            if isinstance(it.get("score"), (int, float)):
                d["score"] = float(it.get("score"))
            dict_list.append(d)
        self._write_list_chunks_bytes(round_idx, "validate", "rejected", dict_list)

    def write_scrapes(self, round_idx: int, scrapes: List[Dict[str, Any]]):
        #  COMPACT before saving (drop html, cap text)
        compact = [_compact_scrape_item(s) for s in (scrapes or [])]
        self._write_list_chunks_bytes(round_idx, "scrape", "scrapes", compact)

    def write_extracted(self, round_idx: int, items: List[Dict[str, Any]]):
        self._write_list_chunks_bytes(round_idx, "extract", "items", items or [])

    def write_aggregated(self, round_idx: int, items: List[Dict[str, Any]]):
        self._write_list_chunks_bytes(round_idx, "aggregate", "items", items or [])
