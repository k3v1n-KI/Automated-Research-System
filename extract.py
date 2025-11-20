# extract.py 
from __future__ import annotations
import os, json, re, uuid, time
from typing import Any, Dict, List, Optional, Tuple

try:
    import dspy
    _DSPY = True
except Exception:
    dspy = None
    _DSPY = False

try:
    from bs4 import BeautifulSoup  # noqa
    _HAS_BS4 = True
except Exception:
    _HAS_BS4 = False


from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
dotenv_path = find_dotenv()
if dotenv_path:
    load_dotenv(dotenv_path)

# ---------------------------
# debug / logging helpers
# ---------------------------

def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def _dbg(phase: str, msg: str) -> None:
    print(f"[{_ts()}][EXTRACT][{phase}] {msg}", flush=True)


# ---------------------------
# utilities
# ---------------------------

def _chunks(txt: str, size: int = 3000, overlap: int = 250):
    n = len(txt)
    if n <= size:
        _dbg("chunk", f"yielding single chunk len={n}")
        yield txt
        return
    i = 0
    idx = 0
    while i < n:
        j = min(i + size, n)
        piece = txt[i:j]
        _dbg("chunk", f"yield chunk#{idx} i={i} j={j} len={len(piece)}")
        yield piece
        if j == n:
            break
        i = max(j - overlap, i + 1)
        idx += 1

def _coalesce_text(scrape: Dict[str, Any]) -> str:
    for k in ("clean_text", "text", "content", "cleaned", "body"):
        v = scrape.get(k)
        if isinstance(v, str) and v.strip():
            _dbg("coalesce", f"picked '{k}' len={len(v)} for url={scrape.get('url')}")
            return v
    if _HAS_BS4:
        html = scrape.get("html")
        if isinstance(html, str) and html.strip():
            try:
                t = BeautifulSoup(html, "lxml").get_text(" ", strip=True)
                _dbg("coalesce", f"converted html->text len={len(t)} for url={scrape.get('url')}")
                return t
            except Exception as e:
                _dbg("coalesce", f"BeautifulSoup failed: {e}")
    _dbg("coalesce", f"no usable text for url={scrape.get('url')}")
    return ""

def _coerce_items(j: Any) -> List[Dict[str, Any]]:
    try:
        if isinstance(j, dict):
            if isinstance(j.get("items"), list):
                return [x for x in j["items"] if isinstance(x, dict)]
            if isinstance(j.get("list"), list):
                return [x for x in j["list"] if isinstance(x, dict)]
            if isinstance(j.get("items"), str):
                try:
                    arr = json.loads(j["items"])
                    if isinstance(arr, list):
                        return [x for x in arr if isinstance(x, dict)]
                except Exception as e:
                    _dbg("coerce", f"items str->json failed: {e}")
        if isinstance(j, list):
            return [x for x in j if isinstance(x, dict)]
    except Exception as e:
        _dbg("coerce", f"unexpected structure: {type(j)} err={e}")
    return []

# More precise hospital regex patterns
_HOSP_NAME_REGEX = re.compile(r"(?im)(?:^|\n)(?!.*(?:List of|Hospitals in|Overview))([A-Z][^\n]{2,70}(?:Hospital|Medical Center|Healthcare)[^\n]{0,40}(?:$|\n))")
_ADDR_REGEX = re.compile(r"(?im)(?:^|\n)\s*(?:located at|address[:]?)?\s*(\d+[^\n]{5,100}(?:Street|St\.|Avenue|Ave\.|Road|Rd\.|Drive|Dr\.|Boulevard|Blvd\.|Lane|Ln\.|Way|Circle|Cir\.|Court|Ct\.)[^\n]{0,40}(?:$|\n))")
_PHONE_REGEX = re.compile(r"(?:(?:\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4})")
_WEBSITE_REGEX = re.compile(r"(?:https?://)?(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+(?:/[^\s]*)?")

def _validate_hospital_item(item: Dict[str, Any]) -> bool:
    """Return True if item meets minimum quality requirements."""
    name = item.get("name", "").strip()
    if not name or len(name) < 5 or name.lower().startswith(("list of", "hospitals in")):
        return False
    # Relaxed acceptance: require a name plus at least one contact/source signal.
    # Accept if item has any of: address, website, phone, source_url.
    address = item.get("address", "").strip()
    website = item.get("website", "").strip()
    phone = item.get("phone", "").strip()
    source = item.get("source_url", "").strip() or item.get("url", "").strip()

    has_valid_addr = bool(address and len(address) > 8)
    has_valid_web = bool(website and (website.startswith(("http://", "https://", "www.")) or "." in website))
    has_phone = bool(phone and len(phone) >= 7)
    has_source = bool(source)

    # If the name already clearly looks like a hospital token, be more permissive.
    name_token_ok = any(tok.lower() in name.lower() for tok in ["hospital", "health", "clinic", "centre", "center", "medical"])

    # Accept item when we have name + at least one signal OR a strong name token.
    return (name and (has_valid_addr or has_valid_web or has_phone or has_source)) or (name and name_token_ok)

def _regex_fallback(txt: str, limit: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    
    # Get all potential hospital names first
    for m in _HOSP_NAME_REGEX.finditer(txt):
        name = re.sub(r"\s+", " ", m.group(1)).strip(" -•\t")
        pos = m.start()
        
        # Look for address, phone, website in nearby context (next 500 chars)
        context = txt[max(0, pos-100):min(len(txt), pos+500)]
        
        addr_match = _ADDR_REGEX.search(context)
        phone_match = _PHONE_REGEX.search(context)
        web_match = _WEBSITE_REGEX.search(context)
        
        item = {
            "name": name,
            "address": addr_match.group(1).strip() if addr_match else "",
            "phone": phone_match.group(0) if phone_match else "",
            "website": web_match.group(0) if web_match else ""
        }
        
        if _validate_hospital_item(item):
            out.append(item)
            if len(out) >= limit:
                break
                
    _dbg("regex", f"regex_fallback produced {len(out)} validated items")
    return out

def _dedupe(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ded, seen = [], set()
    for it in items:
        k = (str(it.get("name","")).strip().lower(), str(it.get("address","")).strip().lower())
        if k in seen:
            continue
        seen.add(k); ded.append(it)
    _dbg("dedupe", f"deduped {len(items)} -> {len(ded)}")
    return ded


# ---------------------------
# DSPy program
# ---------------------------

if _DSPY:
    from DSPy.dsp_signatures import ExtractHospitalsSig
    ExtractSignature = ExtractHospitalsSig
else:
    # dummy signature when dspy is not available so module can be imported for testing/fallbacks
    class ExtractSignature:
        pass

class Extractor:
    """
    DSPy-based extractor with verbose debug logs.
    """

    def __init__(
        self,
        model_name: str = None,
        api_key: Optional[str] = None,
        chunk_size: int = 3000,
        overlap: int = 250,
        min_text_len: int = 100,
        max_items_per_chunk: int = 40,
        max_items_regex_fallback: int = 20,
    ):
        self.model_name = model_name or os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        # Try to configure DSPy if available and api_key provided. Otherwise operate in
        # a degraded fallback mode where LLM calls are skipped or delegated to a local
        # extractor (if available).
        if _DSPY and self.api_key:
            try:
                _dbg("init", f"configuring DSPy LM={self.model_name}")
                dspy.configure(lm=dspy.LM(self.model_name, api_key=self.api_key))
                self.predictor = dspy.Predict(ExtractSignature)
            except Exception as e:
                _dbg("init", f"dspy.configure failed: {e}")
                self.predictor = None
        else:
            if not _DSPY:
                _dbg("init", "dspy not available; extractor will run in fallback mode")
            else:
                _dbg("init", "OPENAI_API_KEY missing; extractor will run in fallback mode")
            self.predictor = None

        self.chunk_size = int(chunk_size)
        self.overlap = int(overlap)
        self.min_text_len = int(min_text_len)
        self.max_items_per_chunk = int(max_items_per_chunk)
        self.max_items_regex_fallback = int(max_items_regex_fallback)

        _dbg("init", f"params: chunk_size={self.chunk_size} overlap={self.overlap} "
                     f"min_text_len={self.min_text_len} max_items_per_chunk={self.max_items_per_chunk} "
                     f"max_items_regex_fallback={self.max_items_regex_fallback}")

    def _llm_once(self, goal: str, page: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Call DSPy predictor. Return (items, raw_json_or_empty_dict)."""
        _dbg("llm", f"calling LLM on page fragment len={len(page)}")
        raw: Dict[str, Any] = {}
        # If predictor is not configured (no dspy or no API key), try a local fallback
        if not getattr(self, "predictor", None):
            _dbg("llm", "predictor not configured; attempting local fallback extractor if available")
            try:
                import url_scraper as _us
                try:
                    cand = _us.extract_information(goal, page[:10000], existing_items=None) or []
                    if isinstance(cand, list):
                        raw = {"items": cand}
                    elif isinstance(cand, dict):
                        raw = cand
                    else:
                        raw = {"items": []}
                except Exception as e:
                    _dbg("llm", f"local extractor failed: {e}")
                    raw = {"items": []}
            except Exception as e:
                _dbg("llm", f"no local extractor available: {e}")
                raw = {"items": []}
        else:
            resp = None
            out = None
            try:
                schema = ['name', 'address', 'phone', 'website', 'source_url']
                resp = self.predictor(
                    goal=goal,
                    page_text=page[:10000],
                    schema=schema
                )

                # Handle DSPy response - try multiple ways to access items
                try:
                    # First try to get items from vars(resp)
                    rv = None
                    try:
                        rv = vars(resp)
                    except Exception:
                        rv = None

                    if isinstance(rv, dict):
                        out = rv.get("items", None)
                        _dbg("llm", f"accessed items via vars -> type={type(out)}")
                    else:
                        # Try _response attr if present
                        try:
                            pred_resp = getattr(resp, "_response", None)
                            if isinstance(pred_resp, dict):
                                out = pred_resp.get("items", None)
                                _dbg("llm", f"accessed items via _response -> type={type(out)}")
                                _dbg("llm", f"raw _response: {pred_resp}")
                        except Exception as e:
                            _dbg("llm", f"failed to access response items: {e}")
                except Exception as e:
                    _dbg("llm", f"accessing items failed: {e}")
                    out = None

                # If out is still None, try other representations
                if out is None:
                    if isinstance(resp, (dict, list)):
                        out = resp
                    else:
                        try:
                            # sometimes predictor returns a string-like JSON
                            out = getattr(resp, "items", None)
                        except Exception:
                            out = None

                # resp could also be a primitive or container directly
                # If string, try to parse JSON. If list, try to coerce elements to dicts.
                if isinstance(out, str):
                    _dbg("llm", f"items is str len={len(out)} (attempt JSON parse)")
                    try:
                        raw = json.loads(out)
                    except Exception:
                        # last resort: try ast.literal_eval then wrap
                        try:
                            import ast
                            raw = ast.literal_eval(out)
                        except Exception as e:
                            _dbg("llm", f"string parse failed: {e}")
                            raw = {"items": []}
                elif isinstance(out, list):
                    # out may be a list of dicts or strings. Try to coerce elements to dicts.
                    parsed = []
                    for el in out:
                        if isinstance(el, dict):
                            parsed.append(el)
                        elif isinstance(el, str):
                            try:
                                parsed.append(json.loads(el))
                            except Exception:
                                try:
                                    import ast
                                    parsed.append(ast.literal_eval(el))
                                except Exception:
                                    # skip elements we can't parse
                                    continue
                    if parsed:
                        raw = {"items": parsed}
                    else:
                        # keep original list; coercion will filter non-dicts later
                        raw = {"items": out}
                elif isinstance(out, dict):
                    raw = out
                else:
                    # As a last attempt, if the predictor returned a dict-like resp
                    # with the items under other keys, try to coerce resp itself.
                    if isinstance(resp, dict):
                        raw = resp
                    elif isinstance(resp, list):
                        raw = {"items": resp}
                    else:
                        raw = {"items": []}
            except Exception as e:
                _dbg("llm", f"exception: {e}")
                raw = {"items": []}

            # Post-process raw -> items and return below

        # Common post-processing and return for both fallback and DSPy paths
        try:
            items = _coerce_items(raw)
            # Filter out invalid/incomplete items
            items = [item for item in items if _validate_hospital_item(item)]
            _dbg("llm", f"coerced and validated items={len(items)}")

            # If no items found, capture the full DSPy response object
            # (vars/attributes truncated) so we can inspect what the LM returned
            # when writing artifacts to the DB.
            if not items:
                try:
                    resp_vars = {}
                    try:
                        rv = vars(resp) if 'resp' in locals() and resp is not None else None
                    except Exception:
                        # Fallback to dir-based capture
                        rv = None

                    if isinstance(rv, dict):
                        for k, v in rv.items():
                            try:
                                resp_vars[k] = str(v)
                            except Exception:
                                resp_vars[k] = repr(v)
                    else:
                        if 'resp' in locals() and resp is not None:
                            try:
                                for k in dir(resp):
                                    if k.startswith("__"):
                                        continue
                                    try:
                                        resp_vars[k] = str(getattr(resp, k))
                                    except Exception:
                                        resp_vars[k] = repr(getattr(resp, k))
                            except Exception:
                                resp_vars = {"repr": repr(resp)[:3000]}

                    if not resp_vars:
                        resp_vars = {"repr": repr(resp)[:3000] if 'resp' in locals() and resp is not None else 'no_resp'}

                    raw = {"_dspy_response": resp_vars}
                    _dbg("llm", f"captured _dspy_response keys={list(resp_vars.keys())}")
                    # Also write the captured response to a local file for quick inspection
                    try:
                        dump_dir = "/tmp/extract_dspy"
                        os.makedirs(dump_dir, exist_ok=True)
                        dump_name = f"dspy_resp_{uuid.uuid4().hex}.json"
                        dump_path = os.path.join(dump_dir, dump_name)
                        with open(dump_path, "w", encoding="utf-8") as fh:
                            json.dump(resp_vars, fh, ensure_ascii=False, indent=2)
                        _dbg("llm", f"wrote local dspy dump {dump_path}")
                    except Exception as e:
                        _dbg("llm", f"failed to write local dspy dump: {e}")
                except Exception as e:
                    _dbg("llm", f"failed to capture resp vars: {e}")

            return items, raw
        except Exception as e:
            _dbg("llm", f"post-processing failed: {e}")
            return [], {"items": []}

    def extract(
        self,
        scrapes: List[Dict[str, Any]],
        goal: str,
        *,
        db=None,                 # Firestore adapter with write_step / write_artifact / write_extracted
        plan_id: Optional[str] = None,
        run_id: Optional[str] = None,
        round_idx: int = 1,
        controls: Optional[Dict[str, Any]] = None,
        existing_items: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        _dbg("start", f"extract() called with scrapes={len(scrapes or [])} goal='{goal[:60]}...' round={round_idx}")

        # Allow controls override at call-time
        if controls:
            self.chunk_size = int(controls.get("extract_chunk_size", self.chunk_size))
            self.overlap = int(controls.get("extract_overlap", self.overlap))
            self.min_text_len = int(controls.get("min_extract_text_len", self.min_text_len))
            self.max_items_per_chunk = int(controls.get("max_items_per_chunk", self.max_items_per_chunk))
            self.max_items_regex_fallback = int(controls.get("max_items_regex_fallback", self.max_items_regex_fallback))
            _dbg("controls", f"updated from controls: chunk_size={self.chunk_size} overlap={self.overlap} "
                             f"min_text_len={self.min_text_len} max_items_per_chunk={self.max_items_per_chunk} "
                             f"max_items_regex_fallback={self.max_items_regex_fallback}")

        usable, ex = [], []
        examples = []
        for s in scrapes or []:
            url = s.get("url")
            txt = _coalesce_text(s)
            _dbg("scan", f"url={url} coalesced_len={len(txt)}")
            if txt and len(txt) >= self.min_text_len:
                ss = dict(s); ss["clean_text"] = txt
                usable.append(ss)
                if len(examples) < 5:
                    examples.append({"url": url, "len": len(txt)})
            else:
                _dbg("scan", f"SKIP url={url} (len<{self.min_text_len})")

        _dbg("scan", f"usable pages for extract: {len(usable)} / {len(scrapes or [])}")
        if db:
            db.write_step("extract_input", {
                "round": round_idx,
                "scrapes_total": len(scrapes or []),
                "usable_for_extract": len(usable),
                "examples": examples
            })

        # loop pages
        for idx, s in enumerate(usable):
            url = s.get("url")
            txt = s["clean_text"]
            _dbg("page", f"[{idx+1}/{len(usable)}] extracting url={url} len={len(txt)}")

            page_items: List[Dict[str, Any]] = []
            chunk_count = 0

            for c_i, ch in enumerate(_chunks(txt, size=self.chunk_size, overlap=self.overlap)):
                chunk_count += 1
                _dbg("chunk", f"url={url} chunk#{c_i} len={len(ch)} -> LLM")
                items, raw = self._llm_once(goal, ch)

                # audit raw JSON for first chunk of each page
                if db and c_i == 0:
                    doc_key = f"extract_{round_idx:03d}_{(hash(url) & 0xffff):04x}_raw"
                    try:
                        # Serialize raw output to a JSON string for safe storage in Firestore
                        try:
                            raw_for_db = json.dumps(raw, default=str, ensure_ascii=False)
                        except Exception:
                            raw_for_db = str(raw)[:3000]

                        db.write_artifact(doc_key, {
                            "url": url,
                            "raw": raw_for_db
                        })
                        _dbg("audit", f"wrote artifact {doc_key}")
                    except Exception as e:
                        _dbg("audit", f"write_artifact failed: {e}")

                _dbg("chunk", f"url={url} chunk#{c_i} items={len(items)}")
                if items:
                    cap = self.max_items_per_chunk
                    page_items.extend(items[:cap])
                    _dbg("chunk", f"url={url} appended {min(len(items), cap)} items (total_page_items={len(page_items)})")

            _dbg("page", f"url={url} chunk_count={chunk_count} page_items_after_llm={len(page_items)}")

            # regex fallback if page looks relevant
            if not page_items and "hospital" in txt.lower():
                fb = _regex_fallback(txt, self.max_items_regex_fallback)
                page_items.extend(fb)
                _dbg("page", f"url={url} regex_fallback_added={len(fb)}")

            for it in page_items:
                it.setdefault("source_url", url)

            _dbg("page", f"url={url} final_page_items={len(page_items)}")
            ex.extend(page_items)

        _dbg("aggregate", f"collected items before dedupe: {len(ex)}")
        out = _dedupe(ex)
        _dbg("aggregate", f"final extracted items: {len(out)}")

        if db:
            try:
                db.write_extracted(round_idx, out)
                db.write_step("extract", {"round": round_idx, "extracted": len(out)})
                _dbg("db", f"wrote extracted ({len(out)}) and step summary")
            except Exception as e:
                _dbg("db", f"write_extracted/write_step failed: {e}")

        _dbg("end", f"extract() returning {len(out)} items")
        return out
