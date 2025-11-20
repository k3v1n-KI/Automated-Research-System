# url_scraper.py
import os
import json
import time
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

import trafilatura
from playwright.sync_api import sync_playwright
from playwright._impl._errors import Error as PlaywrightError

# Optional: OpenAI-based extractor utilities you already had
import tiktoken
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

import requests
from io import BytesIO
from pdfminer.high_level import extract_text as pdf_extract_text

dotenv_path = find_dotenv()
if dotenv_path:
    load_dotenv(dotenv_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# ------------------------ helpers ------------------------

def _looks_like_pdf_url(url: str) -> bool:
    u = (url or "").lower()
    return u.endswith(".pdf")

def _is_pdf_content(resp: requests.Response) -> bool:
    ctype = (resp.headers.get("Content-Type") or "").lower()
    return "application/pdf" in ctype

def fetch_pdf_text(url: str, timeout: int = 20_000) -> str:
    try:
        r = requests.get(url, timeout=timeout/1000, stream=True, headers=DEFAULT_HEADERS, verify=False)
        r.raise_for_status()
        if not _is_pdf_content(r) and not _looks_like_pdf_url(url):
            return ""
        data = BytesIO(r.content)
        # pdfminer can be slow on huge files; you can add size guards if needed
        text = pdf_extract_text(data) or ""
        return text
    except Exception as e:
        print(f"⚠️ PDF fetch/extract error for {url!r}: {e}")
        return ""

def _is_valid_url(url: str) -> bool:
    try:
        p = urlparse(url or "")
        return p.scheme in ("http", "https") and bool(p.netloc)
    except Exception:
        return False

def _chunk_text_for_model(text: str, max_tokens: int, model: str = OPENAI_MODEL):
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    toks = enc.encode(text or "")
    for i in range(0, len(toks), max_tokens):
        yield enc.decode(toks[i : i + max_tokens])


# ------------------------ fast vs full read ------------------------

def read_website_fast(url: str) -> str:
    """Fast extraction with Trafilatura (no browser)."""
    if not _is_valid_url(url):
        return ""
    try:
        downloaded = trafilatura.fetch_url(url, no_ssl=True)  # be tolerant; Playwright will retry anyway
        if not downloaded:
            return ""
        text = trafilatura.extract(
            downloaded,
            include_tables=False,
            include_comments=False,
            include_formatting=True,
            favor_recall=True,
            url=url,
        )
        return text or ""
    except Exception:
        return ""


def read_website_full(url: str, timeout_ms: int = 30000) -> Dict[str, str]:
    """
    Full-browser scrape via Playwright with HTTPS errors ignored.
    Returns {"html": "...", "text": "..."} (text extracted with Trafilatura).
    """
    if not _is_valid_url(url):
        return {"html": "", "text": ""}

    html = ""
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            # IMPORTANT: ignore HTTPS errors to avoid your SSL loop
            context = browser.new_context(ignore_https_errors=True)
            page = context.new_page()
            page.goto(url, timeout=timeout_ms, wait_until="load")
            html = page.content()
            context.close()
            browser.close()
    except PlaywrightError as e:
        print(f"⚠️ Playwright error for URL {url!r}: {e}")
        return {"html": "", "text": ""}
    except Exception as e:
        print(f"⚠️ Full scrape exception for URL {url!r}: {e}")
        return {"html": "", "text": ""}

    try:
        text = trafilatura.extract(
            html,
            include_tables=False,
            include_comments=False,
            include_formatting=True,
            favor_recall=True,
            url=url,
        ) or ""
    except Exception:
        text = ""

    return {"html": html or "", "text": text or ""}


# ------------------------ public API used by dispatcher ------------------------

def scrape_one(url: str, sleep_sec: float = 0.0) -> Dict[str, Any]:
    if sleep_sec > 0:
        time.sleep(sleep_sec)

    if not _is_valid_url(url):
        return {"url": url, "clean_text": "", "html": ""}

    # 0) PDF path first
    if _looks_like_pdf_url(url):
        pdf_text = fetch_pdf_text(url)
        if pdf_text:
            return {"url": url, "clean_text": pdf_text, "html": ""}

    # 1) fast HTML path
    text = read_website_fast(url)
    if text and len(text.strip()) >= 200:
        return {"url": url, "clean_text": text, "html": ""}

    # 1b) if fast path empty, check if it’s actually a PDF by headers
    try:
        head = requests.head(url, allow_redirects=True, timeout=10)
        if _is_pdf_content(head):
            pdf_text = fetch_pdf_text(url)
            if pdf_text:
                return {"url": url, "clean_text": pdf_text, "html": ""}
    except Exception:
        pass

    # 2) browser fallback (ignore HTTPS errors)
    full = read_website_full(url)
    return {"url": url, "clean_text": full.get("text", "") or "", "html": full.get("html", "") or ""}



def scrape_many(urls: List[str], throttle_ms: int = 0) -> List[Dict[str, Any]]:
    """
    Scrape a list of URLs, returning a list of {"url","clean_text","html"}.
    This is what TaskDispatcher.Services.execute_scrape() looks for.
    """
    out: List[Dict[str, Any]] = []
    sleep_sec = max(0.0, throttle_ms / 1000.0)
    for u in urls or []:
        try:
            row = scrape_one(u, sleep_sec=sleep_sec)
            out.append(row)
        except Exception as e:
            print(f"⚠️ scrape_many error for {u!r}: {e}")
            out.append({"url": u, "clean_text": "", "html": ""})
    return out


# ------------------------ optional LLM extractor (kept from your file) ------------------------

def extract_information(plan_goal: str, scraped_text: str, existing_items: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """
    Split the scraped_text into ~3000-token chunks and call the model on each chunk,
    then dedupe the results.
    """
    if not _client:
        return []

    chunk_size = 3000
    all_new: List[Dict[str, Any]] = []
    existing_items = existing_items or []

    for chunk in _chunk_text_for_model(scraped_text, chunk_size):
        # Keep the prompt short and JSON-native to reduce parse errors
        user_payload = {
            "goal": plan_goal,
            "existing_items": existing_items[:50],  # keep context small
            "text": chunk,
            "schema": ["name", "address", "phone", "website"]
        }
        resp = _client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "Extract new entities as JSON only. No commentary."},
                {"role": "user",   "content": json.dumps(user_payload, ensure_ascii=False)}
            ],
            temperature=0.0
        )
        raw = (resp.choices[0].message.content or "").strip()
        # The model might output a JSON array; accept either array or {"items":[...]}
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                cand = parsed
            elif isinstance(parsed, dict) and isinstance(parsed.get("items"), list):
                cand = parsed["items"]
            else:
                cand = []
        except Exception:
            cand = []
        for it in cand:
            if isinstance(it, dict):
                all_new.append(it)

    # final dedupe
    seen, unique = set(), []
    for itm in all_new:
        key = json.dumps(itm, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(itm)
    return unique
