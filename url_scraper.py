import os
import json
import trafilatura
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright
from playwright._impl._errors import Error as PlaywrightError
from openai import OpenAI
from dotenv import find_dotenv, load_dotenv
import tiktoken

dotenv_path = find_dotenv()
if dotenv_path:
    # Load environment variables from .env file 
    load_dotenv(dotenv_path)

_client = OpenAI(api_key=os.getenv("OPENAI_RANDY_KEY"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-2025-04-14")

def _chunk_text_for_model(text: str, max_tokens: int, model: str):
    """
    Yield substrings of `text` each ≲ max_tokens (by token count), 
    using the model’s own tokenizer.
    """
    enc = tiktoken.encoding_for_model(model)
    toks = enc.encode(text)
    for i in range(0, len(toks), max_tokens):
        yield enc.decode(toks[i : i+max_tokens])


def _is_valid_url(url: str) -> bool:
    p = urlparse(url or "")
    return p.scheme in ("http", "https") and bool(p.netloc)


def read_website_fast(url: str) -> str:
    """Fast extraction using Trafilatura; skips invalid URLs."""
    if not _is_valid_url(url):
        return ""
    downloaded = trafilatura.fetch_url(url, timeout=10)
    if not downloaded:
        return ""
    text = trafilatura.extract(downloaded, include_tables=False,
                                include_comments=False,
                                include_formatting=True)
    return text or ""


def read_website_full(url: str) -> str:
    """
    Fallback full-browser scrape via Playwright; skips invalid URLs
    and swallows Playwright errors.
    """
    if not _is_valid_url(url):
        return ""
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(url, timeout=30000)
            content = page.content()
            browser.close()
    except PlaywrightError as e:
        print(f"⚠️ Playwright error for URL {url!r}: {e}")
        return ""
    except Exception as e:
        print(f"⚠️ Full scrape exception for URL {url!r}: {e}")
        return ""

    # Extract main text from rendered HTML
    text = trafilatura.extract(content,
                                include_tables=False,
                                include_comments=False,
                                include_formatting=True,
                                url=url)
    return text or ""


def extract_information(plan_goal, scraped_text, existing_items):
    """
    Split the scraped_text into 3000-token chunks and 
    call GPT-4 on each chunk, then dedupe the results.
    """
    chunk_size = 3000
    all_new = []

    for chunk in _chunk_text_for_model(scraped_text, chunk_size):
        prompt = f"""
                    You are extracting information for the research goal:
                    "{plan_goal}"

                    The existing items are:
                    {json.dumps(existing_items, indent=2)}

                    Here is a portion of the scraped webpage text:
                    \"\"\"{chunk}\"\"\"

                    Extract and return a JSON array of new items relevant to the goal.
                    Each item should be an object with keys "name" and "address".
                    Do NOT include items already in the existing items list.
                    Output raw JSON only.
                """
        resp = _client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role":"system","content":"You extract structured data from text."},
                {"role":"user",  "content":prompt}
            ],
            temperature=0.0
        )
        try:
            chunk_items = json.loads(resp.choices[0].message.content)
        except json.JSONDecodeError:
            continue
        all_new.extend(chunk_items)

    # final dedupe
    seen, unique = set(), []
    for itm in all_new:
        key = json.dumps(itm, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(itm)
    return unique