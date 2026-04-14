"""
Node 6: Deduplicate
Removes duplicates from extracted dataset using layered matching.
"""

import csv
import io
import json
import os
import re
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
from openai import AsyncOpenAI
# GEMINI - Temporarily commented out
# import google.generativeai as genai
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from nodes.base import BaseNode

if TYPE_CHECKING:
    from algorithm import ResearchState, ProgressTracker

_openai_client = None
_openai_model = None

# GEMINI - Temporarily commented out
# _gemini_model = None


def get_openai_client():
    global _openai_client, _openai_model
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")
        _openai_client = AsyncOpenAI(api_key=api_key)
        _openai_model = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    return _openai_client, _openai_model

# GEMINI - Temporarily commented out
# def get_gemini_model():
#     global _gemini_model
#     if _gemini_model is None:
#         api_key = os.getenv("GEMINI_API_KEY")
#         if not api_key:
#             raise ValueError("GEMINI_API_KEY not set in environment")
#         genai.configure(api_key=api_key)
#         model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
#         _gemini_model = genai.GenerativeModel(model_name)
#     return _gemini_model

# GEMINI - Temporarily commented out
# async def _call_gemini_with_retry(model, prompt: str, max_retries: int = 3) -> str:
#     """Call Gemini API with automatic retry on 429 rate limit errors"""
#     for attempt in range(max_retries):
#         try:
#             response = await model.generate_content_async(prompt)
#             return response.text
#         except Exception as e:
#             error_code = getattr(e, 'status_code', None)
#             if error_code == 429 or "429" in str(e) or "quota" in str(e).lower():
#                 if attempt < max_retries - 1:
#                     retry_delay = 15
#                     try:
#                         if hasattr(e, 'retry_delay') and hasattr(e.retry_delay, 'seconds'):
#                             retry_delay = e.retry_delay.seconds + 2
#                         elif "Please retry in" in str(e):
#                             match = re.search(r'Please retry in ([0-9.]+)s', str(e))
#                             if match:
#                                 retry_delay = float(match.group(1)) + 2
#                     except:
#                         retry_delay = (2 ** attempt) + 15
#                     
#                     print(f"⏳ Rate limit hit (429). Retrying in {retry_delay:.1f}s... (Attempt {attempt + 1}/{max_retries})")
#                     await asyncio.sleep(retry_delay)
#                     continue
#                 else:
#                     print(f"✗ Rate limit exceeded. Max retries ({max_retries}) reached.")
#                     raise
#             raise


def _normalize_text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_phone(value: object, default_region: str = "CA") -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    raw = str(value).strip()
    if not raw:
        return ""
    digits_only = re.sub(r"\D", "", raw)
    try:
        import phonenumbers

        parsed = phonenumbers.parse(raw, default_region)
        if not phonenumbers.is_valid_number(parsed):
            return digits_only
        return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
    except Exception:
        return digits_only


def _normalize_hard(value: object, column_name: str, default_region: str) -> str:
    if "phone" in column_name.lower():
        return _normalize_phone(value, default_region)
    return _normalize_text(value)


def _column_score(a: str, b: str) -> Optional[float]:
    if not a or not b:
        return None
    jw = JaroWinkler.similarity(a, b)
    ts = fuzz.token_set_ratio(a, b) / 100.0
    return max(jw, ts)


def _build_candidate_pairs(embeddings: np.ndarray, threshold: float) -> set[tuple[int, int]]:
    radius = 1.0 - threshold
    nn = NearestNeighbors(metric="cosine", radius=radius)
    nn.fit(embeddings)
    distances, indices = nn.radius_neighbors(embeddings, radius=radius)
    pairs: set[tuple[int, int]] = set()
    for i, neighbors in enumerate(indices):
        for j in neighbors:
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            pairs.add((a, b))
    return pairs


async def _llm_judge(
    client: AsyncOpenAI,
    model: str,
    # GEMINI - Temporarily commented out
    # gemini_model,
    pairs: list[dict],
    batch_size: int = 10,
) -> list[bool]:
    decisions: list[bool] = []
    
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        payload = [
            {"left": item["left"], "right": item["right"]}
            for item in batch
        ]
        
        system_prompt = (
            "You are a deduplication judge. Decide if two records describe the same physical facility. "
            "Return JSON only."
        )
        user_prompt = (
            "For each pair, decide if they refer to the same physical facility. "
            "If either side has missing values, use best judgment. "
            "Reply with JSON in the form {\"decisions\":[true,false,...]} in the same order.\n"
            f"Pairs: {json.dumps(payload, ensure_ascii=True)}"
        )

        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = (response.choices[0].message.content or "").strip()

        # GEMINI - Temporarily commented out
        # content = (await _call_gemini_with_retry(gemini_model, prompt)).strip()
        try:
            parsed = json.loads(content)
            batch_decisions = parsed.get("decisions", [])
        except json.JSONDecodeError:
            batch_decisions = []

        if len(batch_decisions) != len(batch):
            batch_decisions = [False] * len(batch)

        decisions.extend(bool(x) for x in batch_decisions)

    return decisions


class DeduplicateNode(BaseNode):
    """
    Removes duplicate records from extracted dataset using LLM.
    
    Input State Keys:
        - extracted_items: List of extracted records
    
    Output State Keys:
        - final_dataset: Deduplicated list of records
    """
    
    async def execute(self, state: "ResearchState", progress: "ProgressTracker") -> "ResearchState":
        """Remove duplicates from extracted dataset"""

        items = state.get("extracted_items", [])
        hard_id_cols = state.get("hard_identifier_columns", []) or []
        soft_id_cols = state.get("soft_identifier_columns", []) or []
        openai_client, openai_model = get_openai_client()
        # GEMINI - Temporarily commented out
        # gemini_model = get_gemini_model()

        progress.update(
            "🗑️  Deduplication Starting",
            f"Removing duplicates from {len(items)} records..."
        )

        if not items:
            state["final_dataset"] = []
            progress.update("🗑️  Deduplication Complete", "No items to deduplicate")
            return state

        buffer = io.StringIO()
        fieldnames = sorted({k for item in items if isinstance(item, dict) for k in item.keys()})
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            if isinstance(item, dict):
                writer.writerow({k: item.get(k, "") for k in fieldnames})
        buffer.seek(0)

        df = pd.read_csv(buffer)
        raw_rows = len(df)

        if not soft_id_cols:
            state["final_dataset"] = items
            progress.update(
                "🗑️  Deduplication Skipped",
                "No soft identifiers selected; keeping all records."
            )
            return state

        for col in soft_id_cols:
            if col not in df.columns:
                df[col] = ""
            df[f"__norm_{col}"] = df[col].apply(_normalize_text)

        hard_norm_cols = []
        for col in hard_id_cols:
            if col in df.columns:
                norm_col = f"__norm_{col}"
                df[norm_col] = df[col].apply(lambda v: _normalize_hard(v, col, "CA"))
                hard_norm_cols.append(norm_col)

        post_hard_rows = raw_rows
        if hard_norm_cols:
            has_all_hard = df[hard_norm_cols].astype(bool).all(axis=1)
            composite_key = df[hard_norm_cols].astype(str).agg("||".join, axis=1)
            df["__hard_key"] = ""
            df.loc[has_all_hard, "__hard_key"] = composite_key[has_all_hard]
            hard_dupes = df[has_all_hard].duplicated(subset=["__hard_key"], keep="first")
            drop_idx = df[has_all_hard][hard_dupes].index
            df = df.drop(index=drop_idx).reset_index(drop=True)
        post_hard_rows = len(df)

        if hard_norm_cols:
            progress.update(
                "🧹 Hard-ID Dedupe",
                f"Rows: {raw_rows} -> {post_hard_rows}",
                {"input_rows": raw_rows, "post_hard_rows": post_hard_rows}
            )

        candidate_pairs = set()
        if soft_id_cols:
            df["__soft_text"] = df[[f"__norm_{c}" for c in soft_id_cols]].agg(" ".join, axis=1).str.strip()
            non_empty_mask = df["__soft_text"].astype(bool)
            if non_empty_mask.any():
                model = SentenceTransformer("all-MiniLM-L6-v2")
                embeddings = model.encode(
                    df["__soft_text"].tolist(),
                    batch_size=64,
                    show_progress_bar=True,
                    normalize_embeddings=True,
                )
                embeddings = np.asarray(embeddings)
                candidate_pairs = _build_candidate_pairs(embeddings, 0.80)
        progress.update(
            "🧠 Vector Blocking",
            f"Candidate pairs: {len(candidate_pairs)}",
            {"candidate_pairs": len(candidate_pairs)}
        )

        duplicates: set[tuple[int, int]] = set()
        gray_pairs: list[tuple[int, int]] = []

        for i, j in candidate_pairs:
            if not df.at[i, "__soft_text"] or not df.at[j, "__soft_text"]:
                gray_pairs.append((i, j))
                continue
            scores = []
            missing_data = False
            for col in soft_id_cols:
                a = df.at[i, f"__norm_{col}"]
                b = df.at[j, f"__norm_{col}"]
                score = _column_score(a, b)
                if score is None:
                    missing_data = True
                    break
                scores.append(score)
            if missing_data or not scores:
                gray_pairs.append((i, j))
                continue
            final_score = min(scores)
            if final_score >= 0.90:
                duplicates.add((i, j))
            elif final_score < 0.60:
                continue
            else:
                gray_pairs.append((i, j))

        progress.update(
            "🔎 String Verify",
            f"Auto-merge: {len(duplicates)} | Gray zone: {len(gray_pairs)}",
            {"auto_merge_pairs": len(duplicates), "gray_zone_pairs": len(gray_pairs)}
        )

        llm_duplicates = 0
        if gray_pairs:
            pair_payload = []
            payload_cols = [c for c in (soft_id_cols + hard_id_cols) if c in df.columns]
            for i, j in gray_pairs:
                if payload_cols:
                    left = df.loc[i, payload_cols].to_dict()
                    right = df.loc[j, payload_cols].to_dict()
                else:
                    left = df.loc[i].to_dict()
                    right = df.loc[j].to_dict()
                pair_payload.append({
                    "left": left,
                    "right": right,
                    "index_pair": (i, j),
                })
            try:
                decisions = await _llm_judge(openai_client, openai_model, pair_payload)
                # GEMINI - Temporarily commented out
                # decisions = await _llm_judge(gemini_model, pair_payload)
                for payload, is_dup in zip(pair_payload, decisions):
                    if is_dup:
                        duplicates.add(payload["index_pair"])
                        llm_duplicates += 1
            except Exception as e:
                print(f"⚠️  LLM judge error: {e}")

        if gray_pairs:
            progress.update(
                "🤖 LLM Judge",
                f"Confirmed duplicates: {llm_duplicates}",
                {"llm_duplicates": llm_duplicates}
            )

        parent = list(range(len(df)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i, j in duplicates:
            union(i, j)

        seen_roots: set[int] = set()
        keep_indices: list[int] = []
        for idx in range(len(df)):
            root = find(idx)
            if root not in seen_roots:
                seen_roots.add(root)
                keep_indices.append(idx)

        df_deduped = df.loc[keep_indices].drop(columns=[c for c in df.columns if c.startswith("__")])
        state["final_dataset"] = df_deduped.to_dict(orient="records")

        final_count = len(state["final_dataset"])
        progress.update(
            "🗑️  Deduplication Complete",
            f"Final dataset: {final_count} unique records (removed {raw_rows - final_count} duplicates)",
            {
                "original_count": raw_rows,
                "post_hard_count": post_hard_rows,
                "final_count": final_count,
                "duplicates_removed": raw_rows - final_count
            }
        )

        return state
