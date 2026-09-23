"""Shared OpenAI-compatible client configuration for ARS LLM providers."""

from __future__ import annotations

import os
from pathlib import Path

from openai import AsyncOpenAI
from dotenv import load_dotenv


_dotenv_path = Path(__file__).resolve().parent / "searxng" / ".env"
if _dotenv_path.exists():
    load_dotenv(_dotenv_path)


def get_async_client() -> tuple[AsyncOpenAI, str]:
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in environment")
        return (
            AsyncOpenAI(
                api_key=api_key,
                base_url=os.getenv(
                    "GEMINI_OPENAI_BASE_URL",
                    "https://generativelanguage.googleapis.com/v1beta/openai/",
                ),
            ),
            os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
        )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment")
    return AsyncOpenAI(api_key=api_key), os.getenv("OPENAI_MODEL", "gpt-5-mini")
