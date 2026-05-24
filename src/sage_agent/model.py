"""LLM factory.

Uses ChatOpenAI pointed at OpenRouter's OpenAI-compatible endpoint because
LangChain's init_chat_model doesn't natively route to OpenRouter. The model
slug is whatever OpenRouter exposes — defaulting to openai/gpt-oss-120b on
the free tier, which has the strongest tool-calling among current free
options.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

DEFAULT_MODEL = "openai/gpt-oss-120b:free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def get_model(model_name: str | None = None, temperature: float = 0.0) -> ChatOpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return ChatOpenAI(
        model=model_name or os.environ.get("MODEL_NAME", DEFAULT_MODEL),
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        temperature=temperature,
    )
