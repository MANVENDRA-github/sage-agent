"""Memory + web tools the LLM can call.

Three tools are bound to the model and drive the ReAct loop:

- ``save_memory(content)`` — record a new fact / preference / event about the
  user. The *schema* of this tool is what the model sees; the real save
  (conflict-resolution judge, type classification, DELETE-then-INSERT) is
  performed by the graph's tool-execution node in ``graph.py``, not by the
  thin body below. The body is retained as the standalone blind-append
  fallback and to define the tool's signature for ``bind_tools``.
- ``search_memory(query)`` — recall what is already known about the user via
  the existing semantic search in ``store.py``. Unlike ``save_memory`` this is
  a genuine wrapper: its body performs the same ``store.asearch`` the graph
  node executes, so there is no second copy of retrieval logic.
- ``web_search(query)`` — look up current / external facts the user did not
  provide and aren't in memory (Phase 2). Wraps the keyless DuckDuckGo
  ``ddgs`` library and returns a short readable summary of the top results.
  It needs no store or user_id, so it has no ``InjectedToolArg``s.

Retrieval is model-driven now: there is no forced retrieve step. The model
calls ``search_memory`` / ``web_search`` when it judges that recall or an
external lookup would help.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Annotated

from ddgs import DDGS
from ddgs.exceptions import DDGSException
from langchain_core.tools import InjectedToolArg, tool
from langgraph.store.base import BaseStore

from sage_agent.store import memory_namespace

# Top-k for a model-issued recall. Mirrors the old forced-retrieve RETRIEVAL_K.
SEARCH_LIMIT = 5

# Web search: how many results to summarise and how long each snippet may be.
WEB_SEARCH_MAX_RESULTS = 3
WEB_SNIPPET_CHARS = 200


@tool
async def save_memory(
    content: str,
    *,
    user_id: Annotated[str, InjectedToolArg],
    store: Annotated[BaseStore, InjectedToolArg],
) -> str:
    """Save a piece of information about the user to long-term memory.

    Call this whenever the user shares a fact, preference, or notable event
    about themselves that should be remembered for future conversations.

    Args:
        content: A short third-person statement, e.g. "User's name is Aman".
    """
    mem_id = str(uuid.uuid4())
    await store.aput(memory_namespace(user_id), key=mem_id, value={"content": content})
    return f"Saved memory {mem_id}"


@tool
async def search_memory(
    query: str,
    *,
    user_id: Annotated[str, InjectedToolArg],
    store: Annotated[BaseStore, InjectedToolArg],
) -> str:
    """Search your long-term memory for what you already know about the user.

    Call this to recall facts, preferences, or past events before you answer —
    for example when the user refers to something they told you earlier, asks
    what you remember, or whenever a known detail would make your reply more
    accurate or personal. Each turn starts with no memories loaded, so search
    when you need them.

    Args:
        query: What to look up, in natural language — e.g. "user's name" or
            "food preferences".
    """
    items = await store.asearch(
        memory_namespace(user_id), query=query, limit=SEARCH_LIMIT
    )
    if not items:
        return "No relevant memories found."
    lines = [
        f"- [{item.value.get('type', 'fact')}] {item.value.get('content', '')}"
        for item in items
    ]
    return "Relevant memories:\n" + "\n".join(lines)


def _run_ddgs_text(query: str, max_results: int) -> list[dict]:
    """Run a synchronous DuckDuckGo text search via ``ddgs``.

    ``ddgs`` (the renamed ``duckduckgo-search``) RAISES ``DDGSException`` rather
    than returning an empty list when a search yields nothing. We translate the
    specific "no results found" case into an empty list — a normal, non-error
    outcome the caller renders as "no results". Every other ``DDGSException``
    (rate limit, timeout, transient network failure) is re-raised so the graph's
    retry-once-then-graceful-ToolMessage path can handle it.
    """
    try:
        return DDGS().text(query, max_results=max_results)
    except DDGSException as exc:
        if "no results" in str(exc).lower():
            return []
        raise


@tool
async def web_search(query: str) -> str:
    """Search the web for current or external information you don't already know.

    Use this for facts the user did NOT give you and that aren't about the user
    — current events, news, weather, prices, public facts, anything past your
    training cutoff. Do NOT use it for things about the user (use search_memory)
    or for things you can answer directly from your own knowledge.

    Args:
        query: A concise web search query, e.g. "current UK prime minister".
    """
    results = await asyncio.to_thread(_run_ddgs_text, query, WEB_SEARCH_MAX_RESULTS)
    if not results:
        return f"No web results found for {query!r}."

    lines = []
    for i, item in enumerate(results[:WEB_SEARCH_MAX_RESULTS], start=1):
        title = (item.get("title") or "").strip()
        snippet = (item.get("body") or "").strip()
        if len(snippet) > WEB_SNIPPET_CHARS:
            snippet = snippet[:WEB_SNIPPET_CHARS].rstrip() + "..."
        url = (item.get("href") or "").strip()
        lines.append(f"{i}. {title}\n   {snippet}\n   {url}")
    return f"Top web results for {query!r}:\n" + "\n".join(lines)
