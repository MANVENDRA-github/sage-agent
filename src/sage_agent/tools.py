"""Memory tools the LLM can call.

Two tools are bound to the model and drive the ReAct loop:

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

Retrieval is model-driven now: there is no forced retrieve step. The model
calls ``search_memory`` when it judges that recall would help.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from langchain_core.tools import InjectedToolArg, tool
from langgraph.store.base import BaseStore

from sage_agent.store import memory_namespace

# Top-k for a model-issued recall. Mirrors the old forced-retrieve RETRIEVAL_K.
SEARCH_LIMIT = 5


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
