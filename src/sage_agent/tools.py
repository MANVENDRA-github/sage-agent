"""Memory tools available to the LLM.

Phase 1 ships a single, intentionally-degraded save_memory: it appends with
no update semantics and no memory_id. The upstream template's upsert_memory
supports updates by passing memory_id — that capability is what Week 2's
conflict-resolution work will (re)introduce, on top of measured baseline
numbers showing how much blind-append actually hurts.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from langchain_core.tools import InjectedToolArg, tool
from langgraph.store.base import BaseStore

from sage_agent.store import memory_namespace


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
