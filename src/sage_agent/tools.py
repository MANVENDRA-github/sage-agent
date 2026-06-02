"""Memory + web + goal tools the LLM can call.

Four tools are bound to the model and drive the ReAct loop:

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
- ``manage_goal(action, ...)`` — set / list / update the user's personal goals
  (Phase 3). Goals live in the SAME store as ``type="goal"`` memories with a
  ``status`` and ``created_at``. This is the ONLY path that writes a goal-type
  memory; save_memory's fact/preference/episodic auto-classifier never emits
  "goal". ``update`` reuses the same DELETE-then-INSERT pattern as a save
  replace, so a status change does not duplicate the goal.

Retrieval is model-driven now: there is no forced retrieve step. The model
calls ``search_memory`` / ``web_search`` / ``manage_goal`` when it judges that
recall, an external lookup, or goal-tracking would help.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Annotated

from ddgs import DDGS
from ddgs.exceptions import DDGSException
from langchain_core.tools import InjectedToolArg, tool
from langgraph.store.base import BaseStore

from sage_agent.store import list_memories, memory_namespace

# Top-k for a model-issued recall. Mirrors the old forced-retrieve RETRIEVAL_K.
SEARCH_LIMIT = 5

# Web search: how many results to summarise and how long each snippet may be.
WEB_SEARCH_MAX_RESULTS = 3
WEB_SNIPPET_CHARS = 200

# Goals are stored as this memory type. Reached ONLY via manage_goal — never by
# save_memory's auto-classifier (which is constrained to fact/preference/episodic).
GOAL_TYPE = "goal"
# How many neighbours to pull before filtering to goal-type when matching the
# goal an update refers to.
GOAL_MATCH_LIMIT = 10


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


def _now_iso() -> str:
    """UTC timestamp in ISO-8601 — same format the store stamps on every put."""
    return datetime.now(timezone.utc).isoformat()


@tool
async def manage_goal(
    action: str,
    *,
    user_id: Annotated[str, InjectedToolArg],
    store: Annotated[BaseStore, InjectedToolArg],
    goal: str = "",
    status: str = "",
    new_goal: str = "",
) -> str:
    """Track the user's personal goals (aims / intentions they want to pursue).

    This is the ONLY way a goal is recorded — do not use save_memory for goals.

    Args:
        action: One of "set", "list", "update".
            - "set": record a NEW goal the user states. Put the goal text in
              `goal`. It is stored with status "active".
            - "list": return all of the user's goals with their statuses.
              No other arguments needed.
            - "update": change an existing goal. Use `goal` to describe WHICH
              goal (matched to the closest existing one), `status` for the new
              status (e.g. "done", "active", "abandoned"), and optionally
              `new_goal` to reword the goal text.
        goal: The goal text (for "set") or which goal to match (for "update").
        status: The new status (for "update"), e.g. "done".
        new_goal: Optional new wording for the goal text (for "update").
    """
    namespace = memory_namespace(user_id)
    act = (action or "").strip().lower()

    if act == "list":
        goals = [m for m in list_memories(store, user_id) if m.get("type") == GOAL_TYPE]
        if not goals:
            return "The user has no goals tracked yet."
        lines = [
            f"- {g.get('content', '')} [status: {g.get('status', 'active')}]"
            for g in goals
        ]
        return "Current goals:\n" + "\n".join(lines)

    if act == "set":
        text = (goal or new_goal or "").strip()
        if not text:
            return "No goal text was provided to set."
        new_id = str(uuid.uuid4())
        await store.aput(
            namespace,
            key=new_id,
            value={
                "content": text,
                "type": GOAL_TYPE,
                "status": "active",
                "created_at": _now_iso(),
            },
        )
        return f"Goal set: {text!r} (status=active)."

    if act == "update":
        match_text = (goal or new_goal or "").strip()
        if not match_text:
            return "No goal was specified to update."
        # Reuse the existing semantic search, then keep only goal-type hits.
        candidates = await store.asearch(
            namespace, query=match_text, limit=GOAL_MATCH_LIMIT
        )
        goal_hits = [c for c in candidates if c.value.get("type") == GOAL_TYPE]
        if not goal_hits:
            return f"No matching goal was found for {match_text!r} to update."
        target = goal_hits[0]  # highest-scored goal
        old = target.value
        new_status = (status or "").strip().lower() or old.get("status", "active")
        new_content = (new_goal or "").strip() or old.get("content", "")
        # DELETE-then-INSERT (same pattern as a save replace) — a new UUID, so
        # the goal is updated in place rather than duplicated. Preserve the
        # original creation timestamp across the update.
        await store.adelete(namespace, key=target.key)
        new_id = str(uuid.uuid4())
        value = {"content": new_content, "type": GOAL_TYPE, "status": new_status}
        created = old.get("created_at")
        if created is not None:
            value["created_at"] = created
        await store.aput(namespace, key=new_id, value=value)
        return f"Goal updated: {new_content!r} (status={new_status})."

    return (
        f"Unknown manage_goal action {action!r}. "
        "Use action='set', 'list', or 'update'."
    )
