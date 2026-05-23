"""Memory store wrapper.

Uses langgraph's InMemoryStore — a BaseStore subclass that's literally a dict
under the hood, but exposes the interface we'll keep when swapping in a
Chroma-backed store in Week 2. Namespace convention is ("memories", user_id)
to match the upstream template.
"""

from __future__ import annotations

from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

MEMORY_NAMESPACE = "memories"


def make_store() -> BaseStore:
    return InMemoryStore()


def memory_namespace(user_id: str) -> tuple[str, str]:
    return (MEMORY_NAMESPACE, user_id)


def list_memories(store: BaseStore, user_id: str) -> list[dict]:
    items = store.search(memory_namespace(user_id))
    return [{"key": item.key, **item.value} for item in items]
