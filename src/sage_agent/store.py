"""Chroma-backed memory store.

``ChromaStore`` is a ``BaseStore`` subclass that uses Chroma plus
sentence-transformers ``all-MiniLM-L6-v2`` to provide semantic search over
user memories. The langgraph store contract is satisfied by implementing
``batch`` / ``abatch``; high-level ops (``get`` / ``put`` / ``search`` /
``delete`` and their async siblings) dispatch through them automatically.

A single Chroma collection holds memories for all users, with namespace
encoded as metadata and scoped via a where-filter on every op. One-shared-
collection beats one-per-namespace because the eval harness creates 50
stores per run and Chroma's per-collection overhead adds up.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

import chromadb
from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
)

MEMORY_NAMESPACE = "memories"
COLLECTION_NAME = "sage_memories"
EMBED_MODEL = "all-MiniLM-L6-v2"

_EMBEDDER = None  # type: ignore[var-annotated]


def _get_embedder():
    """Lazy-load the sentence-transformers model (~3-5s on first call)."""
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer

        _EMBEDDER = SentenceTransformer(EMBED_MODEL)
    return _EMBEDDER


def _chroma_id(namespace: tuple[str, ...], key: str) -> str:
    return f"{'::'.join(namespace)}::{key}"


def _ns_filter(namespace: tuple[str, ...]) -> dict:
    """Chroma where-filter restricting a query to a single namespace."""
    if len(namespace) == 1:
        return {"ns0": namespace[0]}
    clauses = [{"ns" + str(i): v} for i, v in enumerate(namespace)]
    return {"$and": clauses}


def _ns_metadata(namespace: tuple[str, ...]) -> dict:
    return {"ns" + str(i): level for i, level in enumerate(namespace)}


def _ts_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_ts(value: str | None) -> datetime:
    return datetime.fromisoformat(value) if value else datetime.now(timezone.utc)


class ChromaStore(BaseStore):
    """BaseStore implementation backed by a single Chroma collection."""

    __slots__ = ("_client", "_collection")

    def __init__(self, persist_dir: str | None = None) -> None:
        if persist_dir is None:
            self._client = chromadb.EphemeralClient()
        else:
            self._client = chromadb.PersistentClient(path=persist_dir)
        # No embedding_function: we supply embeddings explicitly on every
        # upsert and query, so Chroma never needs to call one itself.
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=None,
        )

    def _embed(self, text: str) -> list[float]:
        vec = _get_embedder().encode(text)
        return vec.tolist() if hasattr(vec, "tolist") else list(vec)

    def _put(self, op: PutOp) -> None:
        cid = _chroma_id(op.namespace, op.key)
        if op.value is None:
            try:
                self._collection.delete(ids=[cid])
            except Exception:
                pass
            return
        content = op.value.get("content", "")
        metadata = {
            **_ns_metadata(op.namespace),
            "key": op.key,
            "content": content,
            "updated_at": _ts_now(),
        }
        self._collection.upsert(
            ids=[cid],
            embeddings=[self._embed(content)],
            metadatas=[metadata],
            documents=[content],
        )

    def _get(self, op: GetOp) -> Item | None:
        cid = _chroma_id(op.namespace, op.key)
        res = self._collection.get(ids=[cid], include=["metadatas"])
        if not res.get("ids"):
            return None
        md = res["metadatas"][0]
        ts = _parse_ts(md.get("updated_at"))
        return Item(
            value={"content": md.get("content", "")},
            key=md.get("key", op.key),
            namespace=op.namespace,
            created_at=ts,
            updated_at=ts,
        )

    def _search(self, op: SearchOp) -> list[SearchItem]:
        where = _ns_filter(op.namespace_prefix)
        if op.query:
            res = self._collection.query(
                query_embeddings=[self._embed(op.query)],
                n_results=op.limit,
                where=where,
                include=["metadatas", "distances"],
            )
            mds = (res.get("metadatas") or [[]])[0]
            dists = (res.get("distances") or [[]])[0]
            items: list[SearchItem] = []
            for md, dist in zip(mds, dists):
                ts = _parse_ts(md.get("updated_at"))
                items.append(
                    SearchItem(
                        namespace=op.namespace_prefix,
                        key=md.get("key", ""),
                        value={"content": md.get("content", "")},
                        created_at=ts,
                        updated_at=ts,
                        score=1.0 - float(dist),
                    )
                )
            return items

        res = self._collection.get(where=where, include=["metadatas"])
        mds = res.get("metadatas") or []
        items = []
        for md in mds:
            ts = _parse_ts(md.get("updated_at"))
            items.append(
                SearchItem(
                    namespace=op.namespace_prefix,
                    key=md.get("key", ""),
                    value={"content": md.get("content", "")},
                    created_at=ts,
                    updated_at=ts,
                    score=None,
                )
            )
        return items[op.offset : op.offset + op.limit]

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        results: list[Result] = []
        for op in ops:
            if isinstance(op, GetOp):
                results.append(self._get(op))
            elif isinstance(op, PutOp):
                self._put(op)
                results.append(None)
            elif isinstance(op, SearchOp):
                results.append(self._search(op))
            elif isinstance(op, ListNamespacesOp):
                raise NotImplementedError(
                    "ListNamespacesOp is not supported by ChromaStore"
                )
            else:
                raise TypeError(f"Unsupported op type: {type(op).__name__}")
        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        # Chroma is sync-only; concurrency lives at the asyncio.gather level
        # inside graph.store_memory, not inside the store itself.
        return self.batch(ops)


def make_store(persist_dir: str | None = None) -> BaseStore:
    return ChromaStore(persist_dir=persist_dir)


def memory_namespace(user_id: str) -> tuple[str, str]:
    return (MEMORY_NAMESPACE, user_id)


def list_memories(store: BaseStore, user_id: str) -> list[dict]:
    # limit=1000 covers any plausible per-user memory count; default of 10
    # would silently truncate the eval's contradiction_update check.
    items = store.search(memory_namespace(user_id), limit=1000)
    return [{"key": item.key, **item.value} for item in items]
