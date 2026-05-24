"""Memory agent graph.

Graph shape:
    START → retrieve_memories → call_model → (conditional) → store_memory → call_model → END

The conditional routes on whether the model emitted a save_memory tool call.
After a save the graph loops back to call_model so the model can produce a
natural response. The store_memory → call_model edge skips re-retrieval; the
user's query hasn't changed mid-turn so the cached retrieved_memories on
state is still correct.

Conflict resolution lives inside store_memory (not as a separate graph
node) so the N tool_calls / N ToolMessages pairing the LLM expects stays
intact. For each save, we semantic-search for the top-3 similar existing
memories and — if any neighbors exist — ask an LLM judge to decide
insert-vs-replace. Replace is implemented as DELETE-then-INSERT (not
upsert) so that a contradiction_update case still counts as a save in the
runner's save-decision metric.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field, model_validator

from sage_agent.model import get_model
from sage_agent.prompts import CLASSIFIER_PROMPT, JUDGE_PROMPT, SYSTEM_PROMPT
from sage_agent.state import State
from sage_agent.store import make_store, memory_namespace
from sage_agent.tools import save_memory

MemoryType = Literal["fact", "preference", "episodic"]
DEFAULT_TYPE: MemoryType = "fact"

TOOLS = [save_memory]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

RETRIEVAL_K = 5
CONFLICT_NEIGHBORS_K = 3


def _user_id(config: RunnableConfig) -> str:
    configurable = (config or {}).get("configurable", {}) or {}
    return configurable.get("user_id", "default")


def _format_user_info(memories: list[dict]) -> str:
    if not memories:
        return "(no memories yet)"
    return "\n".join(
        f"- [{m.get('type', DEFAULT_TYPE)}] {m['content']}" for m in memories
    )


def _last_human_text(messages: list) -> str | None:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return None


class JudgeDecision(BaseModel):
    """Type + insert-or-replace decision produced by the conflict-resolution judge."""

    type: MemoryType = Field(
        description="Memory type: fact / preference / episodic."
    )
    action: Literal["insert", "replace"] = Field(
        description="`insert` for new info, `replace` to update an existing memory."
    )
    target_key: str | None = Field(
        default=None,
        description="Required when action='replace' — the key of the memory to replace.",
    )
    content: str = Field(
        description="The memory content to store (judge may rewrite for clarity)."
    )

    @model_validator(mode="after")
    def _replace_requires_target(self) -> "JudgeDecision":
        if self.action == "replace" and not self.target_key:
            raise ValueError("action='replace' requires target_key")
        return self


class _ClassifierResponse(BaseModel):
    """Type-only response for the no-neighbor save path."""

    type: MemoryType


def _format_neighbors(neighbors: list[dict]) -> str:
    return "\n".join(
        f'- {{"key": "{n["key"]}", "type": "{n.get("type", DEFAULT_TYPE)}", "content": "{n["content"]}"}}'
        for n in neighbors
    )


async def _judge_save(candidate: str, neighbors: list[dict]) -> JudgeDecision:
    prompt = JUDGE_PROMPT.format(
        candidate=candidate,
        neighbors=_format_neighbors(neighbors),
    )
    judge = get_model().with_structured_output(JudgeDecision)
    decision = await judge.ainvoke(prompt)

    valid_keys = {n["key"] for n in neighbors}
    if decision.action == "replace":
        if decision.target_key not in valid_keys:
            return JudgeDecision(
                type=decision.type, action="insert", target_key=None, content=candidate
            )
        target = next(n for n in neighbors if n["key"] == decision.target_key)
        if target.get("type", DEFAULT_TYPE) != decision.type:
            return JudgeDecision(
                type=decision.type, action="insert", target_key=None, content=candidate
            )
    return decision


async def _classify_save(candidate: str) -> MemoryType:
    prompt = CLASSIFIER_PROMPT.format(candidate=candidate)
    classifier = get_model().with_structured_output(_ClassifierResponse)
    try:
        result = await classifier.ainvoke(prompt)
        return result.type
    except Exception:
        return DEFAULT_TYPE


async def retrieve_memories(state: State, config: RunnableConfig, store: BaseStore) -> dict:
    query = _last_human_text(state.messages)
    if not query:
        return {"retrieved_memories": []}
    user_id = _user_id(config)
    items = await store.asearch(memory_namespace(user_id), query=query, limit=RETRIEVAL_K)
    return {
        "retrieved_memories": [
            {"key": item.key, **item.value} for item in items
        ]
    }


async def call_model(state: State, config: RunnableConfig, store: BaseStore) -> dict:
    system_prompt = SYSTEM_PROMPT.format(
        user_info=_format_user_info(state.retrieved_memories)
    )
    messages = [SystemMessage(content=system_prompt), *state.messages]

    model = get_model().bind_tools(TOOLS)
    response = await model.ainvoke(messages)
    return {"messages": [response]}


async def store_memory(state: State, config: RunnableConfig, store: BaseStore) -> dict:
    user_id = _user_id(config)
    namespace = memory_namespace(user_id)
    last = state.messages[-1]
    tool_calls = getattr(last, "tool_calls", []) or []

    async def _run(tc: dict) -> ToolMessage:
        candidate = tc["args"].get("content", "")
        neighbors_items = await store.asearch(
            namespace, query=candidate, limit=CONFLICT_NEIGHBORS_K
        )
        neighbors = [
            {
                "key": n.key,
                "type": n.value.get("type", DEFAULT_TYPE),
                "content": n.value.get("content", ""),
            }
            for n in neighbors_items
        ]

        if not neighbors:
            mem_type = await _classify_save(candidate)
            new_id = str(uuid.uuid4())
            await store.aput(
                namespace, key=new_id, value={"content": candidate, "type": mem_type}
            )
            content_msg = f"Saved memory {new_id} ({mem_type})"
        else:
            try:
                decision = await _judge_save(candidate, neighbors)
            except Exception:
                fallback_type = await _classify_save(candidate)
                decision = JudgeDecision(
                    type=fallback_type, action="insert", target_key=None, content=candidate
                )

            if decision.action == "replace":
                await store.adelete(namespace, key=decision.target_key)
                new_id = str(uuid.uuid4())
                await store.aput(
                    namespace,
                    key=new_id,
                    value={"content": decision.content, "type": decision.type},
                )
                content_msg = (
                    f"Replaced memory {decision.target_key} with new {decision.type}"
                )
            else:
                new_id = str(uuid.uuid4())
                await store.aput(
                    namespace,
                    key=new_id,
                    value={"content": decision.content, "type": decision.type},
                )
                content_msg = f"Saved memory {new_id} ({decision.type})"

        return ToolMessage(content=content_msg, tool_call_id=tc["id"], name=tc["name"])

    results = await asyncio.gather(*(_run(tc) for tc in tool_calls))
    return {"messages": list(results)}


def route_after_model(state: State) -> str:
    last = state.messages[-1] if state.messages else None
    tool_calls = getattr(last, "tool_calls", None) if last is not None else None
    if tool_calls:
        return "store_memory"
    return END


def build_graph(*, checkpointer=None, store: BaseStore | None = None):
    builder = StateGraph(State)
    builder.add_node("retrieve_memories", retrieve_memories)
    builder.add_node("call_model", call_model)
    builder.add_node("store_memory", store_memory)
    builder.add_edge(START, "retrieve_memories")
    builder.add_edge("retrieve_memories", "call_model")
    builder.add_conditional_edges(
        "call_model", route_after_model, {"store_memory": "store_memory", END: END}
    )
    builder.add_edge("store_memory", "call_model")

    compile_kwargs = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer
    if store is not None:
        compile_kwargs["store"] = store
    return builder.compile(**compile_kwargs)


graph = build_graph(store=make_store())
