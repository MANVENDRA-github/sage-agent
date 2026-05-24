"""Memory agent graph.

Graph shape:
    START → retrieve_memories → call_model → (conditional) → store_memory → call_model → END

The conditional routes on whether the model emitted a save_memory tool call.
After a save the graph loops back to call_model so the model can produce a
natural response. The store_memory → call_model edge skips re-retrieval; the
user's query hasn't changed mid-turn so the cached retrieved_memories on
state is still correct.

We hand-roll store_memory (rather than using ToolNode) because we need to
inject `store` and `user_id` into the tool — InjectedToolArg only hides the
arg from the LLM's schema; populating it is on us.
"""

from __future__ import annotations

import asyncio

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore

from sage_agent.model import get_model
from sage_agent.prompts import SYSTEM_PROMPT
from sage_agent.state import State
from sage_agent.store import make_store, memory_namespace
from sage_agent.tools import save_memory

TOOLS = [save_memory]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

RETRIEVAL_K = 5


def _user_id(config: RunnableConfig) -> str:
    configurable = (config or {}).get("configurable", {}) or {}
    return configurable.get("user_id", "default")


def _format_user_info(memories: list[dict]) -> str:
    if not memories:
        return "(no memories yet)"
    return "\n".join(f"- {m['content']}" for m in memories)


def _last_human_text(messages: list) -> str | None:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return None


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
    last = state.messages[-1]
    tool_calls = getattr(last, "tool_calls", []) or []

    async def _run(tc: dict) -> ToolMessage:
        tool = TOOLS_BY_NAME[tc["name"]]
        result = await tool.ainvoke(
            {**tc["args"], "user_id": user_id, "store": store}
        )
        return ToolMessage(content=str(result), tool_call_id=tc["id"], name=tc["name"])

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
