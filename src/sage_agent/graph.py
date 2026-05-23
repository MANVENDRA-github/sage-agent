"""Baseline ReAct memory agent.

Graph shape mirrors the upstream langchain-ai/memory-agent template:
    call_model → (conditional) → store_memory → call_model → END

The conditional routes on whether the model emitted a tool call. After a
save the graph loops back to call_model so the model can produce a natural
response to the user.

We hand-roll store_memory (rather than using ToolNode) because we need to
inject both `store` and `user_id` into the tool. InjectedToolArg only hides
the arg from the LLM's schema; populating it is on us.
"""

from __future__ import annotations

import asyncio

from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore

from sage_agent.model import get_model
from sage_agent.prompts import SYSTEM_PROMPT
from sage_agent.state import State
from sage_agent.store import list_memories, make_store
from sage_agent.tools import save_memory

TOOLS = [save_memory]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


def _user_id(config: RunnableConfig) -> str:
    configurable = (config or {}).get("configurable", {}) or {}
    return configurable.get("user_id", "default")


def _format_user_info(memories: list[dict]) -> str:
    if not memories:
        return "(no memories yet)"
    return "\n".join(f"- {m['content']}" for m in memories)


async def call_model(state: State, config: RunnableConfig, store: BaseStore) -> dict:
    user_id = _user_id(config)
    memories = list_memories(store, user_id)
    system_prompt = SYSTEM_PROMPT.format(user_info=_format_user_info(memories))
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
    builder.add_node("call_model", call_model)
    builder.add_node("store_memory", store_memory)
    builder.add_edge(START, "call_model")
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
