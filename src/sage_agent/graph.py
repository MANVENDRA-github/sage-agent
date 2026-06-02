"""Memory agent graph — model-driven ReAct loop.

Graph shape:
    START → start_turn → call_model → (route_after_model) ⇄ tools → END

The model drives everything. ``call_model`` is bound with four tools —
``search_memory`` (recall), ``save_memory`` (write), ``web_search``
(external lookup, Phase 2), and ``manage_goal`` (goal tracking, Phase 3).
On each hop:

- if the model emits tool calls → ``tools`` executes them and loops back to
  ``call_model`` so the model can read the results and continue;
- if the model emits a plain answer → END.

Retrieval is no longer a forced node: the model chooses when to recall by
calling ``search_memory``. The old forced ``retrieve_memories`` step is gone.

The loop is bounded by a per-turn step counter (``State.step``): ``start_turn``
resets it to 0 each user turn and ``call_model`` increments it. On the
``MAX_MODEL_STEPS``-th model step the model is invoked WITHOUT tools, which
makes a tool call impossible and forces a final text answer — a hard stop
that also keeps us well under LangGraph's recursion limit.

Save behavior is preserved exactly from the previous design: it is folded into
the ``tools`` node (not the tool body) so the N tool_calls / N ToolMessages
pairing the LLM expects stays intact, and so the conflict-resolution judge,
type classification, and DELETE-then-INSERT all still run. For each save we
semantic-search the top-3 similar existing memories; if any exist an LLM judge
decides insert-vs-replace, with replace implemented as DELETE-then-INSERT (not
upsert) so a contradiction_update still counts as a save in the eval metric.

Each tool call runs through a single retry and then degrades gracefully: a
malformed or transiently-failing call returns an error ToolMessage instead of
raising, so the loop continues and the model answers the user in plain text.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Literal

from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field, model_validator

from sage_agent.model import get_model
from sage_agent.prompts import CLASSIFIER_PROMPT, JUDGE_PROMPT, SYSTEM_PROMPT
from sage_agent.state import State
from sage_agent.store import make_store, memory_namespace
from sage_agent.tools import manage_goal, save_memory, search_memory, web_search

# Full set of memory types that may be STORED. "goal" is added by the
# manage_goal tool ONLY (Phase 3) — never by save_memory's auto-classifier.
MemoryType = Literal["fact", "preference", "episodic", "goal"]
# The auto-classifier and conflict judge may ONLY ever assign these three.
# "goal" is deliberately excluded here so save_memory can never label ordinary
# chit-chat as a goal: goals are reachable exclusively through manage_goal.
ClassifiableType = Literal["fact", "preference", "episodic"]
DEFAULT_TYPE: ClassifiableType = "fact"

TOOLS = [search_memory, save_memory, web_search, manage_goal]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

CONFLICT_NEIGHBORS_K = 3
# Cap the ReAct loop at this many model steps per user turn. On the final step
# the model is invoked without tools, forcing a text answer.
MAX_MODEL_STEPS = 5


def _user_id(config: RunnableConfig) -> str:
    configurable = (config or {}).get("configurable", {}) or {}
    return configurable.get("user_id", "default")


class JudgeDecision(BaseModel):
    """Type + insert-or-replace decision produced by the conflict-resolution judge."""

    type: ClassifiableType = Field(
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

    type: ClassifiableType


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


async def _classify_save(candidate: str) -> ClassifiableType:
    prompt = CLASSIFIER_PROMPT.format(candidate=candidate)
    classifier = get_model().with_structured_output(_ClassifierResponse)
    try:
        result = await classifier.ainvoke(prompt)
        return result.type
    except Exception:
        return DEFAULT_TYPE


async def start_turn(state: State) -> dict:
    """Reset the per-turn model-step counter at the start of each user turn.

    The 5-step ReAct cap is per turn. With a checkpointer (CLI / Streamlit)
    State persists across turns, so without this reset the counter would leak
    from one turn into the next and prematurely force final answers. The eval
    runner uses no checkpointer, where this is a harmless no-op (step is
    already 0).
    """
    return {"step": 0}


async def call_model(state: State, config: RunnableConfig, store: BaseStore) -> dict:
    """One ReAct model step.

    Increments the per-turn step counter. Until the cap, the model is bound
    with both tools and may either answer or emit tool calls. On the final
    allowed step the model is invoked WITHOUT tools so it cannot call one —
    guaranteeing the loop terminates with a text answer.
    """
    step = state.step + 1
    force_final = step >= MAX_MODEL_STEPS

    messages = [SystemMessage(content=SYSTEM_PROMPT), *state.messages]
    model = get_model()
    if not force_final:
        model = model.bind_tools(TOOLS)
    response = await model.ainvoke(messages)
    return {"messages": [response], "step": step}


async def _handle_save(tc: dict, *, namespace: tuple[str, str], store: BaseStore) -> str:
    """Execute one save_memory call with full conflict resolution + typing.

    Preserved verbatim from the previous store_memory node: semantic-search
    neighbors, classify (no-neighbor path) or judge (neighbor path), and apply
    replace as DELETE-then-INSERT so contradiction_update still counts as a
    save. Returns the human-readable ToolMessage content string.
    """
    candidate = (tc.get("args") or {}).get("content", "")
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
        return f"Saved memory {new_id} ({mem_type})"

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
        return f"Replaced memory {decision.target_key} with new {decision.type}"

    new_id = str(uuid.uuid4())
    await store.aput(
        namespace,
        key=new_id,
        value={"content": decision.content, "type": decision.type},
    )
    return f"Saved memory {new_id} ({decision.type})"


async def _handle_search(
    tc: dict, *, user_id: str, store: BaseStore
) -> str:
    """Execute one search_memory call by invoking the actual tool.

    The tool wraps store.asearch, so this is genuine tool execution — no second
    copy of retrieval logic. user_id and store are InjectedToolArgs the model
    never supplies, so we pass them explicitly here.
    """
    query = (tc.get("args") or {}).get("query", "")
    return await search_memory.ainvoke(
        {"query": query, "user_id": user_id, "store": store}
    )


async def _handle_web_search(tc: dict) -> str:
    """Execute one web_search call by invoking the actual tool.

    web_search takes no InjectedToolArgs (no store / user_id) — it just wraps
    the keyless ddgs DuckDuckGo search. The tool already turns a no-results
    search into a readable "No web results found" string; any transient ddgs
    failure raises and is handled by the retry-once-then-degrade loop below.
    """
    query = (tc.get("args") or {}).get("query", "")
    return await web_search.ainvoke({"query": query})


async def _handle_manage_goal(tc: dict, *, user_id: str, store: BaseStore) -> str:
    """Execute one manage_goal call by invoking the actual tool.

    manage_goal does all its own store work (set / list / update with the
    DELETE-then-INSERT pattern reused from a save replace). user_id and store
    are InjectedToolArgs the model never supplies, so we pass them explicitly;
    the model-provided action / goal / status / new_goal are forwarded as-is.
    """
    args = tc.get("args") or {}
    return await manage_goal.ainvoke(
        {
            "action": args.get("action", ""),
            "goal": args.get("goal", ""),
            "status": args.get("status", ""),
            "new_goal": args.get("new_goal", ""),
            "user_id": user_id,
            "store": store,
        }
    )


async def _execute_tool_call(
    tc: dict, *, user_id: str, namespace: tuple[str, str], store: BaseStore
) -> ToolMessage:
    """Run one tool call with a single retry, then degrade gracefully.

    A malformed or transiently-failing tool call (bad args, an unknown tool
    name, a free-tier judge hiccup) is retried once. If it still fails we
    return an error ToolMessage rather than raising, so the loop continues back
    to call_model and the model answers the user in plain text instead of the
    whole turn crashing.
    """
    name = tc.get("name", "")
    tool_call_id = tc.get("id", "")
    last_exc: Exception | None = None

    for _attempt in range(2):  # initial try + one retry
        try:
            if name == "save_memory":
                content = await _handle_save(tc, namespace=namespace, store=store)
            elif name == "search_memory":
                content = await _handle_search(tc, user_id=user_id, store=store)
            elif name == "web_search":
                content = await _handle_web_search(tc)
            elif name == "manage_goal":
                content = await _handle_manage_goal(tc, user_id=user_id, store=store)
            else:
                raise ValueError(f"unknown tool {name!r}")
            return ToolMessage(content=content, tool_call_id=tool_call_id, name=name)
        except Exception as exc:  # noqa: BLE001 — deliberate graceful degradation
            last_exc = exc

    return ToolMessage(
        content=(
            f"[tool '{name}' failed after a retry: {type(last_exc).__name__}: "
            f"{last_exc}. Answer the user directly without it.]"
        ),
        tool_call_id=tool_call_id,
        name=name or "unknown",
    )


async def tools_node(state: State, config: RunnableConfig, store: BaseStore) -> dict:
    """Execute every tool call on the last AI message, one ToolMessage each.

    save_memory routes through the conflict-resolution path; search_memory
    invokes the recall tool. Calls run concurrently (matching the prior
    design); each is independently retried-once and degraded.
    """
    user_id = _user_id(config)
    namespace = memory_namespace(user_id)
    last = state.messages[-1] if state.messages else None
    tool_calls = getattr(last, "tool_calls", []) or []

    results = await asyncio.gather(
        *(
            _execute_tool_call(tc, user_id=user_id, namespace=namespace, store=store)
            for tc in tool_calls
        )
    )
    return {"messages": list(results)}


def route_after_model(state: State) -> str:
    last = state.messages[-1] if state.messages else None
    tool_calls = getattr(last, "tool_calls", None) if last is not None else None
    if tool_calls:
        return "tools"
    return END


def build_graph(*, checkpointer=None, store: BaseStore | None = None):
    builder = StateGraph(State)
    builder.add_node("start_turn", start_turn)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", tools_node)
    builder.add_edge(START, "start_turn")
    builder.add_edge("start_turn", "call_model")
    builder.add_conditional_edges(
        "call_model", route_after_model, {"tools": "tools", END: END}
    )
    builder.add_edge("tools", "call_model")

    compile_kwargs = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer
    if store is not None:
        compile_kwargs["store"] = store
    return builder.compile(**compile_kwargs)


graph = build_graph(store=make_store())
