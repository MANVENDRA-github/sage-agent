"""Streamlit UI for the sage memory agent.

Run locally:
    streamlit run src/sage_agent/app.py

The app wraps the same LangGraph build_graph() the CLI uses, with a
persistent Chroma store at .chroma/ so memories survive page reloads
and across user_ids. Each Streamlit session gets its own thread_id;
"new thread" resets the conversation history but keeps the user's
stored memories.

Per-interaction cost: ~1 LLM call for response, +1 LLM call per save
(judge or classifier), so 1-2 free-tier OpenRouter calls per chat turn.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path
from typing import Any

import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

# Streamlit Cloud doesn't run a `.env` file; surface the key via st.secrets
# if it's defined there. `st.secrets` raises if no secrets.toml exists at all
# (typical local-dev case), so swallow that and fall through to .env /
# os.environ.
if not os.environ.get("OPENROUTER_API_KEY"):
    try:
        if "OPENROUTER_API_KEY" in st.secrets:
            os.environ["OPENROUTER_API_KEY"] = st.secrets["OPENROUTER_API_KEY"]
    except Exception:
        pass

from sage_agent.graph import build_graph  # noqa: E402  (after env setup)
from sage_agent.store import list_memories, make_store  # noqa: E402

PERSIST_DIR = ".chroma/"
TYPE_BADGE = {"fact": "🟦", "preference": "🟨", "episodic": "🟪"}


@st.cache_resource(show_spinner="Loading agent (first run downloads the embedder)…")
def get_graph_and_store():
    """Cached so the embedder + Chroma client load once per Streamlit process."""
    store = make_store(persist_dir=PERSIST_DIR)
    checkpointer = MemorySaver()
    graph = build_graph(checkpointer=checkpointer, store=store)
    return graph, store


async def _chat_once(graph, user_id: str, thread_id: str, text: str) -> str:
    config = {"configurable": {"user_id": user_id, "thread_id": thread_id}}
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=text)]}, config=config
    )
    messages = result["messages"]
    for msg in reversed(messages):
        if getattr(msg, "type", None) == "ai" and not getattr(msg, "tool_calls", None):
            return msg.content or ""
    return "(no response)"


def _ensure_state() -> None:
    if "user_id" not in st.session_state:
        st.session_state.user_id = "alice"
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "history" not in st.session_state:
        st.session_state.history = []  # list[(role, text)]


def _new_thread() -> None:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.history = []


def _switch_user(new_user_id: str) -> None:
    if new_user_id and new_user_id != st.session_state.user_id:
        st.session_state.user_id = new_user_id
        _new_thread()


def _render_memories(store, user_id: str) -> None:
    memories = list_memories(store, user_id)
    if not memories:
        st.caption("No memories yet — say something memorable.")
        return
    for m in memories:
        t = m.get("type", "fact")
        badge = TYPE_BADGE.get(t, "⬜")
        st.markdown(f"{badge} **`{t}`** · {m['content']}")


def main() -> None:
    st.set_page_config(page_title="sage-agent", page_icon="🧠", layout="wide")
    _ensure_state()

    if not os.environ.get("OPENROUTER_API_KEY"):
        st.error(
            "OPENROUTER_API_KEY is not set. Add it to `.env` for local runs, or to "
            "Streamlit Secrets when deploying. Get a free key at https://openrouter.ai/keys."
        )
        st.stop()

    graph, store = get_graph_and_store()

    with st.sidebar:
        st.title("sage-agent")
        st.caption("Memory-augmented conversational agent · LangGraph + Chroma")
        st.divider()

        new_user = st.text_input(
            "User ID",
            value=st.session_state.user_id,
            help="Memories scope to this ID. Change to switch personas.",
        )
        if new_user != st.session_state.user_id:
            _switch_user(new_user)
            st.rerun()

        st.caption(f"Thread: `{st.session_state.thread_id[:8]}`")
        if st.button("New thread", help="Reset chat history, keep memories"):
            _new_thread()
            st.rerun()

        st.divider()
        st.subheader("Memories")
        _render_memories(store, st.session_state.user_id)

    st.title("Chat")
    st.caption(
        f"user_id = `{st.session_state.user_id}` · "
        f"memory store at `{PERSIST_DIR}` (persists across reloads)"
    )

    for role, text in st.session_state.history:
        with st.chat_message(role):
            st.markdown(text)

    user_input = st.chat_input("Say something the agent should remember…")
    if user_input:
        st.session_state.history.append(("user", user_input))
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            with placeholder.container():
                with st.spinner("Thinking…"):
                    try:
                        reply = asyncio.run(
                            _chat_once(
                                graph,
                                st.session_state.user_id,
                                st.session_state.thread_id,
                                user_input,
                            )
                        )
                    except Exception as e:  # noqa: BLE001
                        reply = f"_Error: {type(e).__name__}: {e}_"
            placeholder.markdown(reply)
        st.session_state.history.append(("assistant", reply))
        st.rerun()


if __name__ == "__main__":
    main()
