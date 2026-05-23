"""Terminal REPL for chatting with the baseline agent.

Run:  python -m sage_agent.cli --user-id alice

Commands:
    /new        start a new conversation thread (memories persist)
    /memories   dump all stored memories for the current user
    /quit       exit

The store is held in-process for the session, so /new clears conversation
history but keeps everything the agent has remembered about the user — which
is what lets you verify cross-thread memory persistence in one session.
"""

from __future__ import annotations

import argparse
import asyncio
import uuid

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from sage_agent.graph import build_graph
from sage_agent.store import list_memories, make_store


async def _chat_once(graph, store, user_id: str, thread_id: str, text: str) -> str:
    config = {"configurable": {"user_id": user_id, "thread_id": thread_id}}
    result = await graph.ainvoke({"messages": [HumanMessage(content=text)]}, config=config)
    messages = result["messages"]
    for msg in reversed(messages):
        if getattr(msg, "type", None) == "ai" and not getattr(msg, "tool_calls", None):
            return msg.content or ""
    return "(no response)"


def _print_memories(store, user_id: str) -> None:
    memories = list_memories(store, user_id)
    if not memories:
        print("(no memories)")
        return
    for i, m in enumerate(memories, 1):
        print(f"  {i}. {m['content']}")


async def _run(user_id: str) -> None:
    store = make_store()
    checkpointer = MemorySaver()
    graph = build_graph(checkpointer=checkpointer, store=store)
    thread_id = str(uuid.uuid4())

    print(f"sage-agent | user_id={user_id} | thread={thread_id[:8]}")
    print("Commands: /new  /memories  /quit\n")

    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/new":
            thread_id = str(uuid.uuid4())
            print(f"[new thread: {thread_id[:8]}]")
            continue
        if text == "/memories":
            _print_memories(store, user_id)
            continue

        try:
            reply = await _chat_once(graph, store, user_id, thread_id, text)
        except Exception as e:
            print(f"[error] {e}")
            continue
        print(f"bot> {reply}\n")


def main() -> None:
    parser = argparse.ArgumentParser(prog="sage-agent")
    parser.add_argument("--user-id", default="default", help="User identifier for memory scoping.")
    args = parser.parse_args()
    asyncio.run(_run(args.user_id))


if __name__ == "__main__":
    main()
