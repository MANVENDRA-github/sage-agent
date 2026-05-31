"""Graph state.

``messages`` is the running ReAct transcript (user / assistant / tool
messages), accumulated via the ``add_messages`` reducer.

``step`` is the per-turn model-step counter that bounds the ReAct loop: it is
reset to 0 at the start of every user turn (the ``start_turn`` node) and
incremented on each ``call_model`` invocation. When it reaches the cap the
model is invoked without tools, forcing a final text answer. Reset-per-turn
matters because with a checkpointer (CLI / Streamlit) State persists across
turns, and an un-reset counter would leak the cap from one turn into the next.

``retrieved_memories`` is legacy: retrieval is now model-driven via the
``search_memory`` tool (results arrive as ToolMessages in ``messages``), so no
node populates this field anymore. It is kept for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


@dataclass
class State:
    messages: Annotated[list[AnyMessage], add_messages] = field(default_factory=list)
    retrieved_memories: list[dict] = field(default_factory=list)
    step: int = 0
