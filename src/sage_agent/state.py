"""Graph state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


@dataclass
class State:
    messages: Annotated[list[AnyMessage], add_messages] = field(default_factory=list)
    retrieved_memories: list[dict] = field(default_factory=list)
