"""Graph state.

Phase 1 keeps this minimal — just messages with the standard reducer. The
retrieved_memories field arrives in Week 2 when semantic retrieval lands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


@dataclass
class State:
    messages: Annotated[list[AnyMessage], add_messages] = field(default_factory=list)
