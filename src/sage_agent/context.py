"""Per-invocation context (user_id, model, system_prompt).

Mirrors the upstream memory-agent template's Context dataclass shape so the
Week 2+ extensions slot in without restructuring.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields

from sage_agent.prompts import SYSTEM_PROMPT


@dataclass(kw_only=True)
class Context:
    user_id: str = "default"
    model: str = field(
        default_factory=lambda: os.environ.get(
            "MODEL_NAME", "google/gemini-2.0-flash-exp:free"
        )
    )
    system_prompt: str = SYSTEM_PROMPT

    def __post_init__(self) -> None:
        for f in fields(self):
            env_value = os.environ.get(f.name.upper())
            current = getattr(self, f.name)
            if env_value and current == f.default:
                setattr(self, f.name, env_value)
