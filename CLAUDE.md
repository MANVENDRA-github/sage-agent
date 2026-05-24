# CLAUDE.md

Guidance for Claude Code when working in this repository. Read this first
before touching code; it captures intent, conventions, and the roadmap that
file contents alone don't convey.

## What this project is

**sage-agent** — a memory-augmented conversational agent built on LangGraph.
It extends the upstream [`langchain-ai/memory-agent`](https://github.com/langchain-ai/memory-agent)
template with:

1. **Semantic retrieval** — embed-and-search instead of "dump everything."
2. **Conflict resolution** — LLM-judged insert-vs-update on save.
3. **Typed memory** — fact / preference / episodic, each with its own
   retention semantics.
4. **A 50-case evaluation suite** — every improvement is measured as a delta,
   not asserted as a vibe.

The headline of the project is the eval table, not the feature list. Week 1
ships the baseline + harness so Weeks 2–4 can claim measurable wins.

**Owner**: Manvendra. **License**: MIT. **Python**: ≥ 3.11. **Package manager**: `uv`.

## Current status (Phase 1 / Week 1 — shipped)

Baseline ReAct agent + 50-case eval harness + README are merged on `dev`.
Baseline numbers have not yet been generated — the results table in the
README is intentionally blank pending a real run of `tests.eval.runner`.

What is **live**:

- ReAct loop: `call_model` ↔ `store_memory`, with conditional routing on
  whether the LLM emitted a `save_memory` tool call.
- `InMemoryStore` backed by `langgraph.store.memory.InMemoryStore`.
- `save_memory` tool — **blind append only**, no `memory_id`, no update path.
- "Retrieve memories" is a **stub**: it lists ALL memories for the user and
  injects them into the system prompt. Real semantic retrieval lands Week 2.
- Terminal REPL with `/new`, `/memories`, `/quit`.
- 50 eval cases across 6 categories, runner with per-category pass rate and
  global save-decision precision / recall / F1.

What is **intentionally absent** in Phase 1 (do not "fix" these without
checking the roadmap — they are baseline-defining gaps):

- No type classifier — every memory is type-less.
- No conflict check — `contradiction_update` cases are expected to fail
  because every save is an append. That headroom is what Week 2 consumes.
- No vector store / no embeddings — `retrieval_relevance` is also expected
  to score low at baseline.
- No retention metadata, no decay, no consolidation.
- No UI beyond the CLI.

## Roadmap

| Phase | Scope | Ship gate |
|------:|-------|-----------|
| Week 1 ✅ | Baseline agent, in-memory store, eval harness, README | Harness runs end-to-end; baseline JSON written |
| Week 2 | Chroma + `all-MiniLM-L6-v2` for semantic retrieval; conflict-resolution save subgraph (top-k similar + LLM judge → insert vs update) | Measurable lift on `contradiction_update` and `retrieval_relevance` over baseline |
| Week 3 | Typed memory: classifier node → fact / preference / episodic; per-type retention metadata; type-aware retrieval cues | Improvement reflected in the eval (especially category-balanced metrics) |
| Week 4 | Decay / consolidation, Streamlit UI, hosted demo, README rewrite with final numbers, blog post | Live demo URL |

When implementing Week 2+: the file shapes are deliberately roomed for it.
`State` has only `messages` today but is meant to grow a `retrieved_memories`
field. `store.py` returns `BaseStore`, so Chroma swaps in via constructor.
`tools.py` exposes a single `save_memory` today; the conflict-resolution
subgraph will introduce an `upsert_memory(memory_id=...)` variant matching
the upstream template's shape.

## Repo layout

```
sage-agent/
├── pyproject.toml              uv-managed; hatchling build; package = src/sage_agent
├── README.md                   public-facing — leads with numbers, not features
├── CLAUDE.md                   this file
├── docs/
│   └── EVAL.md                 canonical eval methodology — schema, scoring, how to add cases
├── .env.example                OPENROUTER_API_KEY + MODEL_NAME template
├── .gitignore                  notably ignores .claude/ (harness state) but keeps eval results
├── src/sage_agent/
│   ├── __init__.py             __version__ only
│   ├── cli.py                  terminal REPL; /new, /memories, /quit
│   ├── context.py              Context dataclass (user_id, model, system_prompt) — env-var override pattern
│   ├── graph.py                LangGraph state machine: call_model ↔ store_memory
│   ├── model.py                ChatOpenAI factory pointed at OpenRouter
│   ├── prompts.py              SYSTEM_PROMPT with {user_info} slot
│   ├── state.py                State dataclass — messages with add_messages reducer
│   ├── store.py                make_store(), memory_namespace(), list_memories()
│   └── tools.py                save_memory @tool with InjectedToolArg for store + user_id
└── tests/
    ├── __init__.py
    └── eval/
        ├── __init__.py
        ├── cases.json          50 cases across 6 categories
        ├── runner.py           load → validate → run → score → aggregate → write JSON
        └── results/            .gitkept; baseline_*.json lands here (and is committed)
```

## Architecture

```
user turn ──► retrieve memories (stub: list all) ──► chat LLM
                                                      │
                                          ┌───────────┴───────────┐
                                          ▼                       ▼
                                   no tool call            save_memory call
                                          │                       │
                                          ▼                       ▼
                                       respond              store_memory ──► back to chat LLM
```

LangGraph nodes (Phase 1 status in parens):

- **retrieve memories** — embed the latest user message, query the vector
  store for top-k similar memories scoped to `user_id`, attach to state.
  *(Phase 1: stub — dumps ALL memories. Real retrieval = Week 2.)*
- **chat LLM (`call_model`)** — system prompt + memories + history → either a
  natural response or a `save_memory` tool call. *(Phase 1: live.)*
- **classify type** — route a candidate memory to fact / preference / episodic.
  *(Phase 1: absent — Week 3.)*
- **conflict check** — semantic-search top-k similar existing memories; LLM
  judge decides insert vs update. *(Phase 1: absent — Week 2. This is the
  reason baseline `contradiction_update` is expected to be near zero.)*
- **store (`store_memory`)** — write to the vector store + persistent backend
  with type-specific retention metadata. *(Phase 1: `InMemoryStore.aput`, no
  metadata, blind append.)*

### Why hand-rolled `store_memory` instead of `ToolNode`

`save_memory` takes `user_id` and `store` as `InjectedToolArg`s — those are
hidden from the LLM's tool schema, but `ToolNode` won't populate them for us.
The custom node reads `user_id` from `RunnableConfig.configurable`, pulls
`store` from the compiled graph context, and invokes the tool itself.

## Tech stack and key decisions

| Concern | Choice | Why |
|---|---|---|
| Orchestration | LangGraph (`>=0.6.0`) | Matches upstream template; first-class store + checkpointer; explicit state machine beats hidden ReAct loops |
| LLM | `ChatOpenAI` pointed at OpenRouter | `init_chat_model` doesn't natively route to OpenRouter; OpenAI-compatible endpoint works |
| Default model | `google/gemini-2.0-flash-exp:free` | Free tier with the strongest tool-calling among free options. `save_memory` is a tool call, so tool quality is the deciding factor |
| Embeddings (Week 2) | `sentence-transformers` / `all-MiniLM-L6-v2` (local) | $0; good enough for thousands-scale stores; API embeddings buy ~5% retrieval quality but break the free-tier story |
| Vector store (Week 2) | `chromadb` (embedded) | Zero-ops; pip-install only; eval can run offline in CI. Pinecone is over-scoped |
| Memory store (Phase 1) | `langgraph.store.memory.InMemoryStore` | `BaseStore` subclass that's literally a dict; Chroma swap is a one-line constructor change |
| Env management | `python-dotenv` + `.env` | `model.py` calls `load_dotenv()` once at import |
| Tests | `pytest` (dev group) — but the **eval runner is not a pytest suite**; it's `python -m tests.eval.runner` | The 50 cases are an evaluation harness, not unit tests. Pytest is reserved for future unit tests |

**The $0 / free-tier constraint is load-bearing.** A portfolio project where
"clone and demo on a free key" is part of the story dictates: OpenRouter
free tier, local embeddings, embedded vector store. Do not introduce paid
dependencies without flagging the tradeoff.

## Configuration

`.env` (gitignored) supplies:

```
OPENROUTER_API_KEY=sk-or-v1-...
MODEL_NAME=google/gemini-2.0-flash-exp:free
```

The user already has `.env` configured locally. `Context.__post_init__`
implements env-var override: any field with `default`-value gets replaced
by `os.environ[FIELD_NAME.upper()]` if set. Pattern is mirrored from the
upstream template — keep it when adding new context fields.

## How to run

```bash
# Install
uv sync

# Chat
uv run python -m sage_agent.cli --user-id alice

# Eval — dry-run validates the case schema without API calls
uv run python -m tests.eval.runner --dry-run

# Eval — smoke (first 5 cases)
uv run python -m tests.eval.runner --limit 5

# Eval — full 50 cases, baseline label
uv run python -m tests.eval.runner

# Eval — single category
uv run python -m tests.eval.runner --category should_save_fact

# Eval — label runs from later phases
uv run python -m tests.eval.runner --label week2
```

Each non-dry-run writes `tests/eval/results/<label>_<UTC>.json`. **Those
result files are intentionally committed** so README numbers are
reproducible from git history. See the comment in `.gitignore` — the
`tests/eval/results/*.json` ignore is deliberately commented out.

CLI commands inside the REPL: `/new` (new thread, same user — memories
persist across threads in the in-process store), `/memories` (dump store
for the active user), `/quit`.

## Eval harness — schema and scoring

> **Canonical reference: [`docs/EVAL.md`](docs/EVAL.md).** That file owns
> the methodology, the full schema, per-category scoring rules with
> rationale, the result-JSON shape, and the rules for adding cases /
> re-baselining. The summary below is enough to work day-to-day; consult
> EVAL.md before changing the harness or the corpus.

`tests/eval/cases.json` is a JSON array of case objects:

```json
{
  "id": "case_XXX",
  "category": "should_save_fact | should_save_preference | should_save_episodic | should_not_save | contradiction_update | retrieval_relevance",
  "setup_memories": [{"content": "...", "type": "fact|preference|episodic"}],
  "conversation": [{"role": "user|assistant", "content": "..."}],
  "expected": {
    "should_save": true,
    "memory_type": "fact|preference|episodic",
    "memory_content_contains": ["substring", ...],
    "response_contains": ["substring", ...],
    "update": true
  }
}
```

The runner enforces `REQUIRED_TOP_LEVEL = {id, category, conversation, expected}`
and `VALID_CATEGORIES = {...}` — add a category by editing both the runner
and the README. Per-case scoring:

- `should_save_*` cases pass iff predicted_save AND any new memory contains
  all `memory_content_contains` substrings (case-insensitive).
- `should_not_save` cases pass iff predicted_save is false.
- `contradiction_update` cases pass iff exactly **one** memory remains for
  the user AND it carries the new value's substrings. Baseline append-only
  saves N+1 memories → expected to fail. That is the point.
- `retrieval_relevance` cases pass iff NOT predicted_save AND the final
  response contains any of `response_contains`. The agent should answer
  from memory, not re-save.

Aggregate output: per-category pass rate plus a global save-decision
P / R / F1 treating `should_save` as a binary classifier across all cases.

**Per-case isolation**: every case gets a fresh `InMemoryStore` and uses
`user_id = f"eval_{case['id']}"`. The same user is never reused across
cases — don't introduce shared-state shortcuts.

## Coding conventions

- **Commit messages**: lowercase `feature: ...` is the established style.
  Look at `git log --oneline` — every commit follows it. Don't switch to
  Conventional Commits (`feat:`) mid-stream. Use `fix: ...` / `refactor: ...`
  in the same lowercase style when those land.
- **Imports**: `from __future__ import annotations` at the top of every
  module that uses type hints. `__all__` is not currently used.
- **Type hints**: required on function signatures. `dict` and `list` over
  `Dict` / `List` — the project requires Python 3.11+.
- **Async**: the graph and tools are async (`ainvoke`, `aput`). The eval
  runner is async at the top and uses `asyncio.run` in `main()`. The CLI
  follows the same pattern. Don't introduce sync paths for nodes.
- **Docstrings**: every module has a top-level docstring explaining its
  role and the Phase-1-vs-future-phase boundary where relevant. Match this
  voice when adding modules — terse, specific, names a tradeoff if there
  is one.
- **No premature abstraction**: there is one tool, one store, one model.
  Don't add a `BaseTool` / `BaseStoreFactory` / etc. until Week 2 demands it.

## What to be careful about

- **Don't "fix" the blind-append in `tools.py` outside Week 2 work.** It's
  the baseline-defining gap. Patching it would invalidate the baseline
  numbers and the project's evaluation story.
- **Don't replace the retrieve-memories stub in `graph.py` outside Week 2
  work**, for the same reason.
- **Don't rewrite `cases.json`.** It is a fixed evaluation corpus. Adding
  cases is fine; editing existing ones changes the meaning of baseline-vs-
  Week-N comparisons. If you must change a case, note it in the commit and
  flag it for re-baselining. The full "what counts as a re-baseline event"
  table lives in `docs/EVAL.md`.
- **Don't commit `.env`.** It's gitignored, but be paranoid. The user's
  `OPENROUTER_API_KEY` is real.
- **`tests/eval/results/` is gitkept and the JSONs are meant to be
  committed** — see the comment block in `.gitignore`. Don't add a blanket
  ignore there.
- **The default temperature in `model.py` is 0.0**. Eval determinism
  depends on it. Don't bump it for "creativity" without thinking through
  reproducibility.
- **Memory namespace is `("memories", user_id)`** — matches upstream. If
  you ever change the namespace shape, also update `list_memories`, the
  tool, the runner, and any future Chroma collection naming.

## External references

- Upstream template: <https://github.com/langchain-ai/memory-agent>
- LangGraph docs: <https://langchain-ai.github.io/langgraph/>
- OpenRouter (free key, model list): <https://openrouter.ai/keys>
- Chroma (Week 2 dependency, already in `pyproject.toml`): <https://www.trychroma.com/>
- `all-MiniLM-L6-v2` model card (Week 2 embedder): <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>

## Working on this repo as Claude

- **Branching**: current default is `dev`; PRs target `main`. The repo
  history is all on `dev` so far.
- **When asked to "implement Week 2"**: this is a multi-PR effort. Don't
  bundle Chroma + conflict resolution + retrieval changes into one diff.
  Suggested order: (1) introduce real retrieval (replace the stub) and
  prove the lift on `retrieval_relevance` first; (2) add the conflict-
  resolution subgraph and prove the lift on `contradiction_update`. Each
  step gets its own baseline-vs-after eval run with results committed.
- **When asked to "add a feature"**: check the roadmap. If it belongs to a
  future week, say so and ask whether the user wants to pull it forward —
  don't unilaterally accelerate the plan.
- **When asked to "fix a failing test"**: the eval is not a pass/fail
  test. A baseline run is *expected* to fail many cases — that's the
  signal. If the user means "improve the score on category X", clarify.
- **When in doubt about the voice of new code or copy**: re-read the
  README. Numbers over vibes; tradeoffs called out by name; no marketing
  adjectives.
