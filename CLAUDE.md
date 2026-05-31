# CLAUDE.md

Guidance for Claude Code (and future-me) when working in this repository.
Read this first before touching code — it captures intent, conventions,
and the state of play that file contents alone don't convey.

## What this project is

**sage-agent** — a memory-augmented conversational agent built on
LangGraph, extending the upstream
[`langchain-ai/memory-agent`](https://github.com/langchain-ai/memory-agent)
template with:

1. **Semantic retrieval** — embed-and-search via Chroma +
   `sentence-transformers all-MiniLM-L6-v2`, scoped to `user_id`,
   replacing the upstream "dump everything" stub.
2. **Conflict-resolution save subgraph** — LLM-judged insert-vs-replace on
   every save, with same-type-and-same-facet gating; replace implemented
   as DELETE-then-INSERT so the save-decision metric stays honest.
3. **Typed memory** — every memory is `fact` / `preference` / `episodic`,
   assigned by the judge on the conflict path or a dedicated classifier
   on the no-neighbor path. Type surfaces to the assistant in the
   rendered memory list and gates `replace` in the judge.
4. **A 50-case evaluation suite** — every improvement is measured as a
   delta in `tests/eval/results/<label>_<UTC>.json`. The runner emits
   per-category pass rate, save-decision P/R/F1, and (Week 3+)
   type-accuracy. `tests/eval/rescore.py` re-applies score-rule changes
   to stored runs without LLM calls.
5. **A Streamlit UI** + free-tier hosted demo at
   <https://sage-agent.streamlit.app/>.

The headline of the project is the eval table, not the feature list.

**Owner**: Manvendra. **License**: MIT. **Python**: ≥ 3.11. **Package
manager**: `uv`. **Default branch**: `dev`. **Live demo**:
<https://sage-agent.streamlit.app/>.

## Current status — Week 4 + polish complete

All four roadmap weeks plus the polish pass have shipped on `dev`. Headline
numbers from `tests/eval/results/week3_20260524T161027Z.json`:

- `contradiction_update`: 0% (baseline) → 57.1% (Week 2) → **100%** (Week 3
  + polish). Case_040 Camry→Tesla, the long-time holdout, passed on the
  polish-pass run — but the free-tier judge has shown non-determinism on
  these substitutions across re-runs, so one 100% run isn't a permanent
  guarantee.
- `retrieval_relevance`: 100% in baseline, Phase A, and Week 3 (Unicode
  whitespace normalization in the runner fixed case_047's false-fail);
  Week 2 holds at 90% with case_043 as the lone holdout.
- `should_save_fact`, `should_save_preference`, `should_not_save`: 100%
  across all runs.
- `should_save_episodic`: 80% across all runs — case_021 is a save-decision
  false-negative the model hasn't picked up at any week.
- Save-decision F1: **0.983** held constant across baseline / Week 2 /
  Week 3, which is the entire point of the DELETE-then-INSERT pattern —
  `replace` still counts as a save.
- Type accuracy: **76.7%** at Week 3 + polish (24/30 eligible). 100% on
  facts and contradiction_update; the classifier remains fuzzy on
  borderline preference-vs-fact ("does not drink coffee" → fact) and
  episodic-vs-fact ("graduated from IIT Delhi in 2018" → episodic on
  the temporal anchor). Prompt tuning didn't reliably move that needle on
  the free-tier model.

## Agentic build (current)

A second track on top of the shipped memory system: convert the fixed
retrieve → respond → save pipeline into a model-driven **ReAct loop** and grow
the agent's tool set. Five feature phases plus an eval wrap (5+1):

| Phase | Scope | Status |
|------:|-------|--------|
| Phase 1 | ReAct loop + `search_memory` tool. Retrieval becomes a tool the model *chooses* to call (forced `retrieve_memories` node removed); `search_memory` + `save_memory` both bound; model ⇄ tools loop with a per-turn 5-step cap; save still does conflict-resolution + type classification + DELETE-then-INSERT. | ✅ commit `d586607` — branch `agentic-phase-1` |
| Phase 2 | `web_search` tool — keyless DuckDuckGo (`ddgs`) external lookup alongside memory; bound as a third tool on the same retry-once-then-graceful dispatch path; `SYSTEM_PROMPT` routes web_search (current/external) vs search_memory (about the user) vs neither (direct knowledge). | ✅ commit `1529767` — branch `agentic-phase-2` |
| Phase 3 | Goals — a `manage_goal` tool (set / list / update) + a `goal` memory type stored in the same Chroma store with `status` + `created_at`; reached ONLY via manage_goal, never the save_memory auto-classifier; update reuses DELETE-then-INSERT. | **[CURRENT]** — branch `agentic-phase-3` |
| Phase 4 | Decay / consolidation — TTL on episodic memories, periodic dedupe. | planned |
| Phase 5 | Reflection / auto-summarization of accumulated memories. | planned |
| +1 | Eval re-baseline — re-measure the 50-case suite against the agentic agent. Retrieval is now model-driven, so the Week 3 numbers no longer describe this graph; the suite is deliberately untouched until here. | planned |

**Phase 1 — graph shape (replaces the fixed pipeline):**

```
START → start_turn → call_model → (route_after_model) ⇄ tools → END
```

- `start_turn` resets the per-turn step counter (`State.step`). The 5-step cap
  (`MAX_MODEL_STEPS`) is per user turn — load-bearing under the CLI/Streamlit
  checkpointer, where State persists across turns.
- `call_model` binds `[search_memory, save_memory]`; on the 5th model step it
  is invoked **without** tools, so a tool call is impossible and the loop
  terminates with a text answer.
- `tools` (formerly `store_memory`) executes each tool call with **one retry
  then graceful degradation** (an error ToolMessage, never a crashed turn).
  `save_memory` keeps the exact judge + classifier + DELETE-then-INSERT logic;
  `search_memory` invokes the real tool wrapping `store.asearch` — no second
  copy of retrieval logic.
- `SYSTEM_PROMPT` now advertises both tools and drops the forced `{user_info}`
  block; `retrieve_memories` and `_format_user_info` are removed.

**Phase 2 — `web_search` (current):**

- New dependency (the only one this phase adds): **`ddgs`** — the renamed
  `duckduckgo-search` (package + import are both `ddgs`; pinned via `uv add`,
  installed `ddgs==9.14.4`). Keyless, free — preserves the $0 constraint.
  **API gotcha:** the rename also changed the API. The current surface is
  `from ddgs import DDGS; DDGS().text(query, max_results=N)` returning
  `list[dict]` with keys `title` / `href` / `body`, and — importantly — it
  **raises `ddgs.exceptions.DDGSException`** on an empty result set rather than
  returning `[]`. Verified against the installed version, not assumed.
- `tools.web_search(query)` wraps it. No `InjectedToolArg`s (it needs no store
  or user_id). The sync `DDGS().text` runs in `asyncio.to_thread` so it doesn't
  block the event loop under the `tools_node` `asyncio.gather`. A helper
  `_run_ddgs_text` converts the specific "No results found." `DDGSException`
  into an empty list (a normal "no results" outcome the tool renders as a
  readable message) and **re-raises every other `DDGSException`** (rate limit /
  timeout / network) so the graph's retry-once-then-degrade path handles it.
  Output is a short summary of the top `WEB_SEARCH_MAX_RESULTS`=3 results
  (title + truncated snippet + url).
- Bound as a third tool: `TOOLS = [search_memory, save_memory, web_search]`.
  `_execute_tool_call` gains a `web_search` branch (via `_handle_web_search`)
  on the **same** dispatch path — so it inherits the one-retry-then-graceful
  ToolMessage handling, and a failing/empty search degrades to a text answer.
- `SYSTEM_PROMPT` advertises web_search and adds an explicit "choosing a tool"
  block — web_search = current/external facts the user didn't give and aren't
  about them; search_memory = about the user; neither = direct knowledge — to
  curb over-calling web_search. The 5-step cap is unchanged (web_search is in
  `TOOLS`, so it's stripped on the final step like the others).

**Phase 3 — `manage_goal` + `goal` memory type (current):**

- No new dependency. Goals are stored in the SAME Chroma store as memories of
  `type="goal"`, with two extra value fields: `status` (active / done /
  abandoned / …) and `created_at`.
- **Classifier isolation (the load-bearing care).** `MemoryType` is extended to
  `["fact", "preference", "episodic", "goal"]`, but a separate narrow
  `ClassifiableType = Literal["fact", "preference", "episodic"]` is what
  `JudgeDecision.type`, `_ClassifierResponse.type`, and `_classify_save` use —
  so save_memory's auto-classifier/judge **cannot** emit `goal`. Goals are
  reachable ONLY through `manage_goal`. (Bonus: because the judge can only
  output the three classifiable types, its cross-type-replace downgrade means
  save_memory can never replace/delete a goal-type neighbour either.)
- **Store passthrough.** `ChromaStore` previously dropped every value key except
  `content` / `type` / `updated_at`. It now additively carries optional
  `status` / `created_at` through `_put` → metadata and back via a
  `_value_from_md(md)` helper used by `_get` and both `_search` branches.
  Backward compatible: non-goal memories never set these, so their value is
  unchanged. `list_memories` surfaces them via `**item.value`.
- **`tools.manage_goal(action, *, user_id, store, goal, status, new_goal)`** —
  `set` writes a new `goal`-type memory (`status="active"` + `created_at`);
  `list` reuses `list_memories` filtered to `type=="goal"`; `update` semantic-
  searches the user's goals for the closest match and applies the **same
  DELETE-then-INSERT** pattern as a save replace (new UUID, original
  `created_at` preserved) so a status change updates in place, never
  duplicates. `user_id` / `store` are `InjectedToolArg`s.
- Bound as a fourth tool: `TOOLS = [search_memory, save_memory, web_search,
  manage_goal]`. `_execute_tool_call` gains a `manage_goal` branch (via
  `_handle_manage_goal`) on the **same** retry-once-then-graceful dispatch path.
- `SYSTEM_PROMPT` advertises manage_goal and adds routing rules — state an aim
  → `set`; ask about goals → `list`; report progress/completion → `update` —
  plus a "DO NOT save … goals (use manage_goal instead)" line so aims don't go
  to save_memory. The 5-step cap is unchanged (manage_goal is in `TOOLS`, so
  it's stripped on the final step like the others).

**Scope discipline:** Phase 3 adds ONLY goals. Decay / consolidation and
reflection (Phases 4–5) and the eval wrap are NOT built; Phase 3 does not add
TTL / dedupe / summarization and does not touch `tests/eval/`.

## Roadmap (all phases complete)

| Phase | Scope | Status |
|------:|-------|--------|
| Week 1 | Baseline ReAct agent (in-memory store, blind append, dump-all retrieval) + 50-case eval harness + initial README. | ✅ commit `5983707` |
| Week 2 | Chroma + `all-MiniLM-L6-v2` real top-k retrieval (`retrieve_memories` node split out); LLM-judge conflict-resolution save subgraph (top-3 neighbors, insert-vs-replace, DELETE-then-INSERT); CLI gains `--persist-dir .chroma/`. | ✅ commits `6fc4528` (Phase A) and `b0e7e95` (Phase B) |
| Week 3 | Typed memory: judge classifies + gates `replace` on same-type-and-same-facet; dedicated `_classify_save` on the no-neighbor path; `[type]` prefix in the memory render; eval runner gains `type_accuracy`. | ✅ — `contradiction_update` 57.1% → 85.7%, `retrieval_relevance` back to 90% |
| Week 4 | Streamlit UI (`src/sage_agent/app.py`) + hosted demo on Streamlit Community Cloud. | ✅ — live at <https://sage-agent.streamlit.app/> |
| Polish | Eval `_normalize()` for Unicode whitespace; `tests/eval/rescore.py` for offline rescoring; tuned `CLASSIFIER_PROMPT`; fresh full Week 3 eval. | ✅ — both flagship metrics at 100%, type accuracy 76.7% |

**Future work that hasn't shipped:** decay / consolidation (TTL on
episodic memories, periodic dedupe); blog post; second-opinion eval with
a different model (Claude / GPT-4-class) to cross-check the free-tier
numbers; tightening should_save_episodic (case_021).

## Repo layout

```
sage-agent/
├── pyproject.toml              uv-managed; hatchling build; pkg = src/sage_agent
├── uv.lock                     committed; reproducible installs
├── README.md                   public-facing, leads with the live demo URL
├── CLAUDE.md                   this file
├── TROUBLESHOOTING.md          symptom → cause → fix for every gotcha we hit
├── .env.example                OPENROUTER_API_KEY + MODEL_NAME template
├── .env                        local-only (gitignored)
├── .gitignore                  ignores .env, .chroma/, .streamlit/secrets.toml; commits eval results
├── .chroma/                    CLI / Streamlit Chroma persistence dir (gitignored)
├── src/sage_agent/
│   ├── __init__.py             __version__
│   ├── app.py                  Streamlit UI: chat + typed memories sidebar; @st.cache_resource on the graph
│   ├── cli.py                  Terminal REPL: /new /memories /quit; --persist-dir flag
│   ├── context.py              Context dataclass (legacy; graph reads user_id directly from RunnableConfig)
│   ├── graph.py                ReAct loop + MemoryType/ClassifiableType + JudgeDecision + _judge/_classify_save + tool dispatch (save/search/web/goal)
│   ├── model.py                ChatOpenAI factory pointed at OpenRouter; DEFAULT_MODEL = openai/gpt-oss-120b:free
│   ├── prompts.py              SYSTEM_PROMPT + JUDGE_PROMPT + CLASSIFIER_PROMPT
│   ├── state.py                State dataclass: messages + retrieved_memories
│   ├── store.py                ChromaStore(BaseStore) + make_store(persist_dir=None) + list_memories + _value_from_md (goal status/created_at passthrough) + lazy embedder
│   └── tools.py                save_memory + search_memory (InjectedToolArg store/user_id) + web_search (ddgs, no key) + manage_goal (set/list/update)
└── tests/
    ├── __init__.py
    └── eval/
        ├── __init__.py
        ├── cases.json          50 cases across 6 categories
        ├── runner.py           Load → validate → run → score → aggregate → write JSON
        ├── rescore.py          Re-apply current score_case to a stored JSON; no LLM calls
        └── results/            .gitkept; baseline_*.json / week2_*.json / week3_*.json committed
```

## Architecture

```
user turn ──► retrieve_memories ──► call_model
                  (top-k from Chroma)     │
                                ┌────────┴─────────┐
                                ▼                  ▼
                          no tool call       save_memory call
                                │                  │
                                ▼                  ▼
                             respond         store_memory
                                              (conflict check
                                               folded in)
                                                   │
                                                   └──► back to call_model
```

LangGraph nodes (current status):

- **retrieve_memories** *(live, Week 2)* — find the last `HumanMessage`,
  embed via `_get_embedder()` (lazy `all-MiniLM-L6-v2`), `store.asearch`
  for top-`RETRIEVAL_K`=5 in the `("memories", user_id)` namespace, write
  to `state.retrieved_memories`. Loop-back from `store_memory` skips this
  node — the user query hasn't changed mid-turn.
- **call_model** *(live, Week 1)* — system prompt + `state.retrieved_memories`
  rendered as `- [type] content` + history → either a natural response
  or a `save_memory` tool call. Reads memories from state, never directly
  from the store.
- **store_memory** *(live, with Week 2 + Week 3 logic folded in)* — for
  each `save_memory` tool call:
  1. Semantic-search `CONFLICT_NEIGHBORS_K`=3 similar memories.
  2. If zero neighbors → call `_classify_save(candidate)` for the type,
     then `aput` with a new UUID key.
  3. Else → call `_judge_save(candidate, neighbors)` which returns a
     `JudgeDecision(type, action, target_key, content)` in one
     structured-output call. Validator downgrades cross-type replaces
     to inserts.
  4. `replace` → `adelete(target_key)` then `aput(new uuid)`.
  5. `insert` → `aput(new uuid)`.
  6. Emit one `ToolMessage` per tool call (preserves the LLM's
     N tool_calls / N ToolMessages pairing).

### Why conflict-resolution is folded into store_memory, not its own node

The LLM's chat-completions protocol requires N ToolMessages back for N
emitted tool_calls in a single hop. Splitting conflict-resolution into a
graph node either breaks that pairing or forces a second call_model
invocation just to re-emit the tool call. One node, one round-trip.

### Why hand-rolled store_memory instead of ToolNode

`save_memory` takes `user_id` and `store` as `InjectedToolArg`s — hidden
from the LLM's schema, but `ToolNode` won't populate them. The custom
node pulls `user_id` from `RunnableConfig.configurable` and `store` from
the compiled graph context, then invokes the tool itself.

### ChromaStore contract

`ChromaStore(BaseStore)` lives in `src/sage_agent/store.py`. The langgraph
`BaseStore` abstract surface is only `batch` and `abatch`; everything
else (`get`/`put`/`search`/`delete` + async siblings) dispatches through
them. Our implementation:

- Single shared Chroma collection `sage_memories`. Namespace is encoded
  as metadata `{ns0, ns1}` and queried via a where-filter. Composite ID
  `f"{ns0}::{ns1}::{key}"` prevents cross-namespace collisions in the
  shared id-space.
- Embeddings are computed by `_get_embedder()` (lazy
  `SentenceTransformer("all-MiniLM-L6-v2")`) and passed explicitly to
  Chroma — we set `embedding_function=None` on the collection so Chroma
  never auto-embeds.
- Per-memory metadata: `ns0`, `ns1`, `key`, `content`, `type`,
  `updated_at`. Item return shape: `value={"content": ..., "type": ...}`,
  matching what `list_memories` and the runner consume.
- `make_store(persist_dir=None)` → `chromadb.EphemeralClient()` (eval
  hermetic per case). `make_store(persist_dir=".chroma/")` →
  `chromadb.PersistentClient(path=...)` (CLI + Streamlit local).
  **Streamlit Cloud filesystem resets on reboot — `.chroma/` does NOT
  persist there.** See TROUBLESHOOTING.md.

### Checkpointer (thread persistence)

`cli.py` and `app.py` both compile the graph with
`langgraph.checkpoint.memory.MemorySaver` as the checkpointer:

```python
graph = build_graph(checkpointer=MemorySaver(), store=store)
```

This is what makes `/new thread` work in the CLI (and the "New thread"
button in the Streamlit UI): the checkpointer holds per-`thread_id`
conversation state, the store holds per-`user_id` memories. Resetting
the thread clears conversation history but keeps memories — the two
are deliberately decoupled.

`MemorySaver` is in-process only — it doesn't survive a process restart.
That's fine for our shape (a new CLI / Streamlit session is meant to
start with a fresh conversation thread; the persistent `.chroma/` store
carries the durable memories across processes). If you ever need
durable conversation history too (e.g. resume a chat after a crash),
swap `MemorySaver` for `SqliteSaver` or `PostgresSaver` from
`langgraph.checkpoint.*` — same interface, one-line change.

The eval runner does NOT pass a checkpointer to `build_graph` (only
`store`). Each case is a fresh conversation, so checkpointing isn't
needed — and skipping it keeps per-case isolation strict.

## Tech stack and key decisions

| Concern | Choice | Why |
|---|---|---|
| Orchestration | LangGraph (`>=0.6.0`) | Matches upstream template; first-class store + checkpointer; explicit state machine. |
| LLM | `ChatOpenAI` pointed at OpenRouter | `init_chat_model` doesn't natively route to OpenRouter; OpenAI-compatible endpoint works. |
| Default model | `openai/gpt-oss-120b:free` (via OpenRouter) | Strongest tool-calling among current free OpenRouter models. Original baseline used `gemini-2.0-flash-exp:free`; OpenRouter retired it in early 2026. `model.py` is a one-function swap to move to Claude or any other provider. |
| Embeddings | `sentence-transformers` / `all-MiniLM-L6-v2` (local, lazy-loaded) | $0; good enough for thousands-scale stores; lazy load avoids the 3-5s import-time cost on `--help`. |
| Vector store | `chromadb` (embedded) — `EphemeralClient` for tests, `PersistentClient` for CLI / Streamlit local | Zero-ops; pip-install only; per-case isolation for the eval is free with EphemeralClient. |
| UI | Streamlit, `@st.cache_resource` on the graph build | Cached so the embedder loads once per session, not on every rerun. |
| Hosting | Streamlit Community Cloud, free tier | Reads `pyproject.toml` natively; secrets via web UI; no requirements.txt needed unless dep resolution fails. |
| Env management | `python-dotenv` + `.env` (local), `st.secrets` (Cloud) | `model.py` calls `load_dotenv()`; `app.py` reads `st.secrets` with a guarded try/except (no `secrets.toml` locally) and falls through to env. |
| Tests | `pytest` (dev group) — but the **eval runner is not a pytest suite**. | `python -m tests.eval.runner` is an evaluation harness, not unit tests. Pytest is reserved for future unit tests. |

**The $0 / free-tier constraint is load-bearing.** "Clone and demo on a
free key" is part of the project's story. Do not introduce paid
dependencies without flagging the tradeoff.

## Configuration

`.env` (gitignored) supplies:

```
OPENROUTER_API_KEY=sk-or-v1-...
MODEL_NAME=openai/gpt-oss-120b:free
```

For Streamlit Cloud, set the same `OPENROUTER_API_KEY` via the web UI's
Secrets panel. The app falls back through three sources in order:
existing `os.environ`, `st.secrets["OPENROUTER_API_KEY"]` (guarded so it
doesn't raise locally where no `secrets.toml` exists), then
`load_dotenv()` via `model.get_model()`.

## How to run

```powershell
# Activate venv (Windows PowerShell)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& d:\sage-agent\.venv\Scripts\Activate.ps1

# Install
uv sync

# Chat (terminal)
python -m sage_agent.cli --user-id alice
# or: python -m sage_agent.cli --user-id alice --persist-dir .chroma/

# Chat (Streamlit UI, local)
streamlit run src/sage_agent/app.py

# Eval — dry-run (no API calls; validates case schema)
python -m tests.eval.runner --dry-run

# Eval — smoke (first 5 cases)
python -m tests.eval.runner --limit 5

# Eval — single category (~70-120s for retrieval_relevance)
python -m tests.eval.runner --category retrieval_relevance

# Eval — full 50 cases with a label (10-12 minutes; Week 3 adds classifier
# calls on no-neighbor saves)
python -m tests.eval.runner --label week3

# Rescore — re-apply current score_case to an existing JSON (no LLM calls)
python -m tests.eval.rescore tests/eval/results/baseline_<UTC>.json
```

Each non-dry-run writes `tests/eval/results/<label>_<UTC>.json`. **Those
result files are intentionally committed** so README numbers are
reproducible from git history. See the comment block in `.gitignore` —
the `tests/eval/results/*.json` ignore is deliberately commented out.

CLI REPL commands: `/new` (new thread, same user — memories persist),
`/memories` (dump store), `/quit`.

## Eval harness — schema and scoring

`tests/eval/cases.json` is a JSON array of case objects:

```json
{
  "id": "case_XXX",
  "category": "should_save_fact | should_save_preference | should_save_episodic | should_not_save | contradiction_update | retrieval_relevance",
  "setup_memories": [{"content": "...", "type": "fact|preference|episodic"}],
  "conversation": [{"role": "user|assistant", "content": "..."}],
  "expected": {
    "should_save": true,
    "memory_content_contains": ["substring", ...],
    "response_contains": ["substring", ...],
    "update": true
  }
}
```

The runner enforces `REQUIRED_TOP_LEVEL = {id, category, conversation, expected}`
and `VALID_CATEGORIES = {...}`. Per-case scoring (`tests/eval/runner.py::score_case`):

- `should_save_*` pass iff `predicted_save AND any new memory contains all
  memory_content_contains` (case-insensitive, Unicode-whitespace-normalized).
- `should_not_save` pass iff `not predicted_save`.
- `contradiction_update` pass iff `len(all_memories) == 1 AND that one
  memory contains memory_content_contains`. Baseline blind-append leaves
  N+1 memories → expected fail. That's the point.
- `retrieval_relevance` pass iff `not predicted_save AND response contains
  any of response_contains`. The agent answers from memory, not re-saves.

Aggregate output:

- Per-category pass rate
- Global save-decision P / R / F1 (binary classifier across all cases)
- (Week 3+) Type accuracy — per-category and global. Expected type derived
  from category: `should_save_fact → fact`, etc.;
  `contradiction_update → setup_memories[0].type`. Excludes
  `should_not_save` and `retrieval_relevance` (no expected type). Cases
  from runs that didn't capture types (pre-Week 3 JSONs) show `—`, not
  `0%` — `runner.score_case` uses `None` as the explicit "types not
  captured" signal.

**Per-case isolation**: every case gets a fresh `make_store()` (Ephemeral
Chroma client) and `user_id = f"eval_{case['id']}"`. The same user is
never reused across cases.

**Important Unicode quirk**: `_contains_*` in `runner.py` collapses any
Unicode whitespace to a single ASCII space before substring matching.
The free-tier model occasionally emits U+202F (NARROW NO-BREAK SPACE)
between words like "March" and "15", which a strict byte matcher would
false-fail. See `_normalize()`.

## Polish-pass workflow: `tests/eval/rescore.py`

When the scoring logic changes (e.g. Unicode normalization) but the
agent's outputs don't, re-running the LLM across 50 cases is wasteful.
`rescore.py` reads a stored results JSON, re-applies `score_case` and
`aggregate` to each case's stored outcome (new_memories, all_memories,
types, final_response), and writes the file back with the new scores.

**Important `None` vs `[]` distinction**: pre-Week 3 JSONs don't have
`new_memory_types` / `all_memory_types` keys at all. The rescore utility
preserves their absence (passes `None`) rather than collapsing to `[]`,
so `score_case` correctly marks `type_ok = None` (not applicable) for
those runs. If the rescore corrupts an old JSON by injecting `[]`, the
fix is `git checkout HEAD -- <file>` and re-run.

## Coding conventions

- **Commit messages**: lowercase `feature: ...` is the established style.
  Look at `git log --oneline` — every commit follows it. Don't switch to
  Conventional Commits (`feat:`) mid-stream. Use `fix: ...` / `refactor: ...`
  / `docs: ...` in the same lowercase style.
- **Branching**: work on `dev`; PRs (eventually) target `main`.
- **Commit & push**: **the user owns these.** Claude prepares the working
  tree, then stops. Don't run `git commit` or `git push` unilaterally —
  the user is explicit about wanting to control these.
- **Imports**: `from __future__ import annotations` at the top of every
  module that uses type hints. `__all__` is not currently used.
- **Type hints**: required on function signatures. `dict` / `list` /
  `tuple` over `Dict` / `List` / `Tuple` — Python 3.11+.
- **Async**: graph and tools are async (`ainvoke`, `aput`). The eval
  runner is async at the top with `asyncio.run` in `main()`. The CLI
  matches. Don't introduce sync paths for nodes.
- **Docstrings**: every module has a top-level docstring explaining its
  role. Match the voice — terse, specific, names a tradeoff if there is
  one.
- **No premature abstraction**: one tool, one store, one model. Don't
  add `BaseTool` / `BaseStoreFactory` / etc. unless a real second
  implementation arrives.

## What to be careful about

- **Don't replace the model in `model.py` without testing tool-calling
  quality.** The judge uses structured output via
  `with_structured_output(JudgeDecision)`. Free-tier models vary widely
  on this; gpt-oss-120b was selected explicitly for tool-calling
  strength. The OpenRouter `/api/v1/models` endpoint with
  `supported_parameters contains "tools"` and `id ends with ":free"` is
  the filter we used.
- **Don't change `cases.json` without flagging a re-baseline.** Adding
  cases is fine; editing existing ones changes the meaning of week-vs-
  week comparisons. If you must, note it in the commit.
- **Don't commit `.env`** — gitignored, but be paranoid. The real key is
  in there.
- **Don't bump model temperature.** `model.py` defaults to 0.0 — eval
  reproducibility (such as it is on the free tier) depends on it. The
  free tier still has some non-determinism we can't control.
- **`tests/eval/results/` JSONs are committed** — see `.gitignore`
  comment block. Don't add a blanket ignore.
- **Memory namespace is `("memories", user_id)`** — matches upstream. If
  you ever change the namespace shape, also update `list_memories`,
  `memory_namespace`, `_ns_filter` / `_ns_metadata` in `store.py`, the
  tool, and the runner.
- **DELETE-then-INSERT, not upsert, for `replace`.** A same-key overwrite
  of a `setup_*` key would pass the per-category predicate for
  `contradiction_update` but mark the case as `predicted_save = False`
  (the runner filters new memories by `not key.startswith("setup_")`),
  tanking save-decision recall. The new UUID is load-bearing.
- **`.chroma/` doesn't persist on Streamlit Cloud** — the Cloud
  filesystem resets on reboot. This is expected, not a bug.

## Working on this repo as Claude

- **Plan-first for multi-step features.** The user prefers a plan
  layout (chat-output, not `/plan`-mode unless explicitly invoked)
  before implementation. For changes touching multiple files, briefly
  lay out phases / files / verification, then execute.
- **The user commits.** Stop after the working tree is ready. Suggest a
  commit message; don't run `git commit` / `git push`.
- **Run long evals in background.** A full 50-case run is 9-13 minutes;
  use background execution (`run_in_background: true` in Bash/PowerShell
  invocations) and wait for the completion notification, don't poll.
- **Smoke before full.** A `--limit 5` smoke (~45-60s) catches obvious
  breakage; running it before a full 50 saves ~10 min when something is
  wrong.
- **PowerShell is the user's shell** (Windows). The activation pattern
  is `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned;
  & d:\sage-agent\.venv\Scripts\Activate.ps1`. After that, `python -m
  tests.eval.runner` works directly; no `uv run` prefix needed.
- **When asked to "fix a failing eval case"**: check whether it's a
  consistent fail (e.g. case_021 episodic FN — has been failing every
  run since baseline) or model non-determinism (case_043, case_040 have
  both flipped pass/fail across re-runs). Don't tune the prompt for a
  flaky case; document it.
- **Don't unilaterally accelerate the roadmap.** All four weeks are
  done; the "Future work" list is the next pool to pull from. If the
  user asks for a feature, scope it against that list first.

## External references

- Upstream template: <https://github.com/langchain-ai/memory-agent>
- LangGraph docs: <https://langchain-ai.github.io/langgraph/>
- OpenRouter (free key, model list): <https://openrouter.ai/keys>
- Chroma docs: <https://www.trychroma.com/>
- `all-MiniLM-L6-v2` model card: <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>
- Streamlit Community Cloud: <https://streamlit.io/cloud>
- Live demo: <https://sage-agent.streamlit.app/>
