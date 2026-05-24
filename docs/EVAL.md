# Evaluation methodology

This doc is the canonical reference for sage-agent's evaluation harness:
what the 50 cases test, how scoring works, why the choices were made, and
how to extend the suite without breaking comparability across phases.

If you only read one section, read **Design principles** — it's the load-
bearing context that explains the rest.

---

## Why an eval exists at all

A memory-augmented agent has too many moving parts (prompt, retrieval,
classifier, conflict policy, store) for "does it feel better?" to be a
sustainable signal. The eval converts every change into a measurable delta
against a fixed baseline. The roadmap's ship gates — "improvement on
`contradiction_update`," "improvement on `retrieval_relevance`" — only mean
something because the harness exists *before* the work.

The eval is therefore not a test suite. It will fail many cases at
baseline by design. A failing case is information, not a bug.

---

## Design principles

1. **Baseline must be honest.** The Phase 1 agent is deliberately
   degraded — blind-append save, no semantic retrieval, no typed memory.
   The eval is calibrated so its expected baseline shape is *partial pass*:
   high pass-rate on save/no-save decisions (the prompt does the work),
   near-zero on `contradiction_update` and middling on `retrieval_relevance`.
   That spread is the headroom Weeks 2–4 consume.
2. **Cases are frozen, runs are not.** Adding cases is fine; editing or
   deleting them invalidates the cross-phase comparison. If a case must
   change, treat it as a re-baseline event — re-run baseline and note it
   in CHANGELOG.
3. **Per-case isolation.** Every case gets a fresh `InMemoryStore` and a
   unique `user_id = f"eval_{case_id}"`. The same user is never reused.
   Don't introduce shared-state shortcuts; cross-case contamination would
   make individual case failures un-debuggable.
4. **Substring matching, not exact match.** The agent writes memories in
   its own words ("User's name is Aman" vs "user is named Aman"). Scoring
   on substrings tolerates that variation while still catching real
   failures.
5. **Save-decision is a binary classifier.** Across all 50 cases,
   `should_save` is treated as the gold label and the agent's actual save
   action is the prediction. Precision / recall / F1 fall out of that.
6. **No mocks of the LLM.** The agent runs against a real OpenRouter
   model. Reproducibility comes from `temperature=0.0`, a fixed model
   slug, and committing the result JSONs — not from stubbing the LLM.

---

## Cases — what's in the suite

`tests/eval/cases.json` is a JSON array of 50 case objects across six
categories:

| Category | Count | What it probes | Baseline expectation |
|---|---:|---|---|
| `should_save_fact` | 10 | Stable identity facts (name, location, job, age, contacts, etc.) the agent should persist | High pass — prompt explicitly enumerates this |
| `should_save_preference` | 8 | Stable preferences (likes, dislikes, tool choices, dietary) the agent should persist | High pass |
| `should_save_episodic` | 5 | Notable personal events (promotion, trip, milestone) the agent should persist | Medium-to-high pass |
| `should_not_save` | 10 | Small talk, general questions, instructions — the agent should NOT save these | High pass — prompt explicitly enumerates the exclusions |
| `contradiction_update` | 7 | An existing memory is pre-loaded; the user contradicts it; the agent should *update* (single memory remaining) rather than append | **Near zero at baseline.** Blind append always produces N+1 memories |
| `retrieval_relevance` | 10 | An existing memory is pre-loaded; the user asks a question whose answer is in memory; the agent should answer from memory without re-saving | Variable at baseline — the "retrieve" stub dumps everything, so the LLM *can* see the memory, but isn't always primed to use it |

Categories are validated against `VALID_CATEGORIES` in `runner.py`. Adding
a category requires editing both the runner constant and this doc.

---

## Case schema

```json
{
  "id": "case_XXX",
  "category": "should_save_fact | should_save_preference | should_save_episodic | should_not_save | contradiction_update | retrieval_relevance",
  "setup_memories": [
    {"content": "User's name is Aman.", "type": "fact"}
  ],
  "conversation": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "expected": {
    "should_save": true,
    "memory_type": "fact | preference | episodic",
    "memory_content_contains": ["substring", "..."],
    "response_contains": ["substring", "..."],
    "update": true
  }
}
```

Field rules (enforced by `validate_cases` in the runner):

- `id`, `category`, `conversation`, `expected` are required.
- `id` must be unique across the file.
- `category` must be in `VALID_CATEGORIES`.
- `conversation` must be a non-empty list; each turn's `role` must be
  `user` or `assistant`.
- `setup_memories[].content` is required when `setup_memories` is present.
- All `expected` fields are optional individually but specific categories
  expect specific fields (see Scoring below).

`assistant` turns in `conversation` are loaded into the message history
but the runner only *invokes the graph* on `user` turns — assistant turns
are scaffolding for multi-turn cases (none currently exist; all 50 cases
are single-turn, but the schema supports multi-turn).

---

## Scoring

For each case, the runner produces an outcome with:

- `n_setup` — count of `setup_memories` pre-loaded.
- `n_total_after` — total memories in the store after the conversation.
- `new_memories` — memories whose key does NOT start with `setup_`, i.e.
  memories the agent saved during the run.
- `all_memories` — every memory in the store after the run (setup + new).
- `final_response` — the last AI message without a tool call.

`predicted_save = len(new_memories) > 0` — i.e. the agent saved *something*
new.

Per-category pass rules:

| Category | Pass condition |
|---|---|
| `should_save_fact` / `_preference` / `_episodic` | `predicted_save` AND any new memory contains **all** `memory_content_contains` substrings (case-insensitive) |
| `should_not_save` | NOT `predicted_save` |
| `contradiction_update` | `len(all_memories) == 1` AND that single memory contains all `memory_content_contains` substrings |
| `retrieval_relevance` | NOT `predicted_save` AND the final response contains **any** `response_contains` substring |

Two substring matchers:

- `_contains_all` — every needle must appear in the haystack (used for
  memory content checks: all required substrings must be in the *same*
  memory).
- `_contains_any` — any one needle suffices (used for response checks:
  the agent answering "yes, Aman" or "you're Aman" both pass).

Both are case-insensitive.

### Why `contradiction_update` checks `len(all_memories) == 1`

The pass condition is intentionally strict. The point of the category is
to test *update semantics*, not "did the agent write the new value
somewhere." Baseline behaviour — append the new value as memory #2,
leaving the contradicted old value as memory #1 — is the failure mode
the eval is designed to catch. A two-memory store with the new value
present still fails, and it should.

### Why `retrieval_relevance` requires NOT predicted_save

The agent should *use* the existing memory to answer, not re-save the
same information. A "what's my name?" answer where the agent also writes
a new "User's name is Aman" memory means retrieval didn't fire — the
agent treated the user message as a fresh fact disclosure.

---

## Aggregate metrics

After all cases run, the runner emits:

```json
{
  "by_category": {
    "<category>": {
      "total": <int>,
      "passed": <int>,
      "errors": <int>,
      "pass_rate": <float>
    }, ...
  },
  "save_decision": {
    "tp": <int>, "fp": <int>, "fn": <int>, "tn": <int>,
    "precision": <float>, "recall": <float>, "f1": <float>
  }
}
```

`save_decision` treats `should_save` (from `expected`) as gold and the
agent's `predicted_save` as the prediction across *all 50 cases*. So:

- TP: agent saved when it should have.
- FN: agent didn't save when it should have.
- FP: agent saved when it shouldn't have (over-eager saving).
- TN: agent correctly stayed quiet.

This is a global signal independent of category-specific pass rates. F1
is the headline number that compresses both directions of error into one
score.

---

## Running the harness

```bash
# Validate cases without hitting the API
uv run python -m tests.eval.runner --dry-run

# Smoke test — first 5 cases only
uv run python -m tests.eval.runner --limit 5

# One category
uv run python -m tests.eval.runner --category contradiction_update

# Full run — baseline label
uv run python -m tests.eval.runner

# Full run with a different label (Week N comparison)
uv run python -m tests.eval.runner --label week2

# Override the output path
uv run python -m tests.eval.runner --out /tmp/my-run.json
```

Each non-dry-run writes `tests/eval/results/<label>_<UTC>.json`. The
runner also prints a summary table to stdout:

```
==============================================================
Category                      Pass  Total    Rate   Err
--------------------------------------------------------------
contradiction_update             0      7    0.0%     0
retrieval_relevance              4     10   40.0%     0
should_not_save                  9     10   90.0%     0
should_save_episodic             5      5  100.0%     0
should_save_fact                10     10  100.0%     0
should_save_preference           8      8  100.0%     0
==============================================================
Save-decision  P=0.971  R=1.000  F1=0.985  (tp=33 fp=1 fn=0 tn=9)
```

(Numbers above are illustrative — real baseline lives in `results/`.)

### Reproducibility contract

- `temperature=0.0` in `model.py` is part of the contract. Don't change it.
- The model slug is read from `$MODEL_NAME` (default
  `google/gemini-2.0-flash-exp:free`). Record the slug in the result JSON
  if you change it for a run — it materially affects the numbers.
- OpenRouter's free tier can rate-limit or have model-side variance. Two
  back-to-back full runs will not be byte-identical; the harness is
  designed for *trend* comparison, not byte equality.
- Results JSONs are committed (`tests/eval/results/` is gitkept and the
  `*.json` ignore is deliberately commented out in `.gitignore`).

---

## Result JSON shape

```json
{
  "label": "baseline",
  "timestamp_utc": "20260524T143000Z",
  "cases_path": "tests/eval/cases.json",
  "n_cases": 50,
  "elapsed_seconds": 78.2,
  "aggregate": { /* see Aggregate metrics above */ },
  "cases": [
    {
      "id": "case_001",
      "category": "should_save_fact",
      "n_setup": 0,
      "n_total_after": 1,
      "new_memories": ["User's name is Aman."],
      "all_memories": ["User's name is Aman."],
      "final_response": "Got it, Aman — nice to meet you.",
      "expected_save": true,
      "predicted_save": true,
      "content_ok": true,
      "response_ok": true,
      "passed": true
    },
    ...
  ]
}
```

Per-case error rows additionally carry `"error"` and `"traceback"` fields
when the run raised — these don't count against pass rate but are
surfaced in the `errors` column of the summary.

---

## Adding a case

1. Pick a category. If none fits, *think hard* before adding a category —
   it requires updating `VALID_CATEGORIES` in the runner, this doc, and
   the README results table, and means re-running baseline.
2. Mint an ID with the next sequential number (`case_051`, `case_052`,
   …). IDs are forever — don't renumber existing cases to "tidy up."
3. Write the conversation. Single-turn is fine; multi-turn is supported
   but currently unused.
4. Write `expected`. The fields the runner reads depend on the category
   — refer to the Scoring table.
5. Run `--dry-run` to catch schema errors.
6. Run the case in isolation:
   `uv run python -m tests.eval.runner --category <category>` — confirm
   the agent's actual behaviour matches your intent for "what passes."
7. If the new case meaningfully changes the corpus distribution
   (e.g. doubles the size of one category), re-baseline and note it in
   CHANGELOG.

### Writing good cases

- **One concept per case.** "User shares name AND email AND birthday" in
  one turn tests three things at once; if the case fails, you don't know
  which. Split it.
- **Substrings should be load-bearing.** `"Aman"` is a good content
  check; `"name"` is not — the agent might write "I'll remember your
  name" without ever saving "Aman."
- **For `should_not_save`, exercise distinct decoys.** Greetings, world
  knowledge, instructions, third-person facts, and arithmetic each fail
  in different ways. The current corpus covers each.
- **For `contradiction_update`, pre-load exactly one related memory.**
  The pass condition requires `len(all) == 1`. Multiple setup memories
  make the case un-passable without an unrelated change to scoring.
- **For `retrieval_relevance`, ask in a way that doesn't restate the
  fact.** "What's my name?" is good; "My name is Aman, right?" leaks the
  answer into the prompt and the agent might pass for the wrong reason.

---

## When the harness should change (and when it shouldn't)

| Change | Allowed? | Notes |
|---|---|---|
| Add new cases | Yes | Doesn't invalidate prior runs; baseline pass-counts go up but rates are still comparable |
| Edit existing case content | No, except in a re-baseline event | Document in CHANGELOG; re-run baseline label |
| Delete a case | No, except in a re-baseline event | Same as above |
| Add a category | Yes, with care | Update `VALID_CATEGORIES`, this doc, README table; re-baseline |
| Change scoring rules | Re-baseline event | Old result JSONs are no longer comparable to new ones |
| Change the model | Record in result JSON | The slug matters; baseline-vs-Week-N comparisons are only valid against the same model |
| Bump `temperature` | No | Reproducibility floor |
| Replace `InMemoryStore` with Chroma in the harness | Part of Week 2 | The harness should swap when the agent swaps. Run baseline-on-Chroma to confirm parity before adding new behaviour on top |

---

## Roadmap touchpoints

- **Week 2** — `contradiction_update` should go from ~0% to materially
  non-zero once the conflict-resolution subgraph lands. `retrieval_relevance`
  should also lift once real semantic retrieval replaces the dump-all stub.
- **Week 3** — typed memory enables a future evaluation of
  type-classification accuracy. That likely warrants either a new
  category or a side metric ("for each `should_save_*` case, was the
  inferred type correct?"). Design it before the implementation.
- **Week 4** — decay / consolidation invites a longitudinal eval mode:
  run a sequence of cases against the *same* store and observe what
  survives. The current single-store-per-case isolation is not the right
  shape for that — it would be a separate harness mode, not a
  modification of the existing one.

---

## See also

- `tests/eval/cases.json` — the corpus itself.
- `tests/eval/runner.py` — `validate_cases`, `score_case`, `aggregate`,
  `print_summary` are the source of truth for everything in this doc.
- `tests/eval/results/` — committed baseline and per-phase result JSONs.
- `CLAUDE.md` — the "don't accidentally break the eval" guardrails.
- `README.md` — public-facing results table.
