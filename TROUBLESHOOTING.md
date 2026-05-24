# TROUBLESHOOTING.md

Real problems we hit, and how we fixed them. Organized symptom → cause →
fix. If something here doesn't match what you're seeing, check
[`CLAUDE.md`](CLAUDE.md) for the latest architecture state.

---

## Table of contents

- [Setup & environment](#setup--environment)
- [API & model](#api--model)
- [Eval runner & scoring](#eval-runner--scoring)
- [Memory store / Chroma](#memory-store--chroma)
- [Streamlit (local & Cloud)](#streamlit-local--cloud)
- [Dev workflow](#dev-workflow)
- [Known limits we chose to live with](#known-limits-we-chose-to-live-with)

---

## Setup & environment

### `uv: command not found` after a fresh clone

**Cause.** `uv` isn't on PATH. The official Astral installer puts it in
`~/.local/bin` or `~/.cargo/bin`; the pip route puts it under
`%APPDATA%\Python\Python<ver>\Scripts\`.

**Fix.**

```powershell
# Install via pip (works on miniconda Python too)
python -m pip install --user uv

# Find where it landed
Get-ChildItem "$env:APPDATA\Python" -Recurse -Filter "uv.exe" |
  Select-Object FullName

# Prepend to PATH for this session
$env:Path = "C:\Users\<you>\AppData\Roaming\Python\Python<ver>\Scripts;" + $env:Path
uv --version
```

For persistence, add that scripts dir to your User PATH via Windows
Settings → "Edit environment variables for your account".

---

### `Set-ExecutionPolicy` warning when activating the venv

**Symptom.** PowerShell refuses to run `Activate.ps1`.

**Fix.** Use the per-session pattern we've adopted:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& d:\sage-agent\.venv\Scripts\Activate.ps1
```

`-Scope Process` means we don't have to change the user-level policy.
After this, `python -m tests.eval.runner` works directly — no `uv run`
prefix needed within the activated venv.

---

### Git warns about LF → CRLF on every commit

**Symptom.**

```
warning: in the working copy of 'README.md', LF will be replaced by CRLF
the next time Git touches it
```

**Cause.** Files were authored with LF line endings; Windows Git is
configured for CRLF on checkout / LF on commit (`core.autocrlf=true`).

**Verdict.** Cosmetic. Ignore. The repo's content is unaffected. If you
genuinely want to silence it, add a `.gitattributes` with `* text=auto
eol=lf` — but the existing setup is fine.

---

## API & model

### `NotFoundError: Error code: 404 ... No endpoints found for <model>`

**Symptom.** The eval or CLI errors out with a 404 from OpenRouter on
every case.

**Cause.** OpenRouter retired the model slug you're pointing at. We hit
this with `google/gemini-2.0-flash-exp:free` in early 2026 — it was
removed without rotation.

**Fix.**

1. Get the current free-tier tool-supporting models:

   ```powershell
   $key = (Get-Content "D:\sage-agent\.env" |
            Select-String '^OPENROUTER_API_KEY=').ToString().Split('=',2)[1].Trim()
   $resp = Invoke-RestMethod -Uri "https://openrouter.ai/api/v1/models" `
            -Headers @{Authorization="Bearer $key"}
   $resp.data |
     Where-Object { $_.id -like "*:free" -and $_.supported_parameters -contains "tools" } |
     Select-Object id, @{n='ctx';e={$_.context_length}} |
     Sort-Object id
   ```

2. Pick one with strong tool-calling (we use `openai/gpt-oss-120b:free`).

3. Update three places:
   - `.env` → `MODEL_NAME=<new-slug>`
   - `.env.example` → same slug + comment
   - `src/sage_agent/model.py` → `DEFAULT_MODEL = "<new-slug>"`

4. Smoke: `python -m tests.eval.runner --limit 5`. Five passes = good.

---

### `OPENROUTER_API_KEY is not set` on first run

**Cause.** No `.env` file, or `.env` is present but the key isn't loaded.

**Fix (local).**

```powershell
cp .env.example .env
# edit .env, paste the real key
```

`src/sage_agent/model.py` calls `load_dotenv()` at import; the key lands
in `os.environ` after that.

**Fix (Streamlit Cloud).** Open your app on streamlit.io/cloud →
**Advanced settings → Secrets** and paste:

```toml
OPENROUTER_API_KEY = "sk-or-v1-..."
```

`src/sage_agent/app.py` reads this via `st.secrets` and writes it into
`os.environ` before `model.get_model()` runs.

---

### Your OpenRouter key showed up in chat / logs / a screenshot

**Fix.** Rotate immediately at <https://openrouter.ai/keys>. Update
`.env` locally and the Streamlit Cloud Secrets entry. Don't waste effort
trying to scrub the original — assume it's compromised.

---

## Eval runner & scoring

### `case_047` (retrieval_relevance, "When's my birthday?") false-fails

**Symptom.** The agent's response IS "Your birthday is on March 15." but
the eval marks it failed.

**Cause.** The model emits U+202F (NARROW NO-BREAK SPACE) between
"March" and "15". The bytes are `4D 61 72 63 68 E2 80 AF 31 35`, not
`... 20 ...`. Strict-byte `"March 15" in response` returns False even
though the rendered text is identical.

**Fix.** Already shipped: `_normalize()` in `tests/eval/runner.py`
collapses any `\s+` (Unicode-aware) to a single ASCII space before
substring matching. Apply it via `_contains_all` / `_contains_any`.

If a future case hits the same shape, the fix is one line. If you change
the `_contains_*` semantics in any other way, **re-run the rescore**:

```powershell
python -m tests.eval.rescore tests/eval/results/baseline_<UTC>.json
python -m tests.eval.rescore tests/eval/results/week2_<UTC>.json
# ... etc.
```

---

### After rescoring, old JSONs show 0% type accuracy on every category

**Cause.** The first version of `rescore.py` collapsed missing
`new_memory_types` / `all_memory_types` to `[]`, which `score_case` then
interpreted as "captured but empty" → `type_ok = False`. Pre-Week 3 runs
never captured those fields at all.

**Fix.** Two changes (both already applied):

1. `runner.score_case` distinguishes `None` (not captured → `type_ok =
   None`) from `[]` (captured but no memory saved → `type_ok = False`).
2. `rescore.py` passes `saved.get("new_memory_types")` directly — no `or
   []` collapse.

**Recovery.** If the rescore already corrupted a JSON:

```powershell
git checkout HEAD -- tests/eval/results/<file>.json
python -m tests.eval.rescore tests/eval/results/<file>.json
```

---

### `--limit 5` smoke passes but full run errors out

**Cause.** The first 5 cases are all `should_save_fact` with an empty
store — they never trigger the conflict-resolution path. Errors in the
judge / classifier won't show up until later cases.

**Fix.** Run a `--category contradiction_update` (7 cases, ~80s) before
committing to a full 50 if you've changed anything in the save path. It
exercises the judge specifically.

---

### Free-tier rate limits during a full 50-case run

**Symptom.** Cases start failing with `429`-flavored errors midway
through a full run.

**Cause.** OpenRouter free-tier has request-per-minute limits. Week 3+
adds a classifier call per save → ~75-90 LLM calls in a full eval
(50 turn-responses + ~30 saves with classifier or judge). On a tight
window this can clip.

**Fix.** The runner has no retry layer by design — failures are
informative. Wait 60s and re-run. If you're running multiple evals
back-to-back, space them out by ~5 minutes.

---

### A full eval takes ~10-13 minutes — is something wrong?

**Cause.** Nothing. Each case is 1-3 LLM round-trips at ~3-6s each on the
free tier. Week 3 added a classifier call on no-neighbor saves (~30
cases × ~3s extra).

**Fix.** Run in the background:

```powershell
# Run-in-background; you get a notification when done
python -m tests.eval.runner --label week3
```

Use the smoke / category-only flags to iterate fast:

```powershell
python -m tests.eval.runner --dry-run                       # ~1s
python -m tests.eval.runner --limit 5                       # ~45-60s
python -m tests.eval.runner --category contradiction_update # ~80s
python -m tests.eval.runner --category retrieval_relevance  # ~110s
python -m tests.eval.runner                                 # 9-13min
```

---

### A `contradiction_update` case passed last run but failed this run (or vice versa)

**Cause.** Free-tier LLM non-determinism. Even at `temperature=0`, we've
seen case_040 (Camry→Tesla), case_043 (vegetarian-restaurant), and the
judge occasionally flip across re-runs.

**Fix.** Don't tune the judge prompt for a single flaky case —
particularly case_040, where Week 2 tried adding a substitution few-shot
and the net was -1 (gained case_036, lost cases 034 and 039).

If you need a stable number for a screenshot, run twice and report the
worse one. The committed README numbers reflect a single representative
run, not best-of-N.

---

## Memory store / Chroma

### `ChromaStore` instance won't construct: "Can't instantiate abstract class"

**Cause.** You overrode a high-level method (`asearch` / `aput` / etc.)
but didn't implement `batch` / `abatch`. langgraph's `BaseStore` has
*only* `batch` and `abatch` as `@abstractmethod`s; everything else
dispatches through them via `GetOp` / `PutOp` / `SearchOp`.

**Fix.** Implement `batch` and `abatch` with an Op-type dispatch. See
`src/sage_agent/store.py::ChromaStore.batch` for the pattern. `PutOp`
with `value=None` is the delete signal; `SearchOp` with `query is None`
is the "all items in namespace" path.

---

### Chroma starts downloading onnxruntime / the default embedding model on import

**Cause.** Chroma's default `embedding_function` is an ONNX-backed
`all-MiniLM-L6-v2`. If you let it default, Chroma fetches a 90MB+ model
the first time a collection is created.

**Fix.** We set `embedding_function=None` on
`get_or_create_collection(...)` and always pass `embeddings=[...]`
explicitly on `upsert` / `query`. Our embedder is the
`sentence-transformers` package directly, lazy-loaded via
`_get_embedder()`.

If you ever need to add a code path that doesn't pre-compute embeddings,
pass them through `_embed(content)` first — don't lean on Chroma's
default.

---

### Embedder load time spikes the first user turn

**Cause.** `_get_embedder()` lazy-loads
`SentenceTransformer("all-MiniLM-L6-v2")` (~3-5s) on first call. We
deliberately don't preload at module import because both `cli.py` and
the eval runner import `sage_agent.graph` (which calls `make_store()`),
and we don't want `--help` to pay that cost.

**Fix.** Already as designed. If you want to preload (e.g. for a
production server), call `_get_embedder()` once at startup in the
service's main.

---

### `contradiction_update` case "replaces" but `predicted_save = False`

**Cause.** The judge picked `replace` and the code did a same-key
overwrite of a `setup_*` key. The runner's save-decision metric
filters new memories by `not key.startswith("setup_")`, so a same-key
overwrite isn't counted as a save. The case can pass per-category (one
memory, right content) but the case shows up as a save-decision
false-negative.

**Fix.** Already shipped: the `replace` path is **DELETE-then-INSERT**.
`store.adelete(target_key)` removes the setup key; `store.aput(new
uuid)` writes a fresh memory. The new UUID isn't prefixed `setup_` so
the runner counts it as a save. Don't refactor this to upsert.

---

### `list_memories` returns at most 10 memories per user

**Cause.** `BaseStore.search()` default `limit=10`. Easy to miss.

**Fix.** Already handled: `store.py::list_memories` passes
`limit=1000`. If you add a new "give me everything for this user" call
site, do the same — otherwise the runner's `n_total_after` check for
`contradiction_update` silently truncates.

---

### Has `.chroma/` CLI persistence been verified across two processes?

**Status.** No — not manually. The Week 4 plan included a cross-process
sanity check ("start CLI as alice, save a memory, `/quit`, restart CLI,
ask for it back") and it was skipped during the push to ship.

**What IS exercised by existing tests:**

- The eval runner creates a fresh `make_store()` (Ephemeral) per case
  and runs the full conflict-resolution + classifier pipeline against
  it → the Ephemeral path is well-exercised.
- The Streamlit app uses `make_store(persist_dir=".chroma/")` and
  survives within-session page reloads → confirms the Persistent path
  writes and reads back during a single process.

**What ISN'T exercised:** spawning two consecutive
`python -m sage_agent.cli --user-id alice` processes, saving in the
first, asking in the second, and confirming memories survive the
inter-process boundary.

**Why it should work anyway.** `chromadb.PersistentClient(path=...)`
writes to the local SQLite + parquet files on disk; the second process
re-opens the same files. The composite ID scheme + namespace metadata
filter is symmetrical across writes/reads. If it ever fails, the most
likely cause is the SQLite file being locked (concurrent CLI sessions
on the same `.chroma/` aren't supported by Chroma's default backend).

**To verify (when you get around to it):**

```powershell
& d:\sage-agent\.venv\Scripts\Activate.ps1

# Session 1
python -m sage_agent.cli --user-id alice
# you> My name is Manvendra and I prefer green tea over coffee.
# you> /quit

# Session 2 — same .chroma/, same user_id
python -m sage_agent.cli --user-id alice
# you> /memories
# expect: two memories with [fact] / [preference] tags
# you> What's my drink preference?
# expect: green tea
```

---

## Streamlit (local & Cloud)

### `streamlit run` boots fine locally but the deployed app errors with `OPENROUTER_API_KEY is not set`

**Cause.** You forgot to set the Secret on Streamlit Cloud. Local `.env`
isn't deployed.

**Fix.** Streamlit Cloud dashboard → your app → **⋯ menu → Settings →
Secrets** → paste:

```toml
OPENROUTER_API_KEY = "sk-or-v1-..."
```

Save → the app reboots automatically with the new env.

---

### Importing `sage_agent.app` outside `streamlit run` raises `StreamlitSecretNotFoundError`

**Symptom.**

```
streamlit.errors.StreamlitSecretNotFoundError: No secrets found.
Valid paths for a secrets.toml file ...
```

**Cause.** `if "OPENROUTER_API_KEY" in st.secrets` triggers
`st.secrets._parse()`, which raises when no `secrets.toml` exists. This
hits during a plain `python -c "import sage_agent.app"` import-test.

**Fix.** Already shipped:

```python
if not os.environ.get("OPENROUTER_API_KEY"):
    try:
        if "OPENROUTER_API_KEY" in st.secrets:
            os.environ["OPENROUTER_API_KEY"] = st.secrets["OPENROUTER_API_KEY"]
    except Exception:
        pass
```

The bare `except` is deliberate — Streamlit's exact exception class
isn't stable across versions, and the fallback path (`.env` /
`os.environ`) handles everything else.

---

### Streamlit Cloud cold-start takes ~30-60s

**Cause.** First boot installs deps from `pyproject.toml` (~200MB
including torch + sentence-transformers) and downloads the
`all-MiniLM-L6-v2` weights. After the first cache-resource call, the
embedder stays warm for the session.

**Fix.** Already as fast as it gets on the free tier. Add a tagline /
loading spinner in the UI if you want users to know what's happening
(`@st.cache_resource(show_spinner="Loading agent (first run downloads
the embedder)…")` already does this).

---

### Memories in the Streamlit Cloud demo disappear between reboots

**Cause.** Streamlit Cloud's filesystem is ephemeral. The `.chroma/`
directory writes succeed during a session but reset when the container
restarts (which Streamlit does on idle timeout, deploy, or memory
pressure).

**Fix.** This is expected, not a bug. The demo is designed to show
within-session persistence and typed memory, not durable storage. If you
need true persistence, swap `ChromaStore`'s persist backend to a hosted
service (Chroma Cloud, Pinecone, or a managed Postgres + pgvector)
behind a single env-var. That's a Week-5+ change.

If you want to make the demo less surprising, add a banner to
`app.py` noting the ephemeral persistence.

---

### `pyproject.toml` deps don't resolve on Streamlit Cloud

**Symptom.** Deploy log shows `ResolutionImpossible` or similar.

**Fix.** Generate and commit a flat `requirements.txt`:

```powershell
uv export --format requirements-txt > requirements.txt
git add requirements.txt
git commit -m "fix: pin requirements.txt for Streamlit Cloud deploy"
git push
```

Streamlit Cloud prefers `requirements.txt` if present and falls back to
`pyproject.toml` otherwise.

---

## Dev workflow

### PowerShell single-quoted here-string strips quotes from Python code

**Symptom.** You tried to pipe a multi-line Python snippet via
`@'...'@` into `python -c` and got a `SyntaxError` because the embedded
apostrophes (e.g. `"User's"`) terminated the string.

**Fix.** Don't pipe inline. Write the snippet to a `scratch_<name>.py`
file and run it:

```powershell
python "D:\sage-agent\scratch_check.py"
```

Delete the scratch when done. PowerShell here-string quoting rules are
brittle enough that the file-route is faster overall.

---

### PowerShell can't print `→` (UnicodeEncodeError, `'charmap' codec`)

**Symptom.**

```
UnicodeEncodeError: 'charmap' codec can't encode character '→'
in position N: character maps to <undefined>
```

**Cause.** PowerShell on Windows defaults to the cp1252 console
encoding, which lacks `→`, `↦`, etc.

**Fix.** Use ASCII (`->`) in any string you `print()` from a script.
We applied this to `tests/eval/rescore.py`'s log messages. If you must
print Unicode, set `$OutputEncoding = [System.Text.UTF8Encoding]::new()`
and `[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()` at
the top of your session — but ASCII is one line of code instead of
shell setup.

---

### "I want to split my working tree into two commits" without `git add -p`

**Cause.** `git add -p` and `git add -i` are interactive and not usable
in our automation context.

**Fix (the pattern we used for Phase A / B split).**

1. Save the "later" version of mixed files to memory (just read them in
   your head — they're already on disk).
2. For files that span both commits (e.g. graph.py with retrieval +
   conflict-resolution): overwrite with the earlier-phase-only version
   via `Write`.
3. For files that are purely later-phase (e.g. cli.py for Phase B):
   `git restore <file>` to revert to HEAD.
4. `git add` + commit the earlier phase.
5. Re-apply the later-phase changes via `Edit` / `Write`.
6. `git add` + commit the later phase.

It's tedious but unambiguous. Don't use `git stash --keep-index` —
mixing it with the rest of this workflow has bitten me before.

---

### How long does X take on the free tier?

| Operation | Duration |
|---|---|
| `--dry-run` | ~1s |
| `--limit 5` smoke | ~45-60s |
| `--category contradiction_update` (7 cases) | ~80s |
| `--category retrieval_relevance` (10 cases) | ~110s |
| Full 50-case Week 2 eval (judge calls on saves) | ~9 min |
| Full 50-case Week 3 eval (+ classifier calls on no-neighbor saves) | ~10-13 min |
| `python -m tests.eval.rescore <file>` | <2s (no LLM calls) |
| `streamlit run src/sage_agent/app.py` first load | ~5-7s |
| Streamlit Cloud first deploy (dep install + boot) | ~3-5 min |
| Streamlit Cloud subsequent reboots | ~30-60s |
| `uv sync` first install (incl. torch + chromadb) | ~1.5-2 min |
| `uv sync` incremental | ~5-10s |

---

## Known limits we chose to live with

### `case_021` (`should_save_episodic`) — persistent false negative

**Symptom.** Across every eval since baseline, case_021 has shown
`new_memories=[]` — the model decided not to save.

**Status.** The case is borderline (looking at it, the wording is
ambiguous between an off-hand mention and a save-worthy event). We've
opted not to chase it — fixing it would mean tuning the system prompt
toward over-saving, which would hurt `should_not_save`. The 80% on
`should_save_episodic` is therefore by design.

---

### `case_043` (`retrieval_relevance`) — flaky on Week 2

**Symptom.** Setup memory is "User is vegetarian"; user asks "Recommend
a restaurant for dinner tonight"; expected response contains
"vegetarian". On Week 2 the model sometimes asks for clarification
instead of applying the memory; on Week 3 (with `[type]` tags rendered
in the prompt) it usually applies it.

**Status.** Live with it. It's a model-behavior issue at temperature 0
on the free tier. The `[type]` prefix in the rendered memory list
nudges the model toward "established facts" framing, which is why
Week 3 typically passes this case where Week 2 didn't.

---

### `case_040` (`contradiction_update`, Camry → Tesla) — judge fuzziness

**Symptom.** Setup: "User drives a silver Toyota Camry"; user says
"I sold the Camry — I'm driving a Tesla Model 3 now"; expected: single
memory mentioning Tesla.

**Status.** Sometimes passes (Week 3 polish run), sometimes fails (every
Week 2 run, original Week 3 run). The judge correctly identifies the
neighbor most of the time but occasionally picks `insert` for these
substitution patterns. Week 2 tried fixing it with a tuned few-shot;
net was -1 (gained case_036, lost cases 034 + 039). The current 4-shot
prompt is the best we've found without re-introducing other failures.

---

### Classifier under-fires on `preference` and `episodic`

**Symptom.** Type accuracy: 100% on facts and contradiction_update,
~60-65% on preferences, ~40% on episodic.

**Cause.** The free-tier model is conservative — when in doubt, it
falls back to `fact`. Borderline cases:

- "User does not drink coffee" → fact (negative-statement framing).
- "User loved reading Project Hail Mary" → preference (past tense
  doesn't override the verb).
- "User was promoted to senior engineer" → fact (it's both a one-time
  event AND a current role; the model takes the role framing).
- "User graduated from IIT Delhi in 2018" → episodic (the temporal
  anchor wins) — this was actually a fact case the model correctly
  labeled before tuning, and the tuned prompt's emphasis on "anchored
  to a moment in time" tipped it.

**Status.** Live with it. We tried adding explicit guidance + few-shots
in `CLASSIFIER_PROMPT`; preference and episodic counts barely moved
(net even or -1 across attempts). If type-accuracy matters more than
the current trade, you'd need a stronger model (Claude / GPT-4-class)
or fine-tuning — both break the $0 / free-tier constraint.

---

## When in doubt

- Check `git log --oneline` to see what shipped when.
- Check `tests/eval/results/` for committed eval JSONs — they're the
  ground truth for "what numbers did week N actually produce".
- Read [`CLAUDE.md`](CLAUDE.md) for current architecture + conventions.
- Read [`README.md`](README.md) for the public-facing narrative + table.
