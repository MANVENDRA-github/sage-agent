"""Eval harness for the baseline (and future) sage-agent.

Run:
    python -m tests.eval.runner                 # full run, writes results JSON
    python -m tests.eval.runner --dry-run       # validates cases, no LLM calls
    python -m tests.eval.runner --limit 5       # first 5 cases (smoke)
    python -m tests.eval.runner --category should_save_fact

Schema (see tests/eval/cases.json):
    id, category, conversation, expected
    optional: setup_memories=[{content, type}]

Scoring:
    memory_content_contains : ALL substrings must appear across new memories
    response_contains       : ANY substring must appear in final response
    contradiction_update    : pass iff exactly ONE memory exists for the user
                              AND it contains the new value's substrings

Metrics emitted:
    - per-category pass rate (cases passed / cases run)
    - global save-decision precision / recall / F1 (should_save as classifier)
    - JSON report written to tests/eval/results/<label>_<UTC>.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from sage_agent.graph import build_graph
from sage_agent.store import list_memories, make_store

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_CASES = EVAL_DIR / "cases.json"
DEFAULT_RESULTS_DIR = EVAL_DIR / "results"

REQUIRED_TOP_LEVEL = {"id", "category", "conversation", "expected"}
VALID_CATEGORIES = {
    "should_save_fact",
    "should_save_preference",
    "should_save_episodic",
    "should_not_save",
    "contradiction_update",
    "retrieval_relevance",
}


def load_cases(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        cases = json.load(fh)
    if not isinstance(cases, list):
        raise ValueError(f"{path} must contain a JSON array of cases")
    return cases


def validate_cases(cases: list[dict]) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    for i, case in enumerate(cases):
        if not isinstance(case, dict):
            errors.append(f"[{i}] not an object")
            continue
        missing = REQUIRED_TOP_LEVEL - case.keys()
        if missing:
            errors.append(f"[{i}] missing keys: {sorted(missing)}")
        cid = case.get("id")
        if cid in seen_ids:
            errors.append(f"[{i}] duplicate id: {cid}")
        seen_ids.add(cid)
        if case.get("category") not in VALID_CATEGORIES:
            errors.append(f"[{cid}] unknown category: {case.get('category')}")
        conv = case.get("conversation", [])
        if not isinstance(conv, list) or not conv:
            errors.append(f"[{cid}] conversation must be a non-empty list")
        for j, turn in enumerate(conv):
            if turn.get("role") not in {"user", "assistant"}:
                errors.append(f"[{cid}] turn[{j}] bad role: {turn.get('role')}")
        for j, mem in enumerate(case.get("setup_memories", []) or []):
            if "content" not in mem:
                errors.append(f"[{cid}] setup_memories[{j}] missing content")
    return errors


def _contains_all(needles: list[str], haystack: str) -> bool:
    h = haystack.lower()
    return all(n.lower() in h for n in needles)


def _contains_any(needles: list[str], haystack: str) -> bool:
    h = haystack.lower()
    return any(n.lower() in h for n in needles)


async def run_case(case: dict) -> dict:
    user_id = f"eval_{case['id']}"
    store = make_store()
    graph = build_graph(store=store)
    config = {"configurable": {"user_id": user_id, "thread_id": case["id"]}}

    setup_memories = case.get("setup_memories") or []
    for i, mem in enumerate(setup_memories):
        await store.aput(
            ("memories", user_id),
            key=f"setup_{i}",
            value={"content": mem["content"]},
        )
    n_setup = len(setup_memories)

    final_response = ""
    messages: list[Any] = []
    for turn in case["conversation"]:
        if turn["role"] != "user":
            continue
        messages.append(HumanMessage(content=turn["content"]))
        result = await graph.ainvoke({"messages": messages}, config=config)
        messages = list(result["messages"])
        for msg in reversed(messages):
            if getattr(msg, "type", None) == "ai" and not getattr(msg, "tool_calls", None):
                final_response = msg.content or ""
                break

    all_memories = list_memories(store, user_id)
    new_memories = [m for m in all_memories if not m["key"].startswith("setup_")]
    return {
        "id": case["id"],
        "category": case["category"],
        "n_setup": n_setup,
        "n_total_after": len(all_memories),
        "new_memories": [m["content"] for m in new_memories],
        "all_memories": [m["content"] for m in all_memories],
        "final_response": final_response,
    }


def score_case(case: dict, outcome: dict) -> dict:
    expected = case["expected"]
    new_contents = outcome["new_memories"]
    all_contents = outcome["all_memories"]
    response = outcome["final_response"]
    category = case["category"]

    predicted_save = len(new_contents) > 0
    expected_save = bool(expected.get("should_save", False))

    content_required = expected.get("memory_content_contains") or []
    content_ok = (
        not content_required
        or any(_contains_all(content_required, c) for c in new_contents)
    )

    response_required = expected.get("response_contains") or []
    response_ok = (
        not response_required
        or _contains_any(response_required, response)
    )

    if category == "contradiction_update":
        # Pass = exactly one memory exists AND it carries the new value.
        # Baseline appends → len(all) == n_setup + 1 → fail.
        single_memory = len(all_contents) == 1
        updated_value_present = (
            not content_required
            or any(_contains_all(content_required, c) for c in all_contents)
        )
        passed = single_memory and updated_value_present
    elif category == "retrieval_relevance":
        passed = (not predicted_save) and response_ok
    elif category == "should_not_save":
        passed = not predicted_save
    else:  # should_save_*
        passed = predicted_save and content_ok

    return {
        **outcome,
        "expected_save": expected_save,
        "predicted_save": predicted_save,
        "content_ok": content_ok,
        "response_ok": response_ok,
        "passed": passed,
    }


def aggregate(scored: list[dict]) -> dict:
    by_cat: dict[str, dict] = defaultdict(lambda: {"total": 0, "passed": 0, "errors": 0})
    tp = fp = fn = tn = 0

    for s in scored:
        cat = by_cat[s["category"]]
        cat["total"] += 1
        if s.get("error"):
            cat["errors"] += 1
            continue
        if s["passed"]:
            cat["passed"] += 1
        if s["expected_save"] and s["predicted_save"]:
            tp += 1
        elif s["expected_save"] and not s["predicted_save"]:
            fn += 1
        elif not s["expected_save"] and s["predicted_save"]:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "by_category": {
            k: {
                **v,
                "pass_rate": (v["passed"] / v["total"]) if v["total"] else 0.0,
            }
            for k, v in by_cat.items()
        },
        "save_decision": {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        },
    }


def print_summary(agg: dict) -> None:
    print("\n" + "=" * 62)
    print(f"{'Category':<28} {'Pass':>5} {'Total':>6} {'Rate':>7} {'Err':>5}")
    print("-" * 62)
    for cat in sorted(agg["by_category"]):
        v = agg["by_category"][cat]
        print(
            f"{cat:<28} {v['passed']:>5} {v['total']:>6} "
            f"{v['pass_rate']:>6.1%} {v['errors']:>5}"
        )
    print("=" * 62)
    sd = agg["save_decision"]
    print(
        f"Save-decision  P={sd['precision']:.3f}  R={sd['recall']:.3f}  F1={sd['f1']:.3f}  "
        f"(tp={sd['tp']} fp={sd['fp']} fn={sd['fn']} tn={sd['tn']})"
    )
    print()


async def _async_main(args: argparse.Namespace) -> int:
    cases_path = Path(args.cases)
    cases = load_cases(cases_path)
    errors = validate_cases(cases)
    if errors:
        print("Schema errors:")
        for e in errors:
            print(f"  - {e}")
        return 2

    if args.category:
        cases = [c for c in cases if c["category"] == args.category]
    if args.limit:
        cases = cases[: args.limit]

    print(f"Loaded {len(cases)} case(s) from {cases_path}")
    counts = defaultdict(int)
    for c in cases:
        counts[c["category"]] += 1
    for cat in sorted(counts):
        print(f"  {cat:<28} {counts[cat]}")

    if args.dry_run:
        print("\nDry run OK — schema valid, no LLM calls made.")
        return 0

    scored: list[dict] = []
    t0 = time.time()
    for i, case in enumerate(cases, 1):
        sys.stdout.write(f"[{i:>3}/{len(cases)}] {case['id']} ({case['category']})... ")
        sys.stdout.flush()
        try:
            outcome = await run_case(case)
            scored.append(score_case(case, outcome))
            print("pass" if scored[-1]["passed"] else "fail")
        except Exception as e:
            scored.append({
                "id": case["id"],
                "category": case["category"],
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
                "passed": False,
                "expected_save": bool(case["expected"].get("should_save", False)),
                "predicted_save": False,
            })
            print(f"ERROR: {type(e).__name__}: {e}")
    elapsed = time.time() - t0

    agg = aggregate(scored)
    print_summary(agg)
    print(f"Elapsed: {elapsed:.1f}s")

    DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = Path(args.out) if args.out else DEFAULT_RESULTS_DIR / f"{args.label}_{stamp}.json"
    out_path.write_text(
        json.dumps(
            {
                "label": args.label,
                "timestamp_utc": stamp,
                "cases_path": str(cases_path),
                "n_cases": len(cases),
                "elapsed_seconds": round(elapsed, 2),
                "aggregate": agg,
                "cases": scored,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"Wrote results to {out_path}")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(prog="tests.eval.runner")
    p.add_argument("--cases", default=str(DEFAULT_CASES))
    p.add_argument("--out", default=None, help="Override output path.")
    p.add_argument("--label", default="baseline", help="Filename prefix for results.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--category", choices=sorted(VALID_CATEGORIES), default=None)
    p.add_argument("--dry-run", action="store_true", help="Validate cases; no LLM calls.")
    args = p.parse_args()
    sys.exit(asyncio.run(_async_main(args)))


if __name__ == "__main__":
    main()
