"""Action-selection eval — does the agentic agent call the RIGHT tool?

The memory eval (``runner.py``) scores memory OUTCOMES (save decision, type,
retrieval, contradiction). It does not score tool CHOICE. Now that the agent
has four tools (search_memory, save_memory, web_search, manage_goal), picking
the right one is the new capability — this harness measures it.

It reuses ``runner.run_case`` (which now records the tool calls the model
emitted per turn) and scores, per case:

    expected_tools (a set)  vs  called_tools (a set)
    passed = (called == expected)

i.e. the model must call the expected tool AND not over-call (no extra tools).
``should_no_tool`` cases have ``expected_tools = []`` — pass = called nothing.

Cases live in ``tests/eval/action_cases.json`` (separate from the 50 memory
cases, which are untouched). Categories map to the tool they should elicit:

    should_search_memory -> search_memory
    should_web_search    -> web_search
    should_manage_goal   -> manage_goal
    should_save_memory   -> save_memory
    should_no_tool       -> (none)

Run:
    python -m tests.eval.action_runner                  # one run, writes JSON
    python -m tests.eval.action_runner --dry-run        # validate, no LLM calls
    python -m tests.eval.action_runner --limit 5        # smoke
    python -m tests.eval.action_runner --runs 3         # 3 runs, report range

Free-tier tool selection is non-deterministic, so ``--runs N`` runs the whole
suite N times, writes one results JSON per run, and prints the per-run overall
accuracy plus the min-max range.
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

from tests.eval.runner import DEFAULT_RESULTS_DIR, run_case

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_CASES = EVAL_DIR / "action_cases.json"

# Category -> the single tool that category should elicit (None = no tool).
ACTION_CATEGORIES: dict[str, str | None] = {
    "should_search_memory": "search_memory",
    "should_web_search": "web_search",
    "should_manage_goal": "manage_goal",
    "should_save_memory": "save_memory",
    "should_no_tool": None,
}
VALID_TOOLS = {"search_memory", "save_memory", "web_search", "manage_goal"}
REQUIRED_TOP_LEVEL = {"id", "category", "conversation", "expected_tools"}


def load_cases(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        cases = json.load(fh)
    if not isinstance(cases, list):
        raise ValueError(f"{path} must contain a JSON array of cases")
    return cases


def validate_cases(cases: list[dict]) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for i, case in enumerate(cases):
        if not isinstance(case, dict):
            errors.append(f"[{i}] not an object")
            continue
        missing = REQUIRED_TOP_LEVEL - case.keys()
        if missing:
            errors.append(f"[{i}] missing keys: {sorted(missing)}")
        cid = case.get("id")
        if cid in seen:
            errors.append(f"[{i}] duplicate id: {cid}")
        seen.add(cid)
        cat = case.get("category")
        if cat not in ACTION_CATEGORIES:
            errors.append(f"[{cid}] unknown category: {cat}")
        et = case.get("expected_tools")
        if not isinstance(et, list):
            errors.append(f"[{cid}] expected_tools must be a list")
        else:
            for t in et:
                if t not in VALID_TOOLS:
                    errors.append(f"[{cid}] unknown expected tool: {t}")
            # Sanity: a single-tool category should expect exactly that tool.
            want = ACTION_CATEGORIES.get(cat)
            if want is None and et:
                errors.append(f"[{cid}] should_no_tool must have empty expected_tools")
            if want is not None and want not in et:
                errors.append(f"[{cid}] expected_tools should contain {want!r}")
        conv = case.get("conversation", [])
        if not isinstance(conv, list) or not conv:
            errors.append(f"[{cid}] conversation must be a non-empty list")
    return errors


def score_action(case: dict, outcome: dict) -> dict:
    expected = set(case.get("expected_tools") or [])
    called = set(outcome.get("tool_calls") or [])
    hit = expected.issubset(called)          # called every expected tool
    over_called = sorted(called - expected)  # tools called that weren't wanted
    missed = sorted(expected - called)        # expected tools not called
    passed = called == expected
    return {
        "id": case["id"],
        "category": case["category"],
        "expected_tools": sorted(expected),
        "called_tools": sorted(called),
        "tool_calls_raw": outcome.get("tool_calls", []),
        "tool_calls_per_turn": outcome.get("tool_calls_per_turn", []),
        "final_response": outcome.get("final_response", ""),
        "hit": hit,
        "over_called": over_called,
        "missed": missed,
        "passed": passed,
    }


def aggregate(scored: list[dict]) -> dict:
    by_cat: dict[str, dict] = defaultdict(
        lambda: {"total": 0, "passed": 0, "errors": 0, "over_call_fail": 0, "wrong_or_missed": 0}
    )
    total = passed = errors = 0
    for s in scored:
        cat = by_cat[s["category"]]
        cat["total"] += 1
        total += 1
        if s.get("error"):
            cat["errors"] += 1
            errors += 1
            continue
        if s["passed"]:
            cat["passed"] += 1
            passed += 1
        elif s.get("hit") and s.get("over_called"):
            cat["over_call_fail"] += 1  # called the right tool but also extra
        else:
            cat["wrong_or_missed"] += 1  # missed the expected tool / wrong tool
    return {
        "by_category": {
            k: {**v, "accuracy": (v["passed"] / v["total"]) if v["total"] else 0.0}
            for k, v in by_cat.items()
        },
        "overall": {
            "total": total,
            "passed": passed,
            "errors": errors,
            "accuracy": round(passed / total, 3) if total else 0.0,
        },
    }


def print_summary(agg: dict) -> None:
    print("\n" + "=" * 78)
    print(f"{'Category':<24} {'Pass':>5} {'Total':>6} {'Acc':>7} {'OverCall':>9} {'Wrong':>6} {'Err':>4}")
    print("-" * 78)
    for cat in sorted(agg["by_category"]):
        v = agg["by_category"][cat]
        print(
            f"{cat:<24} {v['passed']:>5} {v['total']:>6} {v['accuracy']:>6.1%} "
            f"{v['over_call_fail']:>9} {v['wrong_or_missed']:>6} {v['errors']:>4}"
        )
    print("=" * 78)
    o = agg["overall"]
    print(
        f"OVERALL action-selection accuracy: {o['accuracy']:.1%} "
        f"({o['passed']}/{o['total']}, errors={o['errors']})"
    )
    print()


async def run_suite(cases: list[dict], run_idx: int, n_runs: int) -> tuple[dict, list[dict]]:
    scored: list[dict] = []
    t0 = time.time()
    for i, case in enumerate(cases, 1):
        sys.stdout.write(
            f"[run {run_idx}/{n_runs}][{i:>2}/{len(cases)}] {case['id']} ({case['category']})... "
        )
        sys.stdout.flush()
        try:
            outcome = await run_case(case)
            s = score_action(case, outcome)
            scored.append(s)
            print(f"{'pass' if s['passed'] else 'FAIL'}  called={s['called_tools']}")
        except Exception as e:
            scored.append({
                "id": case["id"],
                "category": case["category"],
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
                "passed": False,
            })
            print(f"ERROR: {type(e).__name__}: {e}")
    agg = aggregate(scored)
    agg["elapsed_seconds"] = round(time.time() - t0, 2)
    print_summary(agg)
    return agg, scored


def _write_results(label: str, run_idx: int, cases_path: Path, agg: dict, scored: list[dict]) -> Path:
    DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = f"_run{run_idx}" if run_idx else ""
    out_path = DEFAULT_RESULTS_DIR / f"{label}{suffix}_{stamp}.json"
    out_path.write_text(
        json.dumps(
            {
                "label": label,
                "run_index": run_idx,
                "timestamp_utc": stamp,
                "cases_path": str(cases_path),
                "n_cases": len(scored),
                "aggregate": agg,
                "cases": scored,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return out_path


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

    print(f"Loaded {len(cases)} action case(s) from {cases_path}")
    counts: dict[str, int] = defaultdict(int)
    for c in cases:
        counts[c["category"]] += 1
    for cat in sorted(counts):
        print(f"  {cat:<24} {counts[cat]}")

    if args.dry_run:
        print("\nDry run OK — schema valid, no LLM calls made.")
        return 0

    run_accuracies: list[float] = []
    for r in range(1, args.runs + 1):
        agg, scored = await run_suite(cases, r, args.runs)
        out_path = _write_results(args.label, r if args.runs > 1 else 0, cases_path, agg, scored)
        print(f"Wrote results to {out_path}")
        run_accuracies.append(agg["overall"]["accuracy"])

    if args.runs > 1:
        print("\n" + "#" * 78)
        print("ACTION-SELECTION ACCURACY ACROSS RUNS")
        for i, acc in enumerate(run_accuracies, 1):
            print(f"  run {i}: {acc:.1%}")
        print(
            f"  range: {min(run_accuracies):.1%} – {max(run_accuracies):.1%}"
            f"  (mean {sum(run_accuracies) / len(run_accuracies):.1%})"
        )
        print("#" * 78)
    return 0


def main() -> None:
    p = argparse.ArgumentParser(prog="tests.eval.action_runner")
    p.add_argument("--cases", default=str(DEFAULT_CASES))
    p.add_argument("--label", default="action", help="Filename prefix for results.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--category", choices=sorted(ACTION_CATEGORIES), default=None)
    p.add_argument("--runs", type=int, default=1, help="Run the whole suite N times.")
    p.add_argument("--dry-run", action="store_true", help="Validate cases; no LLM calls.")
    args = p.parse_args()
    sys.exit(asyncio.run(_async_main(args)))


if __name__ == "__main__":
    main()
