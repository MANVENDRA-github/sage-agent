"""Re-score an existing eval results JSON with the current runner logic.

Used when scoring rules change (e.g. Unicode whitespace normalization)
but the agent's outputs haven't — saves a re-run of the full eval against
the live LLM.

Usage:
    python -m tests.eval.rescore tests/eval/results/baseline_20260524T094912Z.json
    python -m tests.eval.rescore tests/eval/results/baseline_20260524T094912Z.json --out tests/eval/results/baseline_rescored.json

If --out is omitted, overwrites the input file in place.

Reads each case's stored outcome (new_memories, all_memories, types,
final_response), re-applies tests.eval.runner.score_case and
aggregate, and writes back.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tests.eval.runner import aggregate, load_cases, print_summary, score_case

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_CASES = EVAL_DIR / "cases.json"


def rescore_file(in_path: Path, out_path: Path, cases_path: Path) -> None:
    data = json.loads(in_path.read_text(encoding="utf-8"))
    cases_by_id = {c["id"]: c for c in load_cases(cases_path)}

    rescored: list[dict] = []
    for saved in data.get("cases", []):
        case = cases_by_id.get(saved["id"])
        if case is None:
            rescored.append(saved)
            continue
        if saved.get("error"):
            rescored.append(saved)
            continue
        # Preserve absence: None if the key wasn't in the saved JSON (pre-
        # Week 3 runs didn't capture types). Empty list means "captured but
        # nothing saved" — different signal.
        outcome = {
            "id": saved["id"],
            "category": saved["category"],
            "n_setup": saved.get("n_setup", 0),
            "n_total_after": saved.get("n_total_after", 0),
            "new_memories": saved.get("new_memories", []),
            "new_memory_types": saved.get("new_memory_types"),
            "all_memories": saved.get("all_memories", []),
            "all_memory_types": saved.get("all_memory_types"),
            "final_response": saved.get("final_response", ""),
        }
        rescored.append(score_case(case, outcome))

    agg = aggregate(rescored)
    data["aggregate"] = agg
    data["cases"] = rescored
    out_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    print(f"\nRescored {in_path} -> {out_path}")
    print(f"Label: {data.get('label', '?')}  cases: {data.get('n_cases', len(rescored))}")
    print_summary(agg)


def main() -> None:
    p = argparse.ArgumentParser(prog="tests.eval.rescore")
    p.add_argument("input", help="Path to a results JSON to rescore.")
    p.add_argument("--out", default=None, help="Output path (default: overwrite input).")
    p.add_argument("--cases", default=str(DEFAULT_CASES))
    args = p.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Input not found: {in_path}", file=sys.stderr)
        sys.exit(2)
    out_path = Path(args.out) if args.out else in_path
    rescore_file(in_path, out_path, Path(args.cases))


if __name__ == "__main__":
    main()
