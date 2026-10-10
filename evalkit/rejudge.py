"""Re-judge cached QA answers with one judge model, so runs that used different
answering models are scored by the same grader.

    python -m evalkit.rejudge --longmemeval longmemeval_s.json --judge deepseek \\
        --run deepseek:deepseek-flash:off:30 --run claude-cli:claude-haiku-4-5-20251001:90

Each --run is <cached backend name>:<n questions>. The script rebuilds every
prompt exactly as evalkit.qa did, pulls the answer from the reply cache (a
miss means the prompt differs from what that run saw, so misses are reported
as a consistency check), and judges all answers with --judge. No answering
calls are made.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ctxgc.llm import CachedLLM, make_llm
from ctxgc.summarize import ExtractiveSummarizer

from .chat import longmemeval_cases
from .qa import ANSWER_SCHEMA, ANSWER_SYSTEM, JUDGE_SCHEMA, JUDGE_SYSTEM, answer_prompt, contexts_for, judge_prompt


class _Named:
    def __init__(self, name: str):
        self.name = name

    def json(self, *a):
        raise RuntimeError("lookup only")


def rejudge(cases, methods, budgets, run_name: str, judge, cache_path: Path) -> dict:
    lookup = CachedLLM(_Named(run_name), cache_path)
    summarizer = ExtractiveSummarizer()
    keys = [("full", 1.0)] + [(m, f) for m in methods for f in budgets]
    hits = misses = 0
    correct = {k: [] for k in keys}
    by_type = {}
    for case in cases:
        ctxs = contexts_for(case, methods, budgets, summarizer)
        q = case["messages"][case["question_index"]]["content"]
        gold = str(case["answers"][0]) if case["answers"] else ""
        for k in keys:
            cached = lookup.cached(ANSWER_SYSTEM, answer_prompt(ctxs[k], q), ANSWER_SCHEMA)
            if cached is None:
                misses += 1
                continue
            hits += 1
            ok = bool(judge.json(JUDGE_SYSTEM, judge_prompt(q, gold, cached["answer"]), JUDGE_SCHEMA)["correct"])
            correct[k].append(ok)
            by_type.setdefault((k, case["type"]), []).append(ok)
    table = {f"{k[0]}@{k[1]}": (sum(v) / len(v) if v else float("nan")) for k, v in correct.items()}
    types = sorted({t for _, t in by_type})
    per_type = {f"{k[0]}@{k[1]}": {t: (sum(by_type[(k, t)]) / len(by_type[(k, t)]) if (k, t) in by_type else float("nan"))
                                   for t in types} for k in keys}
    return {"run": run_name, "n": len(cases), "answers_found": hits, "answers_missing": misses,
            "accuracy": table, "by_type": per_type, "types": types}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--longmemeval", type=Path, required=True)
    ap.add_argument("--max-turns", type=int, default=100000)
    ap.add_argument("--methods", nargs="*", default=["truncate", "graded", "graded-agent+lex"])
    ap.add_argument("--budgets", nargs="*", type=float, default=[0.2, 0.35])
    ap.add_argument("--run", action="append", required=True, help="<cached backend name>:<n>")
    ap.add_argument("--judge", default="deepseek")
    ap.add_argument("--cache", type=Path, default=Path("evalkit/results/llm_cache.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("evalkit/results/qa_longmemeval_full_rejudged.json"))
    args = ap.parse_args()
    judge = make_llm(args.judge, effort="off", cache_path=args.cache)
    results = []
    for spec in args.run:
        name, _, n = spec.rpartition(":")
        cases = longmemeval_cases(args.longmemeval, int(n), args.max_turns)
        r = rejudge(cases, args.methods, args.budgets, name, judge, args.cache)
        results.append(r)
        print(f"\n## {name} ({r['n']} questions), judged by {judge.name}: answers found {r['answers_found']}, missing {r['answers_missing']}")
        print("| context | accuracy | " + " | ".join(r["types"]) + " |")
        print("|---|---|" + "---|" * len(r["types"]))
        for k, acc in r["accuracy"].items():
            print(f"| {k} | {acc:.2f} | " + " | ".join(f"{r['by_type'][k][t]:.2f}" for t in r["types"]) + " |")
    args.out.write_text(json.dumps({"judge": judge.name, "runs": results}, indent=1))
    inner = getattr(judge, "inner", judge)
    print(f"\n[judge] {judge.name}: cache hits {judge.hits}, misses {judge.misses}, usage {getattr(inner, 'usage', {})}")


if __name__ == "__main__":
    main()
