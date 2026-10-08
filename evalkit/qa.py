"""Model-answered QA on compressed chat histories (the benchmarks' own protocol).

    python -m evalkit.qa --longmemeval longmemeval_s.json --n 120 --budgets 0.2 0.35 --dry-run
    python -m evalkit.qa --longmemeval longmemeval_s.json --n 120 --budgets 0.2 0.35 --llm deepseek

For every question: compress the history with each method at each budget,
have a model answer the question from the compressed history (and once from
the full history, the ceiling), then have a model judge the answer against the
gold answer. Reports accuracy per method x budget and per question type.

--dry-run makes no model calls: it builds every context and reports the token
volume and the price at DeepSeek's off-peak / peak Flash rates, so the cost of
a run is known before it is paid for. Model replies are cached on disk like
everywhere else, so a rerun with the same cache is free.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from ctxgc.compress import compress
from ctxgc.llm import make_llm
from ctxgc.summarize import ExtractiveSummarizer
from ctxgc.tokens import count

from .chat import METHODS as CHAT_METHODS
from .chat import chat_graph, locomo_cases, longmemeval_cases

DEFAULT_METHODS = ["truncate", "graded", "graded-agent+lex"]
DEFAULT_BUDGETS = [0.2, 0.35]

# DeepSeek V4.1-Flash, USD per 1M tokens (api-docs.deepseek.com/quick_start/pricing)
PRICE = {"off_peak": {"in": 0.15, "out": 0.6}, "peak": {"in": 0.3, "out": 1.2}}
CNY_PER_USD = 7.2
TOKEN_SAFETY = 1.15        # chars/4 undercounts DeepSeek's tokenizer a little on punctuation-heavy text

ANSWER_SYSTEM = """You answer a question about a user's past conversations with an assistant.
You see the conversation history, possibly compressed: lines tagged |summary or |brief are
summaries, and bracketed lines like [#m12 … #m57 — 46 chunks: …] stand for parts that were left
out. Use only the history. Be concise: give the answer itself, no preamble. If the history does
not contain the information, answer exactly: not in history."""
ANSWER_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}},
                 "required": ["answer"], "additionalProperties": False}

JUDGE_SYSTEM = """You grade an answer to a question about a conversation history against the gold answer.
Judge facts, not wording: a paraphrase, a different unit of the same quantity, or extra correct
detail is still correct. For counting or date questions the number / date must match. For
preference questions, the answer is correct if it reflects the stated preference. An answer of
"not in history" is wrong when the gold answer exists."""
JUDGE_SCHEMA = {"type": "object", "properties": {"correct": {"type": "boolean"}},
                "required": ["correct"], "additionalProperties": False}


def answer_prompt(context: str, question: str) -> str:
    return f"HISTORY:\n{context}\n\nQUESTION: {question}"


def judge_prompt(question: str, gold: str, answer: str) -> str:
    return f"QUESTION: {question}\nGOLD ANSWER: {gold}\nMODEL ANSWER: {answer}"


def contexts_for(case: dict, methods: list[str], budgets: list[float], summarizer) -> dict[tuple[str, float], str]:
    """(method, budget) -> compressed history text; ("full", 1.0) is the ceiling.
    The question is the final user turn of the graph and is rendered too; it is
    cut off here so the model sees it once, in the prompt."""
    g_plain = chat_graph(case["messages"], case["question_index"])
    g_lex = None
    full_tokens = compress(g_plain, 10**9, method="full", summarizer=summarizer).tokens
    out = {("full", 1.0): _strip_question(compress(g_plain, 10**9, method="full", summarizer=summarizer).text, case)}
    for name in methods:
        kw = dict(CHAT_METHODS[name])
        g = g_plain
        if kw.pop("lexical", False):
            if g_lex is None:
                g_lex = chat_graph(case["messages"], case["question_index"], lexical=True)
            g = g_lex
        for frac in budgets:
            r = compress(g, int(full_tokens * frac), summarizer=summarizer, **kw)
            out[(name, frac)] = _strip_question(r.text, case)
    return out


def _strip_question(text: str, case: dict) -> str:
    q = case["messages"][case["question_index"]]["content"].strip()
    lines = [ln for ln in text.splitlines() if q not in ln]
    return "\n".join(lines)


def run(cases: list[dict], methods: list[str], budgets: list[float], llm=None, dry_run: bool = False,
        workers: int = 4, verbose: bool = True) -> dict:
    summarizer = ExtractiveSummarizer()
    rows = []
    est_in = est_out = 0
    from concurrent.futures import ThreadPoolExecutor

    def ask(item):
        key, ctx, case = item
        gold = str(case["answers"][0]) if case["answers"] else ""
        q = case["messages"][case["question_index"]]["content"]
        ans = llm.json(ANSWER_SYSTEM, answer_prompt(ctx, q), ANSWER_SCHEMA)["answer"]
        correct = llm.json(JUDGE_SYSTEM, judge_prompt(q, gold, ans), JUDGE_SCHEMA)["correct"]
        return key, case, ans, bool(correct)

    for i, case in enumerate(cases):
        ctxs = contexts_for(case, methods, budgets, summarizer)
        q = case["messages"][case["question_index"]]["content"]
        gold = str(case["answers"][0]) if case["answers"] else ""
        if dry_run:
            for key, ctx in ctxs.items():
                est_in += count(ANSWER_SYSTEM) + count(answer_prompt(ctx, q)) + count(JUDGE_SYSTEM) + count(judge_prompt(q, gold, gold)) + 60
                est_out += 80
                rows.append({"case": case["id"], "type": case["type"], "method": key[0], "frac": key[1],
                             "context_tokens": count(ctx)})
        else:
            with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
                for key, c, ans, correct in ex.map(ask, [(k, v, case) for k, v in ctxs.items()]):
                    rows.append({"case": c["id"], "type": c["type"], "method": key[0], "frac": key[1],
                                 "answer": ans, "correct": correct, "context_tokens": count(ctxs[key])})
        if verbose:
            print(f"case {i + 1}/{len(cases)} done", flush=True)
    types = sorted({r["type"] for r in rows})
    table = {}
    keys = [("full", 1.0)] + [(m, f) for m in methods for f in budgets]
    for m, f in keys:
        sel = [r for r in rows if r["method"] == m and r["frac"] == f]
        if not sel:
            continue
        cell = {"context_tokens": statistics.mean(r["context_tokens"] for r in sel)}
        if not dry_run:
            cell["accuracy"] = sum(r["correct"] for r in sel) / len(sel)
            for t in types:
                st = [r for r in sel if r["type"] == t]
                cell[f"accuracy:{t}"] = sum(r["correct"] for r in st) / len(st) if st else float("nan")
        table[f"{m}@{f}"] = cell
    est = None
    if dry_run:
        est_in = int(est_in * TOKEN_SAFETY)
        est = {"input_tokens": est_in, "output_tokens": est_out,
               "cny": {k: round((est_in * p["in"] + est_out * p["out"]) / 1e6 * CNY_PER_USD, 2) for k, p in PRICE.items()}}
    return {"n_cases": len(cases), "methods": methods, "budgets": budgets, "types": types, "table": table,
            "rows": rows, "estimate": est, "dry_run": dry_run}


def to_markdown(s: dict, title: str) -> str:
    lines = [f"# {title}\n", f"{s['n_cases']} questions; methods {', '.join(s['methods'])}; budgets {s['budgets']}.\n"]
    if s["dry_run"]:
        e = s["estimate"]
        lines += [f"Dry run: ≈ {e['input_tokens']/1e6:.1f}M input tokens, {e['output_tokens']/1e3:.0f}k output tokens; "
                  f"DeepSeek Flash ≈ ¥{e['cny']['off_peak']} off-peak / ¥{e['cny']['peak']} peak.\n"]
    keys = [k for k in s["table"]]
    lines += ["| context | avg tokens | " + (" | ".join(["accuracy"] + s["types"]) if not s["dry_run"] else "") + " |",
              "|---|---|" + ("---|" * (1 + len(s["types"])) if not s["dry_run"] else "")]
    for k in keys:
        c = s["table"][k]
        row = f"| {k} | {c['context_tokens']:.0f} |"
        if not s["dry_run"]:
            row += f" {c['accuracy']:.2f} | " + " | ".join(f"{c.get(f'accuracy:{t}', float('nan')):.2f}" for t in s["types"]) + " |"
        lines.append(row)
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--longmemeval", type=Path)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--max-turns", type=int, default=120, help="LongMemEval history cap; a huge value keeps the full history")
    ap.add_argument("--locomo", type=Path)
    ap.add_argument("--per-conv", type=int, default=5)
    ap.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    ap.add_argument("--budgets", nargs="*", type=float, default=DEFAULT_BUDGETS)
    ap.add_argument("--llm", default=None, help="deepseek | anthropic; omit with --dry-run")
    ap.add_argument("--effort", default="off", help="thinking effort for answering and judging")
    ap.add_argument("--cache", type=Path, default=Path("evalkit/results/llm_cache.jsonl"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("evalkit/results"))
    args = ap.parse_args()
    if not args.dry_run and not args.llm:
        raise SystemExit("--llm is required unless --dry-run")
    llm = make_llm(args.llm, effort=args.effort, cache_path=args.cache) if args.llm else None
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = []
    if args.longmemeval:
        jobs.append(("longmemeval", longmemeval_cases(args.longmemeval, args.n, args.max_turns),
                     f"Model-answered QA on LongMemEval_s (histories ≤{args.max_turns} turns)"))
    if args.locomo:
        jobs.append(("locomo", locomo_cases(args.locomo, args.per_conv), "Model-answered QA on LoCoMo"))
    for name, cases, title in jobs:
        s = run(cases, args.methods, args.budgets, llm=llm, dry_run=args.dry_run, workers=args.workers)
        md = to_markdown(s, title)
        print(md)
        if not args.dry_run:
            (args.out / f"qa_{name}.md").write_text(md)
            (args.out / f"qa_{name}.json").write_text(json.dumps({k: v for k, v in s.items() if k != "rows"}, indent=1))
    if llm is not None:
        inner = getattr(llm, "inner", llm)
        print(f"[qa] {llm.name}: cache hits {getattr(llm, 'hits', 0)}, misses {getattr(llm, 'misses', 0)}, usage {getattr(inner, 'usage', {})}")


if __name__ == "__main__":
    main()
