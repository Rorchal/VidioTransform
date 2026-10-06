"""Run the offline eval.

    python -m evalkit.run --cases 30 --out evalkit/results
    python -m evalkit.run --cases 30 --llm deepseek          # + model-built edges / summaries

Compares the graded dependency-graph compressor against baselines at equal
token budgets on synthetic conversations with known answers. Reports answer
retention per question category, constraint retention, stale-value exposure,
retrievability (answer kept OR a stub pointing at it kept), tokens used, and
the quality of the inferred edges (model-free heuristics and, with --llm, the
model's mark phase) against oracle edges.

With --llm, four more methods run: graded+llm (model edges replace the
heuristics), graded+llm+heur (both), graded+llmsum (heuristic edges, model
L1/L2 summaries), graded+llm+llmsum (model edges and summaries). Model replies
are cached on disk (--cache) so a re-run with the same cache is free and
reproduces the numbers without credentials.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from ctxgc.compress import build_graph, compress
from ctxgc.llm import LLMSummarizer, graph_context, make_llm
from ctxgc.summarize import ExtractiveSummarizer

from .metrics import edge_quality, score_question
from .synth import make_case

BUDGET_FRACTIONS = [0.10, 0.20, 0.35, 0.60]
RANDOM_SEEDS = 3

# name -> (graph options, compress options)
METHODS: dict[str, tuple[dict, dict]] = {
    "full":            ({}, {"method": "full"}),
    "truncate":        ({}, {"method": "truncate"}),
    "random":          ({}, {"method": "random"}),
    "uniform":         ({}, {"method": "uniform"}),
    "gc_binary":       ({}, {"method": "gc_binary"}),
    "graded":          ({}, {"method": "graded"}),
    "graded+oracle":   ({"oracle": True}, {"method": "graded"}),
    "graded-facts":    ({"split_facts": False}, {"method": "graded"}),
    "graded-tomb":     ({"tombstones": False}, {"method": "graded"}),
    "graded-frame":    ({}, {"method": "graded", "frame_pop": False}),
    "graded-cascade":  ({}, {"method": "graded", "cascade": False}),
    "graded-symbolic": ({"symbolic": False}, {"method": "graded"}),
}

# only with --llm; "llm" / "llmsum" in the graph options are resolved at run time
LLM_METHODS: dict[str, tuple[dict, dict]] = {
    "graded+llm":        ({"llm": True}, {"method": "graded"}),
    "graded+llm+heur":   ({"llm": True, "llm_heuristics": True}, {"method": "graded"}),
    "graded+llmsum":     ({"llmsum": True}, {"method": "graded"}),
    "graded+llm+llmsum": ({"llm": True, "llmsum": True}, {"method": "graded"}),
}


def run(n_cases: int, out_dir: Path, methods: list[str], llm=None, edge_llm=None, summary_llm=None,
        workers: int = 8, max_edges: int = 4) -> dict:
    """llm: legacy single backend for both stages; edge_llm / summary_llm override
    per stage (the summaries are usually run without thinking)."""
    edge_llm = edge_llm or llm
    summary_llm = summary_llm or llm
    summarizer = ExtractiveSummarizer()
    rows = []            # one per (case, method, budget, question)
    tokens = defaultdict(list)
    edge_stats = []
    llm_edge_stats = []
    examples = {}
    all_methods = {**METHODS, **LLM_METHODS}
    for seed in range(n_cases):
        case = make_case(seed)
        edge_stats.append(edge_quality(case))
        graphs = {}
        full_tokens = None
        for name in methods:
            gopts, copts = all_methods[name]
            gkey = json.dumps(gopts, sort_keys=True)
            if gkey not in graphs:
                opts = dict(gopts)
                if opts.pop("oracle", False):
                    opts["oracle"] = case.oracle
                use_llm_sum = opts.pop("llmsum", False)
                if opts.pop("llm", False):
                    if edge_llm is None:
                        raise SystemExit(f"method {name} needs --llm")
                    opts["llm"] = edge_llm
                    opts["llm_workers"] = workers
                    opts["llm_max_edges"] = max_edges
                g = build_graph(case.messages, **opts)
                if "llm" in opts and not opts.get("llm_heuristics"):
                    llm_edge_stats.append(edge_quality(case, g))
                sm = summarizer
                if use_llm_sum:
                    if summary_llm is None:
                        raise SystemExit(f"method {name} needs --llm")
                    sm = LLMSummarizer(summary_llm, context=graph_context(g))
                    sm.prefetch(g.ordered(), workers=workers)
                graphs[gkey] = (g, sm)
            g, sm = graphs[gkey]
            if full_tokens is None:
                full_tokens = compress(g, 10**9, method="full", summarizer=summarizer).tokens
            budgets = [int(full_tokens * f) for f in BUDGET_FRACTIONS]
            for frac, budget in zip(BUDGET_FRACTIONS, budgets):
                seeds = range(RANDOM_SEEDS) if name == "random" else [0]
                for rs in seeds:
                    r = compress(g, budget, summarizer=sm, seed=rs, **copts)
                    tokens[(name, frac)].append(r.tokens / full_tokens)
                    for q in case.questions:
                        s = score_question(q, r)
                        s.update(case=seed, method=name, frac=frac)
                        rows.append(s)
                    if seed == 0 and rs == 0 and frac == 0.20 and name in (
                            "graded", "graded+oracle", "truncate", "uniform", "graded+llm", "graded+llm+llmsum"):
                        examples[name] = r.text
        print(f"case {seed + 1}/{n_cases} done", flush=True)
    summary = summarize(rows, tokens, edge_stats, methods, llm_edge_stats)
    if edge_llm is not None:
        summary["llm"] = {"edges": getattr(edge_llm, "name", "?"), "summaries": getattr(summary_llm, "name", "?"),
                          "max_edges": max_edges}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    (out_dir / "summary.md").write_text(to_markdown(summary, n_cases))
    for name, text in examples.items():
        (out_dir / f"example_{name}_20pct.txt").write_text(text)
    return summary


def summarize(rows, tokens, edge_stats, methods, llm_edge_stats=()) -> dict:
    cats = sorted({r["category"] for r in rows})
    out = {"budgets": BUDGET_FRACTIONS, "methods": methods, "categories": cats, "table": {}, "edges": {}}
    if llm_edge_stats:
        out["llm_edges"] = {k: statistics.mean(e[k] for e in llm_edge_stats) for k in llm_edge_stats[0]}
    for name in methods:
        for frac in BUDGET_FRACTIONS:
            sel = [r for r in rows if r["method"] == name and r["frac"] == frac]
            if not sel:
                continue
            cell = {
                "kept": _mean(r["kept"] for r in sel),
                "retrievable": _mean(r["retrievable"] for r in sel),
                "tokens": statistics.mean(tokens[(name, frac)]),
                "stale_exposed": _mean(r["stale_exposed"] for r in sel if "stale_exposed" in r),
            }
            for c in cats:
                cell[f"kept:{c}"] = _mean(r["kept"] for r in sel if r["category"] == c)
            out["table"][f"{name}@{frac}"] = cell
    for k in edge_stats[0]:
        out["edges"][k] = statistics.mean(e[k] for e in edge_stats)
    return out


def _mean(it) -> float:
    vals = [float(v) for v in it]
    return sum(vals) / len(vals) if vals else float("nan")


def to_markdown(summary: dict, n_cases: int) -> str:
    fr = summary["budgets"]
    lines = [f"# ctxgc offline eval — {n_cases} synthetic cases, {len(summary['categories'])} question categories\n",
             "Budget = fraction of the full-context token count. Cells = fraction of questions whose answer "
             "string survives in the compressed context (mean over cases and questions).\n",
             "## Answer retention (all questions)\n",
             "| method | " + " | ".join(f"{int(f*100)}%" for f in fr) + " |",
             "|---|" + "---|" * len(fr)]
    for m in summary["methods"]:
        cells = [summary["table"].get(f"{m}@{f}", {}).get("kept", float("nan")) for f in fr]
        lines.append(f"| {m} | " + " | ".join(f"{c:.2f}" for c in cells) + " |")

    lines += ["\n## Retrievable (answer kept OR a stub pointing at its node kept)\n",
              "| method | " + " | ".join(f"{int(f*100)}%" for f in fr) + " |", "|---|" + "---|" * len(fr)]
    for m in summary["methods"]:
        cells = [summary["table"].get(f"{m}@{f}", {}).get("retrievable", float("nan")) for f in fr]
        lines.append(f"| {m} | " + " | ".join(f"{c:.2f}" for c in cells) + " |")

    lines += ["\n## Stale value still presented as live (lower is better)\n",
              "| method | " + " | ".join(f"{int(f*100)}%" for f in fr) + " |", "|---|" + "---|" * len(fr)]
    for m in summary["methods"]:
        cells = [summary["table"].get(f"{m}@{f}", {}).get("stale_exposed", float("nan")) for f in fr]
        lines.append(f"| {m} | " + " | ".join(f"{c:.2f}" for c in cells) + " |")

    lines += ["\n## Tokens actually used (fraction of full; budget in header)\n",
              "| method | " + " | ".join(f"{int(f*100)}%" for f in fr) + " |", "|---|" + "---|" * len(fr)]
    for m in summary["methods"]:
        cells = [summary["table"].get(f"{m}@{f}", {}).get("tokens", float("nan")) for f in fr]
        lines.append(f"| {m} | " + " | ".join(f"{c:.2f}" for c in cells) + " |")

    for f in fr:
        lines += [f"\n## Retention by question category @ {int(f*100)}% budget\n",
                  "| method | " + " | ".join(summary["categories"]) + " |",
                  "|---|" + "---|" * len(summary["categories"])]
        for m in summary["methods"]:
            cell = summary["table"].get(f"{m}@{f}", {})
            lines.append(f"| {m} | " + " | ".join(f"{cell.get(f'kept:{c}', float('nan')):.2f}" for c in summary["categories"]) + " |")

    lines += ["\n## Model-free edge heuristics vs oracle edges (mark phase)\n"] + _edge_lines(summary["edges"])
    if summary.get("llm_edges"):
        llm = summary.get("llm", {})
        lines += [f"\n## Model-built edges ({llm.get('edges', '?')}, at most {llm.get('max_edges', 4)} edges per chunk) "
                  "vs oracle edges (mark phase)\n"] + _edge_lines(summary["llm_edges"])
        lines.append(f"- summaries for the *llmsum methods: {llm.get('summaries', '?')}")
    return "\n".join(lines) + "\n"


def _edge_lines(e: dict) -> list[str]:
    return [f"- precision {e['precision']:.2f}, recall {e['recall']:.2f} "
            f"(≈{e['n_pred']:.0f} predicted vs {e['n_truth']:.0f} true edges per case, format-given tool/reads edges excluded)",
            f"- outdated value tombstoned: {e['supersedes_recall']:.2f} (attributed to the exact updating node: {e['supersedes_strict']:.2f}); "
            f"rejected proposal tombstoned: {e['rejects_recall']:.2f} (exact: {e['rejects_strict']:.2f}); "
            f"false tombstones per case: {e['false_tombstones']:.2f}",
            f"- user-sentence role accuracy (goal/constraint/user/chatter): {e['user_role_acc']:.2f}"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=30)
    ap.add_argument("--out", type=Path, default=Path("evalkit/results"))
    ap.add_argument("--methods", nargs="*", default=None,
                    help="default: all baselines/ablations, plus the LLM methods when --llm is given")
    ap.add_argument("--llm", default=None, help="model backend: deepseek | deepseek:<model> | anthropic[:<model>]")
    ap.add_argument("--edge-effort", default="low", help="thinking effort for edge inference (off/low/high)")
    ap.add_argument("--summary-effort", default="off", help="thinking effort for L1/L2 summaries (off/low/high)")
    ap.add_argument("--cache", type=Path, default=Path("evalkit/results/llm_cache.jsonl"), help="model reply cache file")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-edges", type=int, default=4, help="edges the model may attach per chunk")
    args = ap.parse_args()
    methods = args.methods or list(METHODS) + (list(LLM_METHODS) if args.llm else [])
    edge_llm = summary_llm = None
    if args.llm:
        edge_llm = make_llm(args.llm, effort=args.edge_effort, cache_path=args.cache)
        summary_llm = make_llm(args.llm, effort=args.summary_effort, cache_path=args.cache)
    summary = run(args.cases, args.out, methods, edge_llm=edge_llm, summary_llm=summary_llm, workers=args.workers,
                  max_edges=args.max_edges)
    print(to_markdown(summary, args.cases))
    for label, llm in (("edges", edge_llm), ("summaries", summary_llm)):
        if llm is not None:
            inner = getattr(llm, "inner", llm)
            print(f"[{label}] {llm.name}: cache hits {getattr(llm, 'hits', 0)}, misses {getattr(llm, 'misses', 0)}, "
                  f"usage {getattr(inner, 'usage', {})}")


if __name__ == "__main__":
    main()
