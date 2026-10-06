"""Run the offline eval.

    python -m evalkit.run --cases 30 --out evalkit/results

Compares the graded dependency-graph compressor against baselines at equal
token budgets on synthetic conversations with known answers. Reports answer
retention per question category, constraint retention, stale-value exposure,
retrievability (answer kept OR a stub pointing at it kept), tokens used, and
the quality of the model-free edge heuristics against oracle edges.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from ctxgc.compress import build_graph, compress
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


def run(n_cases: int, out_dir: Path, methods: list[str]) -> dict:
    summarizer = ExtractiveSummarizer()
    rows = []            # one per (case, method, budget, question)
    tokens = defaultdict(list)
    edge_stats = []
    examples = {}
    for seed in range(n_cases):
        case = make_case(seed)
        edge_stats.append(edge_quality(case))
        graphs = {}
        full_tokens = None
        for name in methods:
            gopts, copts = METHODS[name]
            gkey = json.dumps(gopts, sort_keys=True)
            if gkey not in graphs:
                opts = dict(gopts)
                if opts.pop("oracle", False):
                    opts["oracle"] = case.oracle
                graphs[gkey] = build_graph(case.messages, **opts)
            g = graphs[gkey]
            if full_tokens is None:
                full_tokens = compress(g, 10**9, method="full", summarizer=summarizer).tokens
            budgets = [int(full_tokens * f) for f in BUDGET_FRACTIONS]
            for frac, budget in zip(BUDGET_FRACTIONS, budgets):
                seeds = range(RANDOM_SEEDS) if name == "random" else [0]
                for rs in seeds:
                    r = compress(g, budget, summarizer=summarizer, seed=rs, **copts)
                    tokens[(name, frac)].append(r.tokens / full_tokens)
                    for q in case.questions:
                        s = score_question(q, r)
                        s.update(case=seed, method=name, frac=frac)
                        rows.append(s)
                    if seed == 0 and rs == 0 and frac == 0.20 and name in ("graded", "graded+oracle", "truncate", "uniform"):
                        examples[name] = r.text
    summary = summarize(rows, tokens, edge_stats, methods)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    (out_dir / "summary.md").write_text(to_markdown(summary, n_cases))
    for name, text in examples.items():
        (out_dir / f"example_{name}_20pct.txt").write_text(text)
    return summary


def summarize(rows, tokens, edge_stats, methods) -> dict:
    cats = sorted({r["category"] for r in rows})
    out = {"budgets": BUDGET_FRACTIONS, "methods": methods, "categories": cats, "table": {}, "edges": {}}
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

    e = summary["edges"]
    lines += ["\n## Model-free edge heuristics vs oracle edges (mark phase)\n",
              f"- precision {e['precision']:.2f}, recall {e['recall']:.2f} "
              f"(≈{e['n_pred']:.0f} predicted vs {e['n_truth']:.0f} true edges per case, format-given tool/reads edges excluded)",
              f"- outdated value tombstoned: {e['supersedes_recall']:.2f} (attributed to the exact updating node: {e['supersedes_strict']:.2f}); "
              f"rejected proposal tombstoned: {e['rejects_recall']:.2f} (exact: {e['rejects_strict']:.2f}); "
              f"false tombstones per case: {e['false_tombstones']:.2f}",
              f"- user-sentence role accuracy (goal/constraint/user/chatter): {e['user_role_acc']:.2f}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=30)
    ap.add_argument("--out", type=Path, default=Path("evalkit/results"))
    ap.add_argument("--methods", nargs="*", default=list(METHODS))
    args = ap.parse_args()
    summary = run(args.cases, args.out, args.methods)
    print(to_markdown(summary, args.cases))


if __name__ == "__main__":
    main()
