"""Eval on non-agent conversations: long chat histories with a question at the end.

    python -m evalkit.chat --longmemeval longmemeval_s.json --n 60
    python -m evalkit.chat --locomo locomo10.json --per-conv 10

Two labelled datasets, no model calls:

  LongMemEval (xiaowu0162/longmemeval, the _s variant): a user/assistant chat
  history of ~50 sessions, a question asked afterwards, its answer, and which
  sessions (and turns, `has_answer`) hold the evidence. Six question types:
  single-session user / assistant / preference, multi-session, temporal
  reasoning, knowledge update.
  LoCoMo (snap-research/locomo): ten long two-person conversations with
  questions, answers and evidence turn ids. Speaker A is mapped to "user",
  speaker B to "assistant".

Per question the history (plus the question as the final user turn) is
compressed at equal budgets and scored:

  answer      the answer string survives in the compressed context
  evidence    labelled evidence turns rendered verbatim (L0), and at L0 or L1
  retrievable the answer survives, or a stub for an evidence node survives
  tokens      fraction of the full history actually used

Histories are long (LongMemEval ~120k tokens); to keep the allocator tractable
each LongMemEval instance is cut down to its evidence sessions plus random
distractor sessions up to --max-turns turns, in chronological order.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from ctxgc.compress import build_graph, compress
from ctxgc.edges import add_lexical_edges
from ctxgc.ingest import assign_roots
from ctxgc.model import L0, L1, Role
from ctxgc.summarize import ExtractiveSummarizer

from .metrics import present, stub_present

BUDGET_FRACTIONS = [0.10, 0.20, 0.35, 0.60]
METHODS: dict[str, dict] = {
    "full": {"method": "full"},
    "truncate": {"method": "truncate"},
    "random": {"method": "random"},
    "uniform": {"method": "uniform"},
    "gc_binary": {"method": "gc_binary"},
    "graded": {"method": "graded"},
    "graded+lex": {"method": "graded", "lexical": True},          # + word-overlap edges question -> turns
    "graded-agent+lex": {"method": "graded", "gain": "idents", "lexical": True},
}


# --------------------------------------------------------------------------- loaders

def longmemeval_cases(path: str | Path, n: int, max_turns: int = 120, seed: int = 0) -> list[dict]:
    """Stratified sample of n questions (equal per type). Each case: messages
    (history + question), answer strings, evidence message indices, type."""
    data = json.loads(Path(path).read_text())
    rng = random.Random(seed)
    by_type: dict[str, list] = defaultdict(list)
    for inst in data:
        by_type[inst["question_type"]].append(inst)
    types = sorted(by_type)
    per = max(1, n // len(types))
    cases = []
    for t in types:
        pool = by_type[t]
        rng.shuffle(pool)
        for inst in pool[:per]:
            cases.append(_longmemeval_case(inst, max_turns, rng))
    return cases


def _longmemeval_case(inst: dict, max_turns: int, rng: random.Random) -> dict:
    sessions = list(zip(inst["haystack_session_ids"], inst["haystack_dates"], inst["haystack_sessions"]))
    evidence_ids = set(inst["answer_session_ids"])
    keep = [i for i, (sid, _, _) in enumerate(sessions) if sid in evidence_ids]
    turns = sum(len(sessions[i][2]) for i in keep)
    others = [i for i in range(len(sessions)) if i not in keep]
    rng.shuffle(others)
    for i in others:
        if turns + len(sessions[i][2]) > max_turns:
            continue
        keep.append(i)
        turns += len(sessions[i][2])
    keep.sort()                                             # chronological
    messages: list[dict] = []
    evidence: list[int] = []
    for i in keep:
        sid, date, sess = sessions[i]
        first = True
        for m in sess:
            role = m["role"] if m["role"] in ("user", "assistant") else "user"
            content = (m.get("content") or "").strip()
            if not content:
                continue
            if first:
                content = f"[{date}] {content}"
                first = False
            if m.get("has_answer"):
                evidence.append(len(messages))
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": inst["question"].strip()})
    answers = _answer_variants(inst["answer"])
    return {"id": inst["question_id"], "type": inst["question_type"], "messages": messages,
            "answers": answers, "evidence": evidence, "question_index": len(messages) - 1}


def _answer_variants(answer) -> list[str]:
    """Short answers are matched as strings; alternatives separated by ' or ' /
    parentheses count too. Long explanatory answers are kept whole (they will
    rarely match, and the evidence metrics carry those categories)."""
    text = str(answer).strip()
    variants = [text]
    for sep in (" (or ", "(", " or "):
        if sep in text:
            head, _, tail = text.partition(sep)
            variants += [head.strip(" ."), tail.strip(" ().")]
    return [v for v in dict.fromkeys(variants) if len(v) >= 2]


def locomo_cases(path: str | Path, per_conv: int = 10, seed: int = 0, skip_categories=(5,)) -> list[dict]:
    """per_conv questions per conversation (category 5 = adversarial, no answer,
    skipped by default). Speaker A -> user, speaker B -> assistant; each session's
    first turn carries the session date."""
    data = json.loads(Path(path).read_text())
    rng = random.Random(seed)
    cases = []
    for conv in data:
        c = conv["conversation"]
        a, b = c.get("speaker_a"), c.get("speaker_b")
        keys = sorted((k for k in c if k.startswith("session_") and not k.endswith("date_time")),
                      key=lambda k: int(k.split("_")[1]))
        messages: list[dict] = []
        index_of: dict[str, int] = {}
        for k in keys:
            date = c.get(f"{k}_date_time", "")
            for j, turn in enumerate(c[k]):
                text = (turn.get("text") or "").strip()
                if not text:
                    continue
                if j == 0 and date:
                    text = f"[{date}] {text}"
                role = "user" if turn.get("speaker") == a else "assistant"
                index_of[turn["dia_id"]] = len(messages)
                messages.append({"role": role, "content": f"{turn.get('speaker')}: {text}"})
        qas = [q for q in conv["qa"] if q.get("category") not in skip_categories and q.get("answer") is not None]
        rng.shuffle(qas)
        for q in qas[:per_conv]:
            ev = [index_of[d] for d in q.get("evidence", []) if d in index_of]
            cases.append({"id": f"{conv['sample_id']}:{q['question'][:40]}", "type": f"cat{q['category']}",
                          "messages": messages + [{"role": "user", "content": str(q["question"]).strip()}],
                          "answers": _answer_variants(q["answer"]), "evidence": ev,
                          "question_index": len(messages)})
    return cases


# --------------------------------------------------------------------------- scoring

def chat_graph(messages: list[dict], question_index: int, lexical: bool = False):
    """A chat history has no task statement: the opening turn is just the first
    chat, so its sentences are plain user text (root 0.6), not pinned goals. The
    question appended at the end is the latest turn and the 1.0 root."""
    g = build_graph(messages)
    for n in g.nodes.values():
        if n.role == Role.GOAL and n.msg_index != question_index:
            n.role = Role.USER
    assign_roots(g)
    if lexical:
        add_lexical_edges(g, [n.id for n in g.nodes.values() if n.msg_index == question_index])
    return g


def score_case(case: dict, summarizer, methods: dict[str, dict] = METHODS, budgets=BUDGET_FRACTIONS,
               random_seeds: int = 2) -> list[dict]:
    g_plain = chat_graph(case["messages"], case["question_index"])
    g_lex = None
    ev_nodes = lambda g: [n.id for n in g.nodes.values() if n.msg_index in set(case["evidence"])]  # noqa: E731
    full_tokens = compress(g_plain, 10**9, method="full", summarizer=summarizer).tokens
    rows = []
    for name, kw in methods.items():
        kw = dict(kw)
        g = g_plain
        if kw.pop("lexical", False):
            if g_lex is None:
                g_lex = chat_graph(case["messages"], case["question_index"], lexical=True)
            g = g_lex
        evidence = ev_nodes(g)
        for frac in budgets:
            seeds = range(random_seeds) if kw.get("method") == "random" else [0]
            for seed in seeds:
                r = compress(g, int(full_tokens * frac), summarizer=summarizer, seed=seed, **kw)
                kept = any(present(r.text, a) for a in case["answers"])
                shown = [nid for nid in evidence if nid not in r.dropped]
                ev_l0 = sum(r.level.get(nid, 3) == L0 for nid in shown) / len(evidence) if evidence else None
                ev_l1 = sum(r.level.get(nid, 3) <= L1 for nid in shown) / len(evidence) if evidence else None
                rows.append({
                    "case": case["id"], "type": case["type"], "method": name, "frac": frac,
                    "answer": kept,
                    "evidence_l0": ev_l0, "evidence_l1": ev_l1,
                    "retrievable": kept or any(stub_present(r.text, nid) for nid in evidence),
                    "tokens": r.tokens / full_tokens, "full_tokens": full_tokens, "nodes": len(g.nodes),
                })
    return rows


def run(cases: list[dict], methods: dict[str, dict] = METHODS, budgets=BUDGET_FRACTIONS, verbose: bool = True) -> dict:
    summarizer = ExtractiveSummarizer()
    rows = []
    for i, case in enumerate(cases):
        rows += score_case(case, summarizer, methods, budgets)
        if verbose:
            print(f"case {i + 1}/{len(cases)} done", flush=True)
    types = sorted({r["type"] for r in rows})
    table = {}
    for method in methods:
        for frac in budgets:
            sel = [r for r in rows if r["method"] == method and r["frac"] == frac]
            if not sel:
                continue
            cell = {k: _mean(r[k] for r in sel if r[k] is not None)
                    for k in ("answer", "evidence_l0", "evidence_l1", "retrievable", "tokens")}
            for t in types:
                cell[f"answer:{t}"] = _mean(r["answer"] for r in sel if r["type"] == t)
                cell[f"evidence_l1:{t}"] = _mean(r["evidence_l1"] for r in sel if r["type"] == t and r["evidence_l1"] is not None)
            table[f"{method}@{frac}"] = cell
    base = [r for r in rows if r["method"] == "full" and r["frac"] == budgets[0]]
    return {"budgets": budgets, "methods": list(methods), "types": types, "table": table, "rows": rows,
            "n_cases": len(cases), "tokens_per_case": _mean(r["full_tokens"] for r in base),
            "nodes_per_case": _mean(r["nodes"] for r in base)}


def _mean(it) -> float:
    vals = [float(v) for v in it]
    return sum(vals) / len(vals) if vals else float("nan")


def to_markdown(summary: dict, title: str) -> str:
    fr = summary["budgets"]
    head = "| method | " + " | ".join(f"{int(f*100)}%" for f in fr) + " |\n|---|" + "---|" * len(fr)
    lines = [f"# {title}\n",
             f"{summary['n_cases']} questions; history ≈ {summary['tokens_per_case']:.0f} tokens, "
             f"≈ {summary['nodes_per_case']:.0f} nodes per question (the question itself is the final user turn). "
             "Budget = fraction of the full history's token count.\n"]
    for key, desc in (("answer", "Answer string present in the compressed context"),
                      ("evidence_l1", "Labelled evidence turns kept verbatim or as a paragraph summary (L0/L1)"),
                      ("evidence_l0", "Labelled evidence turns kept verbatim (L0)"),
                      ("retrievable", "Retrievable (answer present, or a stub for an evidence turn survives)"),
                      ("tokens", "Tokens actually used (fraction of full)")):
        lines += [f"\n## {desc}\n", head]
        for m in summary["methods"]:
            cells = [summary["table"].get(f"{m}@{f}", {}).get(key, float("nan")) for f in fr]
            lines.append(f"| {m} | " + " | ".join(f"{c:.2f}" for c in cells) + " |")
    for f in (0.2,):
        if f not in fr:
            continue
        for key, desc in (("answer", "Answer present"), ("evidence_l1", "Evidence at L0/L1")):
            lines += [f"\n## {desc} by question type @ {int(f*100)}% budget\n",
                      "| method | " + " | ".join(summary["types"]) + " |", "|---|" + "---|" * len(summary["types"])]
            for m in summary["methods"]:
                cell = summary["table"].get(f"{m}@{f}", {})
                lines.append(f"| {m} | " + " | ".join(f"{cell.get(f'{key}:{t}', float('nan')):.2f}" for t in summary["types"]) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--longmemeval", type=Path, help="longmemeval_s.json")
    ap.add_argument("--n", type=int, default=60, help="LongMemEval questions (stratified over the 6 types)")
    ap.add_argument("--max-turns", type=int, default=120, help="LongMemEval: evidence sessions + distractors up to this many turns")
    ap.add_argument("--locomo", type=Path, help="locomo10.json")
    ap.add_argument("--per-conv", type=int, default=10, help="LoCoMo questions per conversation")
    ap.add_argument("--out", type=Path, default=Path("evalkit/results"))
    ap.add_argument("--methods", nargs="*", default=list(METHODS))
    args = ap.parse_args()
    methods = {m: METHODS[m] for m in args.methods}
    args.out.mkdir(parents=True, exist_ok=True)
    if args.longmemeval:
        s = run(longmemeval_cases(args.longmemeval, args.n, args.max_turns), methods)
        md = to_markdown(s, f"ctxgc on LongMemEval_s (histories cut to ≤{args.max_turns} turns: evidence sessions + distractors)")
        (args.out / "chat_longmemeval.md").write_text(md)
        (args.out / "chat_longmemeval.json").write_text(json.dumps({k: v for k, v in s.items() if k != "rows"}, indent=1))
        print(md)
    if args.locomo:
        s = run(locomo_cases(args.locomo, args.per_conv), methods)
        md = to_markdown(s, "ctxgc on LoCoMo (two-person conversations; speaker A as user, B as assistant)")
        (args.out / "chat_locomo.md").write_text(md)
        (args.out / "chat_locomo.json").write_text(json.dumps({k: v for k, v in s.items() if k != "rows"}, indent=1))
        print(md)


if __name__ == "__main__":
    main()
