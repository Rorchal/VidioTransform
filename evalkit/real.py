"""Eval on real agent trajectories, without labels.

    python -m evalkit.real --swe-agent path/to/shard.parquet --n 60
    python -m evalkit.real --claude-code ~/.claude/projects/<proj>/<session>.jsonl

The synthetic eval (evalkit/run.py) knows the answers because it planted them.
A real trajectory has no labels, but it has a future: whatever the agent
referred to AFTER a cut point is, by definition, what its working set needed at
that point. So for each trajectory we cut the transcript at a few points,
compress the prefix at equal budgets, and measure

  carried    identifiers (file paths, symbols, constants, numbers with units,
             error codes) that appear in the agent's own later messages or
             commands AND were present in the prefix (discovered through tools;
             identifiers already in the opening request are excluded): fraction
             still present in the compressed prefix
  retrievable  carried, or a stub naming a prefix node that holds it survives
  reread     the subset of `carried` that are file paths in later commands: a
             path the agent will touch again whose content was evicted is a
             likely re-read (wasted steps)
  constraints  user sentences in the prefix that read as hard rules: fraction
             kept verbatim (Claude Code sessions only; SWE-agent has one turn)

Two converters: SWE-agent trajectories as published in nebius/SWE-agent-
trajectories (issue, then thought+command / observation pairs), and Claude Code
session logs (.jsonl with tool_use / tool_result content blocks).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from ctxgc.compress import build_graph, compress
from ctxgc.edges import identifiers as _identifiers
from ctxgc.llm import LLMSummarizer, graph_context, make_llm
from ctxgc.ingest import CONSTRAINT_RE, split_sentences
from ctxgc.model import L1, Role
from ctxgc.summarize import ExtractiveSummarizer
from ctxgc.tokens import count

from .metrics import node_order, present, stub_present

BUDGET_FRACTIONS = [0.10, 0.20, 0.35, 0.60]
# name -> compress() keyword arguments
METHODS: dict[str, dict] = {
    "full": {"method": "full"},
    "truncate": {"method": "truncate"},
    "random": {"method": "random"},
    "uniform": {"method": "uniform"},
    "gc_binary": {"method": "gc_binary"},
    "graded": {"method": "graded"},                                  # default allocator (level-table gain)
    "graded-agent": {"method": "graded", "gain": "idents"},          # gain = identifiers a finer version adds
    "graded-pinL0": {"method": "graded", "pin_fraction": None, "pin_promote": "always"},   # the old rule
}
# only with --llm: model-written L1/L2 for tool results of at least LLM_MIN_TOKENS tokens
LLM_METHODS: dict[str, dict] = {
    "graded+llmsum": {"method": "graded", "llmsum": True},
    "graded-agent+llmsum": {"method": "graded", "gain": "idents", "llmsum": True},
}
LLM_MIN_TOKENS = 80
CUT_FRACTIONS = [0.5, 0.75]           # cut after this share of the trajectory's tool steps
PATH_RE = re.compile(r"(?:[\w./-]+/)?[\w-]+\.(?:py|js|ts|sql|md|ya?ml|json|toml|txt|log|cfg|ini)\b")



def identifiers(text: str) -> set[str]:
    """ctxgc's identifier extractor with file paths reduced to their basename, so
    `open mailmerge/sendmail_client.py` and `sendmail_client.py` count as the
    same referent."""
    return {i.rsplit("/", 1)[-1] if PATH_RE.fullmatch(i) else i for i in _identifiers(text)}


# --------------------------------------------------------------------------- converters

SWE_TRAILER_RE = re.compile(r"\n\(Open file: [^\n]*\)\n\(Current directory: [^\n]*\)\nbash-\$\s*$")
FENCE_RE = re.compile(r"```(?:bash|sh)?\n(.*?)```", re.S)


def from_swe_agent(trajectory: list[dict]) -> list[dict]:
    """nebius/SWE-agent-trajectories row -> transcript. Roles there are
    system / user / ai; the first user message is the issue + instructions, later
    user messages are command observations."""
    out: list[dict] = []
    pending_call: str | None = None
    n_calls = 0
    first_user = True
    for m in trajectory:
        role, text = m.get("role"), (m.get("text") or "")
        if role == "system" or not text.strip():
            continue
        if role == "user":
            if first_user:
                out.append({"role": "user", "content": text.strip()})
                first_user = False
            elif pending_call is not None:
                out.append({"role": "tool", "tool_call_id": pending_call, "content": SWE_TRAILER_RE.sub("", text).strip()})
                pending_call = None
            else:                                   # observation without a command: keep as user text
                out.append({"role": "user", "content": SWE_TRAILER_RE.sub("", text).strip()})
        elif role in ("ai", "assistant"):
            fence = FENCE_RE.search(text)
            thought = FENCE_RE.sub("", text).strip()
            msg: dict = {"role": "assistant", "content": thought}
            if fence:
                cmd = fence.group(1).strip()
                n_calls += 1
                cid = f"swe_{n_calls}"
                name = (cmd.split() or ["cmd"])[0]
                msg["tool_calls"] = [{"id": cid, "name": name, "args": {"command": cmd}}]
                pending_call = cid
            out.append(msg)
    return out


SYSTEM_REMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.S)


def _block_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return ""


def from_claude_code_jsonl(path: str | Path) -> list[dict]:
    """Claude Code session log -> transcript. Keeps main-thread user text,
    assistant text and tool_use blocks, and tool_result blocks (as tool
    messages); drops thinking blocks, attachments and harness reminders."""
    out: list[dict] = []
    for line in Path(path).read_text().splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("type") not in ("user", "assistant") or e.get("isSidechain"):
            continue
        m = e.get("message") or {}
        content = m.get("content")
        if m.get("role") == "user":
            if isinstance(content, str):
                text = SYSTEM_REMINDER_RE.sub("", content).strip()
                if text:
                    out.append({"role": "user", "content": text})
                continue
            texts = []
            for b in content or []:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "tool_result":
                    out.append({"role": "tool", "tool_call_id": b.get("tool_use_id", ""),
                                "content": _block_text(b.get("content"))})
                elif b.get("type") == "text":
                    texts.append(b.get("text", ""))
            text = SYSTEM_REMINDER_RE.sub("", "\n".join(texts)).strip()
            if text:
                out.append({"role": "user", "content": text})
        elif m.get("role") == "assistant":
            texts, calls = [], []
            for b in content or []:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text":
                    texts.append(b.get("text", ""))
                elif b.get("type") == "tool_use":
                    calls.append({"id": b.get("id", ""), "name": b.get("name", "tool"), "args": b.get("input") or {}})
            text = "\n".join(texts).strip()
            if not text and not calls:
                continue
            # one assistant message streams as several events (thinking / text /
            # tool_use blocks) sharing the message id: merge them
            mid = m.get("id")
            if out and out[-1]["role"] == "assistant" and mid and out[-1].get("_id") == mid:
                prev = out[-1]
                prev["content"] = (prev["content"] + "\n" + text).strip()
                if calls:
                    prev.setdefault("tool_calls", []).extend(calls)
            else:
                msg: dict = {"role": "assistant", "content": text, "_id": mid}
                if calls:
                    msg["tool_calls"] = calls
                out.append(msg)
    for m in out:
        m.pop("_id", None)
    return out


# --------------------------------------------------------------------------- cuts and needs

def tool_steps(messages: list[dict]) -> list[int]:
    """Indices of tool-result messages: the natural cut points."""
    return [i for i, m in enumerate(messages) if m["role"] == "tool"]


def cut_points(messages: list[dict], fractions=CUT_FRACTIONS, min_prefix_steps: int = 6) -> list[int]:
    steps = tool_steps(messages)
    cuts = []
    for f in fractions:
        k = int(len(steps) * f)
        if k < min_prefix_steps or k >= len(steps):
            continue
        cuts.append(steps[k - 1] + 1)        # prefix = messages[:cut], ends right after a tool result
    return sorted(set(cuts))


def _args_text(args) -> str:
    """Tool arguments as plain text (no JSON escaping, so \\n does not glue on to
    the next word)."""
    if isinstance(args, dict):
        return "\n".join(_args_text(v) for v in args.values())
    if isinstance(args, list):
        return "\n".join(_args_text(v) for v in args)
    return str(args)


def agent_text(m: dict) -> str:
    """What the agent itself produced in a message: text plus command/args."""
    if m["role"] != "assistant":
        return ""
    parts = [m.get("content") or ""]
    for c in m.get("tool_calls") or []:
        parts.append(_args_text(c.get("args", {})))
    return "\n".join(parts)


def needs(g, messages: list[dict], cut: int) -> tuple[dict[str, set[str]], set[str]]:
    """Identifiers the agent uses after `cut` that the prefix graph holds
    (outside the opening request). Returns {ident: prefix node ids holding it}
    and the subset that are file paths named in later commands. Prefix
    identifiers are taken from the graph's node texts, i.e. exactly what the
    compressor renders, so `full` scores 1.0 by construction."""
    opening = identifiers(messages[0]["content"]) if messages and messages[0]["role"] == "user" else set()
    where: dict[str, set[str]] = defaultdict(set)
    for n in g.nodes.values():
        for ident in identifiers(n.text):
            where[ident].add(n.id)
    future_ids: set[str] = set()
    future_cmd_paths: set[str] = set()
    for m in messages[cut:]:
        if m["role"] != "assistant":
            continue
        future_ids |= identifiers(agent_text(m))
        for c in m.get("tool_calls") or []:
            future_cmd_paths |= set(PATH_RE.findall(_args_text(c.get("args", {}))))
    carried = {i: where[i] for i in future_ids if i in where and i not in opening}
    reread = {p.rsplit("/", 1)[-1] for p in future_cmd_paths} & set(carried)
    return carried, reread


def goal_needs(messages: list[dict], cut: int) -> set[str]:
    """Identifiers from the opening request that the agent still uses after the
    cut: forgetting the task shows up here (and nowhere in `carried`)."""
    if not messages or messages[0]["role"] != "user":
        return set()
    fut: set[str] = set()
    for m in messages[cut:]:
        if m["role"] == "assistant":
            fut |= identifiers(agent_text(m))
    return identifiers(messages[0]["content"]) & fut


def constraints_in(messages: list[dict], cut: int) -> list[str]:
    out = []
    for m in messages[1:cut]:           # the opening request is pinned anyway
        if m["role"] == "user":
            out += [s for s in split_sentences(m["content"]) if CONSTRAINT_RE.search(s)]
    return out


# --------------------------------------------------------------------------- scoring

def score_cut(messages: list[dict], cut: int, summarizer, methods: dict[str, dict] = METHODS,
              budgets=BUDGET_FRACTIONS, random_seeds: int = 2, llm=None, workers: int = 5) -> list[dict]:
    prefix = messages[:cut]
    g = build_graph(prefix)
    carried, reread = needs(g, messages, cut)
    goal_ids = goal_needs(messages, cut)
    cons = constraints_in(messages, cut)
    if not carried:
        return []
    full_tokens = compress(g, 10**9, method="full", summarizer=summarizer).tokens
    llm_sm = None
    if llm is not None and any(kw.get("llmsum") for kw in methods.values()):
        llm_sm = LLMSummarizer(llm, context=graph_context(g), roles=(Role.TOOL_RESULT,), min_tokens=LLM_MIN_TOKENS)
        llm_sm.prefetch(g.ordered(), workers=workers)
    rows = []
    for name, kw in methods.items():
        kw = dict(kw)
        sm = summarizer
        if kw.pop("llmsum", False):
            if llm_sm is None:
                raise SystemExit(f"method {name} needs --llm")
            sm = llm_sm
        for frac in budgets:
            seeds = range(random_seeds) if kw.get("method") == "random" else [0]
            for seed in seeds:
                r = compress(g, int(full_tokens * frac), summarizer=sm, seed=seed, **kw)
                found = identifiers(r.text)           # same extractor as the needs, so full scores 1.0
                kept = {i: i in found for i in carried}
                order = node_order(r)
                retr = {i: kept[i] or any(stub_present(r.text, nid, order) for nid in nids) for i, nids in carried.items()}
                rows.append({
                    "method": name, "frac": frac, "cut": cut, "n_carried": len(carried), "n_reread": len(reread),
                    "carried": sum(kept.values()) / len(carried),
                    "retrievable": sum(retr.values()) / len(carried),
                    "reread": (sum(kept[p] for p in reread) / len(reread)) if reread else None,
                    "goal": (sum(i in found for i in goal_ids) / len(goal_ids)) if goal_ids else None,
                    "constraints": (sum(present(r.text, c.rstrip(".")) for c in cons) / len(cons)) if cons else None,
                    "tokens": r.tokens / full_tokens,
                    "full_tokens": full_tokens,
                })
    return rows


def run(trajectories: list[tuple[str, list[dict]]], methods: dict[str, dict] = METHODS, budgets=BUDGET_FRACTIONS,
        llm=None, workers: int = 5) -> dict:
    summarizer = ExtractiveSummarizer()
    rows = []
    n_cuts = 0
    for name, messages in trajectories:
        for cut in cut_points(messages):
            cut_rows = score_cut(messages, cut, summarizer, methods, budgets, llm=llm, workers=workers)
            if cut_rows:
                n_cuts += 1
            for row in cut_rows:
                row["trajectory"] = name
            rows += cut_rows
    table = {}
    for method in methods:
        for frac in budgets:
            sel = [r for r in rows if r["method"] == method and r["frac"] == frac]
            if not sel:
                continue
            table[f"{method}@{frac}"] = {
                k: _mean(r[k] for r in sel if r[k] is not None)
                for k in ("carried", "retrievable", "reread", "goal", "constraints", "tokens")
            }
    return {"budgets": budgets, "methods": list(methods), "table": table, "rows": rows,
            "n_trajectories": len(trajectories), "n_cuts": n_cuts,
            "carried_per_cut": _mean(r["n_carried"] for r in rows if r["method"] == "full" and r["frac"] == budgets[0]),
            "prefix_tokens": _mean(r["full_tokens"] for r in rows if r["method"] == "full" and r["frac"] == budgets[0])}


def _mean(it) -> float:
    vals = [float(v) for v in it]
    return sum(vals) / len(vals) if vals else float("nan")


def to_markdown(summary: dict, title: str) -> str:
    fr = summary["budgets"]
    head = "| method | " + " | ".join(f"{int(f*100)}%" for f in fr) + " |\n|---|" + "---|" * len(fr)
    lines = [f"# {title}\n",
             f"{summary['n_trajectories']} trajectories, {summary['n_cuts']} cut points; prefix ≈ "
             f"{summary['prefix_tokens']:.0f} tokens, ≈ {summary['carried_per_cut']:.0f} carried identifiers per cut. "
             "Budget = fraction of the prefix's full token count.\n"]
    for key, desc in (("carried", "Carried identifiers still present (the agent uses them later)"),
                      ("retrievable", "Retrievable (present, or a stub for a node holding it survives)"),
                      ("reread", "File paths named in later commands still present (1 − likely re-reads)"),
                      ("goal", "Identifiers from the opening request the agent reuses later, still present"),
                      ("constraints", "Mid-conversation user constraints kept verbatim"),
                      ("tokens", "Tokens actually used (fraction of full)")):
        cells_any = any(summary["table"].get(f"{m}@{f}", {}).get(key) == summary["table"].get(f"{m}@{f}", {}).get(key)
                        for m in summary["methods"] for f in fr)      # nan != nan
        if not cells_any:
            continue
        lines += [f"\n## {desc}\n", head]
        for m in summary["methods"]:
            cells = [summary["table"].get(f"{m}@{f}", {}).get(key, float("nan")) for f in fr]
            lines.append(f"| {m} | " + " | ".join(f"{c:.2f}" for c in cells) + " |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- loading

def load_swe_agent(path: str | Path, n: int, min_msgs: int = 15, max_msgs: int = 80, seed: int = 0,
                   skip: int = 0) -> list[tuple[str, list[dict]]]:
    """n trajectories of moderate length, in a seeded shuffle; `skip` leaves out
    the first trajectories of that order (seed 0, skip 0 is the dev sample the
    allocator settings were chosen on; skip past it for a held-out sample)."""
    import random

    import pyarrow.parquet as pq   # optional dependency, only for the parquet loader

    table = pq.read_table(path, columns=["instance_id", "model_name", "target", "trajectory", "exit_status"])
    rows = [r for r in table.to_pylist() if min_msgs <= len(r["trajectory"]) <= max_msgs]
    random.Random(seed).shuffle(rows)
    out = []
    for r in rows[skip:skip + n]:
        msgs = from_swe_agent(r["trajectory"])
        out.append((f"{r['instance_id']}:{r['model_name']}:{'pass' if r['target'] else 'fail'}", msgs))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--swe-agent", type=Path, help="parquet shard of nebius/SWE-agent-trajectories")
    ap.add_argument("--n", type=int, default=60, help="trajectories to sample from the shard")
    ap.add_argument("--skip", type=int, default=10, help="skip the first trajectories of the shuffle (the dev sample)")
    ap.add_argument("--claude-code", type=Path, nargs="*", default=[], help="Claude Code session .jsonl files")
    ap.add_argument("--out", type=Path, default=Path("evalkit/results"))
    ap.add_argument("--methods", nargs="*", default=None,
                    help="default: all model-free methods, plus the *llmsum ones when --llm is given")
    ap.add_argument("--llm", default=None, help="model backend for L1/L2 summaries of big tool results: deepseek | anthropic")
    ap.add_argument("--summary-effort", default="off")
    ap.add_argument("--cache", type=Path, default=Path("evalkit/results/llm_cache.jsonl"))
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()
    all_methods = {**METHODS, **LLM_METHODS}
    names = args.methods or list(METHODS) + (list(LLM_METHODS) if args.llm else [])
    methods = {m: all_methods[m] for m in names}
    llm = make_llm(args.llm, effort=args.summary_effort, cache_path=args.cache) if args.llm else None
    args.out.mkdir(parents=True, exist_ok=True)
    if args.swe_agent:
        trajs = load_swe_agent(args.swe_agent, args.n, skip=args.skip)
        s = run(trajs, methods, llm=llm, workers=args.workers)
        md = to_markdown(s, "ctxgc on real SWE-agent trajectories (nebius/SWE-agent-trajectories)")
        (args.out / "real_swe_agent.md").write_text(md)
        (args.out / "real_swe_agent.json").write_text(json.dumps({k: v for k, v in s.items() if k != "rows"}, indent=1))
        print(md)
    if args.claude_code:
        trajs = [(p.stem, from_claude_code_jsonl(p)) for p in args.claude_code]
        s = run(trajs, methods, llm=llm, workers=args.workers)
        md = to_markdown(s, "ctxgc on Claude Code session logs")
        (args.out / "real_claude_code.md").write_text(md)
        print(md)
    if llm is not None:
        inner = getattr(llm, "inner", llm)
        print(f"[summaries] {llm.name}: cache hits {getattr(llm, 'hits', 0)}, misses {getattr(llm, 'misses', 0)}, "
              f"usage {getattr(inner, 'usage', {})}")


if __name__ == "__main__":
    main()
