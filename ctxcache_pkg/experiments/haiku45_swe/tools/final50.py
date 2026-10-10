"""Side-by-side replay (real tokens) of Gemini vs Haiku 4.5 (no system prompt) on the same tasks.

Usage: python3 -I final50.py <ctxcache_pkg> <scratch_dir> <tasks.json> <budgets> <out_prefix>
Reads Gemini trajectories from <scratch>/data/minitraj, Haiku transcripts via ~/.claude/projects,
grading from <scratch>/work45ns/<iid>/eval.json and Gemini's report.json. Tasks whose gold patch
fails its own FAIL_TO_PASS tests locally are reported as environment-broken and left out of the
solve rate.
"""
import glob
import json
import os
import sys

pkg, S, tasks_path, budgets, outp = sys.argv[1:6]
sys.path.insert(0, pkg)
import numpy as np  # noqa: E402

from ctxcache.cache import Replay  # noqa: E402
from ctxcache.model import load  # noqa: E402
from ctxcache.replay import CONFIGS, label  # noqa: E402

budgets = [float(b) for b in budgets.split(",")]
iids = [t["iid"] for t in json.load(open(tasks_path))]
lines = []
P = lines.append

# ---------- grading / cost ----------
rows, missing = [], []
for i in iids:
    ev_p, run_p = f"{S}/work45ns/{i}/eval.json", f"{S}/work45ns/{i}/run.json"
    jl = glob.glob(os.path.expanduser(f"~/.claude/projects/*work45ns-{i.replace('_', '-')}/*.jsonl"))
    if not (os.path.exists(ev_p) and os.path.exists(run_p) and jl):
        missing.append(i)
        continue
    ev, run = json.load(open(ev_p)), json.load(open(run_p))
    rep = f"{S}/data/minitraj/logs/{i}/report.json"
    rows.append(dict(iid=i, jsonl=jl[0], haiku=ev["resolved_env_adj"], strict=ev["resolved_strict"],
                     env_broken=bool(ev["gold"]["f2p_fail"]),
                     gemini=json.load(open(rep))[i]["resolved"] if os.path.exists(rep) else False,
                     cost=run.get("total_cost_usd") or 0, turns=run.get("num_turns"),
                     touched_tests=any(f.startswith(("tests/",)) or "/tests/" in f for f in ev["agent_files"])))
ok = [r for r in rows if not r["env_broken"]]
P(f"tasks: {len(rows)} graded, {len(missing)} missing {missing if missing else ''}")
P(f"environment-broken (gold patch fails locally): {[r['iid'] for r in rows if r['env_broken']]}")
P(f"solve rate on {len(ok)} usable tasks: Haiku 4.5 {sum(r['haiku'] for r in ok)}/{len(ok)} "
  f"({sum(r['haiku'] for r in ok)/len(ok):.0%}), Gemini 3.5 Flash {sum(r['gemini'] for r in ok)}/{len(ok)} "
  f"({sum(r['gemini'] for r in ok)/len(ok):.0%}); both {sum(r['haiku'] and r['gemini'] for r in ok)}, "
  f"only Haiku {sum(r['haiku'] and not r['gemini'] for r in ok)}, only Gemini {sum(r['gemini'] and not r['haiku'] for r in ok)}")
P(f"Haiku strict (no env adjustment): {sum(r['strict'] for r in ok)}/{len(ok)}; "
  f"runs that edited test files despite instructions: {sum(r['touched_tests'] for r in rows)}/{len(rows)}")
P(f"Haiku cost: total ${sum(r['cost'] for r in rows):.2f}, median ${np.median([r['cost'] for r in rows]):.2f}/task")

# ---------- replay ----------
groups = {"Gemini": {r["iid"]: load(f"{S}/data/minitraj/trajs/{r['iid']}.traj.json") for r in rows},
          "Haiku 4.5": {r["iid"]: load(r["jsonl"]) for r in rows}}
P("")
P(f"{'real context (tokens)':28} {'Gemini':>12} {'Haiku 4.5':>12}")
for name, fn in (("model calls / task (median)", lambda t: len(t.real_ctx)),
                 ("tool steps / task (median)", lambda t: len(t.steps)),
                 ("fixed part = first call", lambda t: t.real_ctx[0] / 1000),
                 ("peak context, k", lambda t: max(t.real_ctx) / 1000),
                 ("total sent per task, k", lambda t: sum(t.real_ctx) / 1000)):
    P(f"{name:28} " + " ".join(f"{np.median([fn(t) for t in g.values()]):12.1f}" for g in groups.values()))

res = []
for g, ts in groups.items():
    for b in budgets:
        for p, hi, lo in CONFIGS:
            for i, t in ts.items():
                r = Replay(p, b, t, hi=hi, lo=lo).run()
                r.pop("sizes")
                r.update(group=g, iid=i)
                res.append(r)
json.dump(res, open(outp + ".json", "w"))
P("")
P("medians (peak, sent) / means (compactions, misses, fresh misses, calls over budget); sent change vs raw")
hdr = f"{'peak':>6} {'sent':>6} {'Δsent':>6} {'cmp':>5} {'miss':>5} {'fresh':>5} {'over':>5}"
P(f"{'budget':>7} {'policy':12} | {'Gemini':^44} | {'Haiku 4.5':^44}")
P(f"{'':>7} {'':12} | {hdr} | {hdr}")
for b in budgets:
    for p, hi, lo in CONFIGS:
        cells = []
        for g in groups:
            rs = [r for r in res if r["group"] == g and r["budget"] == b and r["policy"] == p and r["hi"] == hi]
            raw = {r["iid"]: r["sent"] for r in res if r["group"] == g and r["budget"] == b and r["policy"] == "raw"}
            d = np.median([r["sent"] / raw[r["iid"]] - 1 for r in rs])
            cells.append(f"{np.median([r['peak'] for r in rs])/1000:6.1f} {np.median([r['sent'] for r in rs])/1000:6.0f} "
                         f"{d:+6.0%} {np.mean([r['compactions'] for r in rs]):5.1f} {np.mean([r['misses'] for r in rs]):5.2f} "
                         f"{np.mean([r['fresh_misses'] for r in rs]):5.2f} {np.mean([r['over_budget_steps'] for r in rs]):5.1f}")
        P(f"{int(b):>7} {label(dict(policy=p, hi=hi, lo=lo)):12} | " + " | ".join(cells))
    P("")
open(outp + ".txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
