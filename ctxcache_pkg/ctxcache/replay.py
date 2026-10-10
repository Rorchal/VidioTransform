"""Replay all trajectories under all policies and budgets.

Usage: python -m ctxcache.replay <traj_dir_or_glob> <out.json> [budgets=8000,16000,32000] [--chars]
Token costs come from the API usage recorded in the trajectories; --chars uses chars/4 instead.
"""
import glob
import json
import os
import sys

import numpy as np

from .cache import Replay
from .model import load, set_token_source

# (policy, hi, lo) — watermarks only matter for 'ours'
CONFIGS = [
    ("raw", 0, 0), ("tail10", 0, 0), ("compact", 0, 0), ("slim", 0, 0),
    ("ours", 0.9, 0.7), ("ours", 0.8, 0.5), ("ours", 0.6, 0.4),
]


def label(r):
    return r["policy"] if r["policy"] != "ours" else f"ours {int(r['hi']*100)}/{int(r['lo']*100)}"


def calibration(trajs):
    """For trajectories with recorded usage: how far the raw replay's context is from the real one."""
    errs, n_real = [], 0
    for t in trajs:
        if not t.real_ctx:
            continue
        n_real += 1
        sizes = Replay("raw", 1e12, t).run()["sizes"]
        # one size per model call, recorded after its last step = what the next call received
        last = [s for j, s in enumerate(t.steps)
                if not (s.call >= 0 and j + 1 < len(t.steps) and t.steps[j + 1].call == s.call)]
        errs += [abs(a - s.ctx_next) / s.ctx_next for a, s in zip(sizes, last) if s.ctx_next]
    if not n_real:
        return "token source: chars/4 (no usage recorded)"
    return (f"token source: API usage for {n_real}/{len(trajs)} trajectories; raw replay vs real context per call: "
            f"median error {np.median(errs):.1%}, p90 {np.percentile(errs, 90):.1%}")


def main():
    args = [a for a in sys.argv[1:] if a != "--chars"]
    set_token_source("--chars" not in sys.argv)
    src, out = args[0], args[1]
    budgets = [float(b) for b in (args[2] if len(args) > 2 else "8000,16000,32000").split(",")]
    files = (sorted(glob.glob(os.path.join(src, "*.json")) + glob.glob(os.path.join(src, "*.jsonl")))
             if os.path.isdir(src) else sorted(glob.glob(src)))
    trajs = [t for t in (load(f) for f in files) if len(t.steps) >= 5]
    print(f"trajectories: {len(trajs)}  (steps median {int(np.median([len(t.steps) for t in trajs]))})")
    print(calibration(trajs))
    results = []
    for b in budgets:
        for p, hi, lo in CONFIGS:
            for t in trajs:
                results.append(Replay(p, b, t, hi=hi, lo=lo).run())
    json.dump(results, open(out, "w"))

    def med(v):
        return float(np.median(v)) if len(v) else float("nan")

    def table(filter_fn, title):
        print(f"\n{title}")
        print(f"{'budget':>7} {'policy':12} {'peak k':>7} {'sent k':>7} {'compacts':>9} {'misses':>7} {'fresh':>6} {'over':>5}  n")
        for b in budgets:
            for p, hi, lo in CONFIGS:
                rs = [r for r in results if r["budget"] == b and r["policy"] == p and r["hi"] == hi and filter_fn(r)]
                if not rs:
                    continue
                print(f"{int(b):>7} {label(rs[0]):12} {med([r['peak'] for r in rs])/1000:7.1f} {med([r['sent'] for r in rs])/1000:7.0f} "
                      f"{np.mean([r['compactions'] for r in rs]):9.2f} {np.mean([r['misses'] for r in rs]):7.2f} "
                      f"{np.mean([r['fresh_misses'] for r in rs]):6.2f} {np.mean([r['over_budget_steps'] for r in rs]):5.1f}  {len(rs)}")
            print()

    table(lambda r: True, "ALL trajectories — medians (peak, sent) / means (compacts, misses, steps over budget)")
    table(lambda r: r["steps"] >= 50, "LONG trajectories (>= 50 steps) only")


if __name__ == "__main__":
    main()
