"""Compute compaction metrics on parsed trajectories.

Usage: python -I agent_metrics.py <agent_steps.jsonl> <out_prefix>
Tokens are approximated as chars/4 throughout (no tokenizer available offline).
"""
import json
import sys
from collections import Counter, defaultdict

import numpy as np

CH = 4.0  # chars per token approximation


def tok(c):
    return c / CH


def policies_at_end(steps):
    """Return kept-observation-token totals under policies, evaluated at every step t.

    Returns dict policy -> array of length n (kept obs tokens after step t), plus dirty array.
    Policies:
      raw     : keep all observations
      safe    : drop obs if identical command re-run later, or any of its path keys edited later
      path    : drop obs if any of its path keys referenced (any kind) later
      tail10  : keep only last 10 observations
    """
    n = len(steps)
    o = np.array([tok(s["o_chars"]) for s in steps])
    a = np.array([tok(s["a_chars"]) for s in steps])
    raw = np.cumsum(o)
    dirty = np.cumsum(a)
    safe = np.zeros(n)
    path = np.zeros(n)
    tail = np.zeros(n)
    # online: at step t, which earlier obs are superseded by steps <= t
    for t in range(n):
        last_cmd = {}
        last_key_any = {}
        last_key_edit = {}
        for j in range(t + 1):
            s = steps[j]
            last_cmd[s["cmd"]] = j
            for k in s["keys"]:
                last_key_any[k] = j
                if s["kind"] == "edit":
                    last_key_edit[k] = j
        ks = 0.0
        kp = 0.0
        for j in range(t + 1):
            s = steps[j]
            sup_safe = last_cmd.get(s["cmd"], j) > j or any(
                last_key_edit.get(k, -1) > j for k in s["keys"]
            )
            sup_path = sup_safe or any(last_key_any.get(k, j) > j for k in s["keys"])
            if not sup_safe:
                ks += o[j]
            if not sup_path:
                kp += o[j]
        safe[t] = ks
        path[t] = kp
        tail[t] = o[max(0, t - 9) : t + 1].sum()
    return {"raw": raw, "safe": safe, "path": path, "tail10": tail, "dirty": dirty}


def main():
    inp, outp = sys.argv[1:3]
    trajs = [json.loads(l) for l in open(inp)]
    print("trajectories:", len(trajs))

    # 1. clean vs dirty composition
    clean = dirty = task = 0.0
    max_single_share = []
    per_traj_clean_share = []
    for t in trajs:
        o = sum(s["o_chars"] for s in t["steps"])
        a = sum(s["a_chars"] for s in t["steps"])
        tk = t["task_chars"] + t["sys_chars"]
        clean += o
        dirty += a
        task += tk
        tot = o + a + tk
        if tot > 0:
            per_traj_clean_share.append(o / tot)
            max_single_share.append(max(s["o_chars"] for s in t["steps"]) / tot)
    tot = clean + dirty + task
    print(f"\n[1] token composition (all trajs): tool outputs {clean/tot:.1%}, assistant msgs {dirty/tot:.1%}, task+system prompt {task/tot:.1%}")
    print(f"    per-traj clean share median {np.median(per_traj_clean_share):.1%}, p10 {np.percentile(per_traj_clean_share,10):.1%}, p90 {np.percentile(per_traj_clean_share,90):.1%}")
    print(f"[2] largest single observation / whole context: median {np.median(max_single_share):.1%}, p90 {np.percentile(max_single_share,90):.1%}, max {max(max_single_share):.1%}")

    # 3. reachable-but-irrelevant: obs tokens whose path keys are all outside final patch files
    irr = rel = nokey = 0.0
    irr_n = []
    for t in trajs:
        pf = set(t["patch_files"])
        if not pf:
            continue
        ti = tr = tn = 0.0
        for s in t["steps"]:
            if s["kind"] in ("submit", "gitdiff"):
                continue
            ks = [k for k in s["keys"] if "." in k.split("/")[-1]]  # file-like keys only
            if not ks:
                tn += s["o_chars"]
            elif any(k.lstrip("./") in pf or any(p.endswith(k.lstrip("./")) for p in pf) for k in ks):
                tr += s["o_chars"]
            else:
                ti += s["o_chars"]
        irr += ti
        rel += tr
        nokey += tn
        if ti + tr > 0:
            irr_n.append(ti / (ti + tr))
    print(f"[3] observations keyed by files: about files in final patch {rel/(irr+rel+nokey):.1%}, about files NOT in final patch {irr/(irr+rel+nokey):.1%}, unkeyed (runs/searches w/o path) {nokey/(irr+rel+nokey):.1%}")
    print(f"    per-traj share of file-keyed obs tokens that are about non-patch files: median {np.median(irr_n):.1%}")

    # 4/5. compaction policies: end-of-trajectory kept share and growth curves
    end_share = defaultdict(list)
    curves = defaultdict(list)
    MINSTEPS = 50
    slope_ratio = defaultdict(list)
    for t in trajs:
        st = t["steps"]
        if len(st) < 5:
            continue
        P = policies_at_end(st)
        rawT = P["raw"][-1]
        if rawT <= 0:
            continue
        for p in ("safe", "path", "tail10"):
            end_share[p].append(P[p][-1] / rawT)
        if len(st) >= MINSTEPS:
            for p in ("raw", "safe", "path", "tail10", "dirty"):
                curves[p].append(P[p][:MINSTEPS])
            # slope in last 20 steps vs first 20 steps
            for p in ("raw", "safe", "path", "dirty"):
                s1 = (P[p][19] - P[p][0]) / 19
                s2 = (P[p][MINSTEPS - 1] - P[p][MINSTEPS - 20]) / 19
                if s1 > 0:
                    slope_ratio[p].append(s2 / s1)
    print(f"\n[4] observation tokens still kept at END of trajectory (share of raw):")
    for p in ("safe", "path", "tail10"):
        v = end_share[p]
        print(f"    {p:7s}: median {np.median(v):.1%}  p10 {np.percentile(v,10):.1%}  p90 {np.percentile(v,90):.1%}")
    print(f"[5] growth over first {MINSTEPS} steps (trajs with >= {MINSTEPS} steps: {len(curves['raw'])}); mean kept tokens at step 10/20/30/40/50:")
    for p in ("raw", "safe", "path", "tail10", "dirty"):
        m = np.mean(curves[p], axis=0)
        print(f"    {p:7s}: " + "  ".join(f"{m[i]:8.0f}" for i in (9, 19, 29, 39, 49)) + f"   slope(last20)/slope(first20) median {np.median(slope_ratio[p]) if slope_ratio[p] else float('nan'):.2f}")
    np.save(outp + "_curves.npy", {p: np.array(v) for p, v in curves.items()}, allow_pickle=True)

    # 6. re-read rate
    reads = rereads = reread_after_edit = 0
    for t in trajs:
        seen = set()
        edited_since = set()
        for s in t["steps"]:
            if s["kind"] == "edit":
                edited_since.update(s["keys"])
            if s["kind"] == "read":
                for k in s["keys"]:
                    reads += 1
                    if k in seen:
                        rereads += 1
                        if k in edited_since:
                            reread_after_edit += 1
                    seen.add(k)
                    edited_since.discard(k)
    print(f"\n[6] file reads: {reads}; of which re-reads of an already-read file {rereads/reads:.1%}; re-reads that follow an edit of that file {reread_after_edit/reads:.1%}")

    # 7. use distance: edit of X <- last read of X
    dist = []
    never = 0
    edits = 0
    for t in trajs:
        last_read = {}
        for s in t["steps"]:
            if s["kind"] in ("read", "search"):
                for k in s["keys"]:
                    last_read[k] = s["i"]
            if s["kind"] == "edit":
                for k in s["keys"]:
                    if k.startswith("test_") or "repro" in k or k.startswith("/tmp"):
                        continue  # scratch files the agent wrote itself
                    edits += 1
                    # match by suffix (relative path spellings differ)
                    cands = [i for kk, i in last_read.items() if kk.lstrip("./") == k.lstrip("./") or kk.endswith("/" + k) or k.endswith("/" + kk)]
                    if cands:
                        dist.append(s["i"] - max(cands))
                    else:
                        never += 1
    d = np.array(dist)
    print(f"[7] edits of repo files: {edits}; file never read/searched before edit: {never/edits:.1%}")
    print(f"    steps since last read of the edited file: median {np.median(d):.0f}, <=5 steps {np.mean(d<=5):.1%}, <=10 {np.mean(d<=10):.1%}, >20 {np.mean(d>20):.1%}, max {d.max()}")

    # 8. resolved vs not
    for flag in (True, False):
        sub = [t for t in trajs if t["resolved"] == flag]
        steps = [len(t["steps"]) for t in sub]
        raw = [sum(s["o_chars"] + s["a_chars"] for s in t["steps"]) / CH for t in sub]
        print(f"[8] resolved={flag}: n={len(sub)}, median steps {np.median(steps):.0f}, median total tokens {np.median(raw):.0f}")


if __name__ == "__main__":
    main()
