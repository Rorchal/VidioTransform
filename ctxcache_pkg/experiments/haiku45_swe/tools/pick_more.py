"""Randomly add django/sympy SWE-bench Verified tasks (resolved or not by Gemini) to an existing task list.

Usage: python3 -I pick_more.py <minitraj_dir> <swebv.parquet> <existing_tasks.json> <n_new> <seed> <out_prefix>
Writes <out_prefix>.json (all tasks) and <out_prefix>_new.txt ("iid base_sha python" per new task).
Python per task: django < 2.2 -> 3.7, django >= 5.0 -> 3.11, everything else -> 3.9.
"""
import json
import os
import random
import sys
from collections import Counter

import pyarrow.parquet as pq

mt, pqp, existing, n_new, seed, outp = sys.argv[1:7]
rows = {r["instance_id"]: r for r in pq.read_table(pqp).to_pylist()}
have = json.load(open(existing))
have_ids = {t["iid"] for t in have}
pool = []
for f in sorted(os.listdir(os.path.join(mt, "trajs"))):
    iid = f.split(".traj")[0]
    if iid.startswith(("django__", "sympy__")) and iid not in have_ids and iid in rows:
        pool.append(iid)
random.seed(int(seed))
new = random.sample(pool, int(n_new))


def py_for(r):
    v = tuple(int(x) for x in r["version"].split("."))
    if r["repo"] == "django/django":
        return "3.7" if v < (2, 2) else "3.11" if v >= (5, 0) else "3.9"
    return "3.9"


out = list(have)
with open(outp + "_new.txt", "w") as fo:
    for iid in new:
        r = rows[iid]
        rep = os.path.join(mt, "logs", iid, "report.json")
        res = json.load(open(rep))[iid]["resolved"] if os.path.exists(rep) else False
        d = json.load(open(os.path.join(mt, "trajs", f"{iid}.traj.json")))
        steps = sum(1 for m in d["messages"] if m["role"] == "assistant")
        out.append(dict(iid=iid, gemini_steps=steps, gemini_resolved=res, base=r["base_commit"]))
        fo.write(f"{iid} {r['base_commit']} {py_for(r)}\n")
json.dump(out, open(outp + ".json", "w"), indent=1)
print("pool", len(pool), "| new", len(new), "| repos", dict(Counter(i.split("__")[0] for i in new)),
      "| python", dict(Counter(py_for(rows[i]) for i in new)),
      "| gemini resolved among new", sum(t["gemini_resolved"] for t in out[len(have):]), "/", len(new))
