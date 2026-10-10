"""Pick django/sympy SWE-bench Verified tasks from the Gemini mini-SWE-agent run.

Usage: python3 -I pick_tasks.py <minitraj_dir> <out_tasks.json>
Base commit is recovered from the eval log (`git checkout <sha> <test files>`).
"""
import glob
import gzip
import json
import os
import random
import re
import statistics
import sys

src, out = sys.argv[1], sys.argv[2]
rows = []
for f in sorted(glob.glob(os.path.join(src, "trajs", "*.json"))):
    iid = os.path.basename(f).split(".traj")[0]
    if not iid.startswith(("django__", "sympy__")):
        continue
    d = json.load(open(f))
    n = sum(1 for m in d["messages"] if m["role"] == "assistant")
    rp = os.path.join(src, "logs", iid, "report.json")
    res = json.load(open(rp))[iid]["resolved"] if os.path.exists(rp) else None
    to = os.path.join(src, "logs", iid, "test_output.txt.gz")
    sha = None
    if os.path.exists(to):
        m = re.search(r"git checkout ([0-9a-f]{40}) ", gzip.open(to, "rt", errors="replace").read())
        sha = m and m[1]
    rows.append((iid, n, res, sha))

print("django+sympy:", len(rows), " with sha:", sum(1 for r in rows if r[3]), " resolved:", sum(1 for r in rows if r[2]))
print("steps median", statistics.median(r[1] for r in rows))
long = [r for r in rows if r[1] >= 50 and r[3] and r[2]]
mid = [r for r in rows if 25 <= r[1] < 50 and r[3] and r[2]]
print("long&resolved", len(long), " mid&resolved", len(mid))
random.seed(0)
pick = random.sample(long, 7) + random.sample(mid, 3)
json.dump([dict(iid=a, gemini_steps=b, gemini_resolved=c, base=d) for a, b, c, d in pick], open(out, "w"), indent=1)
for r in pick:
    print(r)
