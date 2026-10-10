"""Write each task's <pr_description> (from the Gemini trajectory's task message) to <work>/<iid>/problem.md.

Usage: python3 -I extract_problems.py <minitraj_dir> <tasks.json> <work_root>
"""
import json
import os
import re
import sys

src, tasks, work = sys.argv[1:4]
for t in json.load(open(tasks)):
    iid = t["iid"]
    d = json.load(open(os.path.join(src, "trajs", f"{iid}.traj.json")))
    u = next(m["content"] for m in d["messages"] if m["role"] == "user")
    m = re.search(r"<pr_description>\s*(?:Consider the following PR description:\s*)?(.*?)</pr_description>", u, re.S)
    body = m[1].strip()
    open(os.path.join(work, iid, "problem.md"), "w").write(body + "\n")
    print(f"{iid}: {len(body)} chars")
