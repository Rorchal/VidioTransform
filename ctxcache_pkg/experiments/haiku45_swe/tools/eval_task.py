"""Grade one task locally (no docker): gold patch as environment control, then the agent's patch.

Usage: python3 -I eval_task.py <swebv.parquet> <minitraj_dir> <work_root> <instance_id>

- The agent's final repo state (tracked + untracked) is saved once to <work>/<iid>/agent.patch, and
  restored at the end, so the repo is left exactly as the agent left it.
- Test command and test files come from the official eval log; FAIL_TO_PASS / PASS_TO_PASS and the
  test patch from SWE-bench Verified. Logs are parsed with swebench's own parsers.
"""
import gzip
import json
import os
import re
import subprocess
import sys

import pyarrow.parquet as pq
from swebench.harness.log_parsers.python import parse_log_django, parse_log_sympy

pq_path, mt, work, iid = sys.argv[1:5]
row = next(r for r in pq.read_table(pq_path).to_pylist() if r["instance_id"] == iid)
f2p, p2p = json.loads(row["FAIL_TO_PASS"]), json.loads(row["PASS_TO_PASS"])
d = os.path.join(work, iid)
repo, py = os.path.join(d, "repo"), os.path.join(d, "venv", "bin", "python")

log = gzip.open(os.path.join(mt, "logs", iid, "test_output.txt.gz"), "rt", errors="replace").read().splitlines()
start = next(i for i, l in enumerate(log) if "Start Test Output'" in l)
env, cmd = dict(os.environ), None
for l in log[start + 1:]:
    if not l.startswith("+ "):
        break
    toks = l[2:].split()
    if re.match(r"^[A-Z_]+=", toks[0]):
        k, v = toks[0].split("=", 1)
        env[k] = v
    else:
        cmd = toks
        break
if cmd[0] in ("./tests/runtests.py", "tests/runtests.py"):
    cmd = [py, "tests/runtests.py"] + cmd[1:]
elif cmd[0] == "bin/test":
    cmd = [py, "bin/test"] + cmd[1:]
else:
    sys.exit(f"unknown test command {cmd}")
parser = parse_log_django if iid.startswith("django") else parse_log_sympy


def git(*a, inp=None):
    return subprocess.run(["git", "-C", repo, *a], input=inp, capture_output=True, text=True, check=True).stdout


def reset():
    git("checkout", "-q", "-f", row["base_commit"], "--", ".")
    git("clean", "-fdq")


agent_patch = os.path.join(d, "agent.patch")
if not os.path.exists(agent_patch):
    git("add", "-A")
    open(agent_patch, "w").write(git("diff", "--cached", row["base_commit"]))
    git("reset", "-q")
ap = open(agent_patch).read()


TEST_FILES = re.findall(r"^diff --git a/(\S+) b/", row["test_patch"], flags=re.M)


def run(label, patch):
    reset()
    if patch:
        git("apply", "-", inp=patch)
    # as the official harness does: restore the test files to base before applying the test patch,
    # so an agent's own edits to those files can neither conflict with nor survive into grading
    for f in TEST_FILES:
        if subprocess.run(["git", "-C", repo, "cat-file", "-e", f"{row['base_commit']}:{f}"]).returncode == 0:
            git("checkout", "-q", row["base_commit"], "--", f)
        elif os.path.exists(os.path.join(repo, f)):
            os.remove(os.path.join(repo, f))
    git("apply", "-", inp=row["test_patch"])
    p = subprocess.run(cmd, cwd=repo, env=env, capture_output=True, text=True, timeout=3600)
    out = p.stdout + "\n" + p.stderr
    open(os.path.join(d, f"eval_{label}.log"), "w").write(out)
    st = parser(out, None)
    ok = lambda t: st.get(t) in ("PASSED", "XFAIL")
    return {"f2p_pass": [t for t in f2p if ok(t)], "f2p_fail": [t for t in f2p if not ok(t)],
            "p2p_fail": [t for t in p2p if not ok(t)], "n_parsed": len(st)}


try:
    res = {"iid": iid, "n_f2p": len(f2p), "n_p2p": len(p2p), "cmd": " ".join(cmd[1:]),
           "agent_files": re.findall(r"^diff --git a/(\S+) b/", ap, flags=re.M)}
    res["gold"] = run("gold", row["patch"])
    res["agent"] = run("agent", ap)
    gold_bad = set(res["gold"]["p2p_fail"]) | set(res["gold"]["f2p_fail"])
    res["resolved_strict"] = not res["agent"]["f2p_fail"] and not res["agent"]["p2p_fail"]
    res["resolved_env_adj"] = (not [t for t in res["agent"]["f2p_fail"] if t not in gold_bad]
                               and not [t for t in res["agent"]["p2p_fail"] if t not in gold_bad])
finally:
    reset()
    if ap:
        git("apply", "-", inp=ap)
json.dump(res, open(os.path.join(d, "eval.json"), "w"), indent=1)
print(json.dumps({k: (v if not isinstance(v, dict) else {kk: (vv if not isinstance(vv, list) else len(vv)) for kk, vv in v.items()}) for k, v in res.items()}))
