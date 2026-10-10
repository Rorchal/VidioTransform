"""Parse mini-SWE-agent trajectories into a flat step table.

Usage: python -I agent_parse.py <trajs_dir> <results.json> <out.jsonl>
Each output line = one trajectory with its step list.
"""
import glob
import json
import os
import re
import sys

PATH_EXT = r"(?:py|pyx|pxd|txt|rst|cfg|toml|ini|yml|yaml|json|md|c|h|cpp|html|tex|csv|in|po|js|ts|conf)"
# path-like tokens: contains a slash, or bare filename with a known extension
PATH_RE = re.compile(
    r"(?<![\w@:])((?:\.{0,2}/)?(?:[\w.\-]+/)+[\w.\-*]*|[\w\-]+\." + PATH_EXT + r")\b"
)
WRITE_OPEN_RE = re.compile(r"""open\(\s*(['"])([^'"]+)\1\s*,\s*(['"])[wa]""")
HEREDOC_RE = re.compile(r"cat\s*<<\s*-?\s*['\"]?\w+['\"]?\s*>\s*([\w./\-]+)")
REDIRECT_RE = re.compile(r"(?:^|[^>])>\s*([\w./\-]+\.\w+)\s*(?:<<|$|\n)")
SEDI_RE = re.compile(r"sed\s+-i[^\n]*?\s([\w./\-]+\.\w+)")
WRITE_TEXT_RE = re.compile(r"""Path\(\s*(['"])([^'"]+)\1\s*\)\.write_text""")
GIT_CHECKOUT_RE = re.compile(r"git\s+checkout\s+(?:--\s+)?([\w./\-]+\.\w+)")

READ_HEADS = {"cat", "sed", "head", "tail", "less", "more"}
SEARCH_HEADS = {"grep", "find", "rg", "ls", "ack"}
RUN_HEADS = {"python", "python3", "pytest", "py.test", "tox", "bin/test", "./bin/test", "./tests/runtests.py"}


def clean_paths(cmd):
    out = []
    for m in PATH_RE.finditer(cmd):
        p = m.group(1).rstrip(".,;:'\")")
        if p.startswith("http") or p in ("/", "./", "../"):
            continue
        if p.endswith("/"):
            p = p.rstrip("/")
        if not p:
            continue
        out.append(p)
    return out


PREFIX_CD = re.compile(r"cd\s+(\S+)\s*(?:&&|;|\n)\s*")
PREFIX_VAR = re.compile(r"([A-Za-z_]\w*)=\S+\s*(?:&&|;|\n)\s*")
PREFIX_TIMEOUT = re.compile(r"timeout\s+\S+\s+")
PYTHON_HEAD = re.compile(r"(?:^|/)python[\d.]*$")


def split_prefix(cmd):
    """Strip leading `cd DIR &&`, `VAR=value;` and `timeout N` segments, and rewrite a head that is
    a python binary by path (`.../venv/bin/python3.9`) or a variable set in the prefix (`$PY`) to
    `python`. Returns (command, cd_dir or None). Agents in persistent shells (Claude Code) write most
    commands this way; without it every such command classifies as 'other'."""
    c, cd, names = cmd.strip(), None, set()
    while True:
        m = PREFIX_CD.match(c)
        if m:
            d = m[1].strip("\"'")
            cd = d if cd is None else os.path.join(cd, d)
            c = c[m.end():]
            continue
        m = PREFIX_VAR.match(c) or PREFIX_TIMEOUT.match(c)
        if m:
            if m.re is PREFIX_VAR:
                names.add(m[1])
            c = c[m.end():]
            continue
        break
    head, sep, rest = c.partition(" ")
    h = head.strip("\"'")
    if PYTHON_HEAD.search(h) or h.lstrip("$").strip("{}") in names:
        c = "python" + sep + rest
    return c, cd


def classify(cmd):
    """Return (kind, keys). kind in read/search/edit/run/gitdiff/submit/other."""
    c, _ = split_prefix(cmd)
    # strip leading env assignments
    tokens = c.split()
    head = ""
    for t in tokens:
        if re.match(r"^[A-Z_][A-Z0-9_]*=", t):
            continue
        head = t
        break
    if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in c:
        return "submit", []
    edit_keys = []
    for rx in (WRITE_OPEN_RE, WRITE_TEXT_RE):
        edit_keys += [m.group(2) for m in rx.finditer(c)]
    edit_keys += [m.group(1) for m in HEREDOC_RE.finditer(c)]
    edit_keys += [m.group(1) for m in SEDI_RE.finditer(c)]
    edit_keys += [m.group(1) for m in GIT_CHECKOUT_RE.finditer(c)]
    if head in ("echo", "printf") or head.startswith("cat"):
        edit_keys += [m.group(1) for m in REDIRECT_RE.finditer(c)]
    edit_keys = [k for k in edit_keys if not k.startswith("/dev/")]
    if edit_keys:
        return "edit", list(dict.fromkeys(edit_keys))
    if head == "git" and len(tokens) > 1 and tokens[1] == "diff":
        ps = clean_paths(c)
        return "gitdiff", (["<git diff>:" + p for p in ps] if ps else ["<git diff>"])
    if head == "rm":
        return "edit", clean_paths(c)
    paths = clean_paths(c)
    if head == "git" and len(tokens) > 1 and tokens[1] in ("grep", "ls-files"):
        return "search", paths
    if head in READ_HEADS:
        return "read", paths
    if head in SEARCH_HEADS:
        return "search", paths
    if head in RUN_HEADS or head.endswith("runtests.py") or head.endswith("/test"):
        return "run", paths
    return "other", paths


def parse_traj(path):
    d = json.load(open(path))
    msgs = d["messages"]
    steps = []
    sys_chars = 0
    task_chars = 0
    for i, m in enumerate(msgs):
        if m["role"] == "system":
            sys_chars += len(m.get("content") or "")
        elif m["role"] == "user" and not steps:
            task_chars += len(m.get("content") or "")
        elif m["role"] == "assistant":
            actions = (m.get("extra") or {}).get("actions", [])
            cmd = actions[0]["command"] if actions else ""
            kind, keys = classify(cmd) if cmd else ("other", [])
            steps.append(
                {
                    "i": len(steps),
                    "kind": kind,
                    "keys": keys,
                    "cmd": cmd[:200],
                    "a_chars": len(m.get("content") or "") + len(cmd),
                    "o_chars": 0,
                    "o_rc": None,
                }
            )
        elif m["role"] == "tool" and steps:
            content = m.get("content") or ""
            steps[-1]["o_chars"] = len(content)
            ex = m.get("extra") or {}
            steps[-1]["o_rc"] = ex.get("returncode")
        elif m["role"] == "user" and steps:
            # non-tool user message (e.g. format error reminder)
            steps[-1]["o_chars"] += len(m.get("content") or "")
    patch = (d.get("info") or {}).get("submission") or ""
    patch_files = re.findall(r"^diff --git a/(\S+) b/", patch, flags=re.M)
    return {
        "instance_id": d.get("instance_id") or os.path.basename(path).split(".traj")[0],
        "sys_chars": sys_chars,
        "task_chars": task_chars,
        "steps": steps,
        "patch_files": patch_files,
        "exit_status": (d.get("info") or {}).get("exit_status"),
    }


def main():
    trajs_dir, results_json, out = sys.argv[1:4]
    res = json.load(open(results_json))
    resolved = set(res.get("resolved", []))
    n = 0
    with open(out, "w") as fo:
        for f in sorted(glob.glob(os.path.join(trajs_dir, "*.json"))):
            t = parse_traj(f)
            t["resolved"] = t["instance_id"] in resolved
            fo.write(json.dumps(t) + "\n")
            n += 1
    print("parsed", n)


if __name__ == "__main__":
    main()
