"""Step model + parsers for two trajectory formats.

A Step is one agent action and its observation:
    kind   : read | search | run | edit | gitdiff | submit | other
    keys   : file paths (or other identifiers) the action touches
    cmd    : the command / tool call rendered as text
    text   : the agent's own prose for that step
    out    : the raw observation text
    rc     : return code if known
    rng    : (lo, hi) line range for reads, (1, None) = whole file
"""
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from agent_parse import classify  # noqa: E402

CH = 4.0  # chars per token


def tok(s: str) -> float:
    return len(s) / CH


@dataclass
class Step:
    i: int
    kind: str
    keys: List[str]
    cmd: str
    text: str
    out: str
    rc: Optional[int] = None
    rng: Optional[Tuple[int, Optional[int]]] = None
    script_body: str = ""  # inline script text inside the call (python -c / heredoc)


@dataclass
class Traj:
    id: str
    prefix_tok: float  # system prompt + task description
    steps: List[Step] = field(default_factory=list)
    patch_files: List[str] = field(default_factory=list)


SED_RANGE = re.compile(r"sed\s+-n\s+'?(\d+),(\d+)p'?\s+(\S+)")
HEAD_N = re.compile(r"head\s+-n?\s*(\d+)\s+(\S+)")
PY_INLINE = re.compile(r"python3?\s+-c\s+([\"'])(.*?)\1\s*$", re.S)
HEREDOC = re.compile(r"<<\s*-?\s*['\"]?(\w+)['\"]?\s*>\s*\S+\n(.*?)\n\1", re.S)


def norm(k: str) -> str:
    return k.lstrip("./")


def is_file(k: str) -> bool:
    return "." in k.split("/")[-1] and not k.startswith("<")


def _range(cmd):
    m = SED_RANGE.search(cmd)
    if m:
        return (int(m[1]), int(m[2]))
    m = HEAD_N.search(cmd)
    if m:
        return (1, int(m[1]))
    return (1, None)


def _script_body(cmd):
    for rx in (PY_INLINE, HEREDOC):
        m = rx.search(cmd)
        if m:
            return m.group(2)
    return ""


# ---------- mini-SWE-agent .traj.json ----------
def parse_mini(path: str) -> Traj:
    d = json.load(open(path))
    msgs = d["messages"]
    prefix = 0.0
    steps: List[Step] = []
    for m in msgs:
        role = m["role"]
        if role == "system":
            prefix += tok(m.get("content") or "")
        elif role == "user" and not steps:
            prefix += tok(m.get("content") or "")
        elif role == "assistant":
            acts = (m.get("extra") or {}).get("actions", [])
            cmd = acts[0]["command"] if acts else ""
            kind, keys = classify(cmd) if cmd else ("other", [])
            keys = [norm(k) for k in keys]
            steps.append(
                Step(
                    i=len(steps), kind=kind, keys=keys, cmd=cmd,
                    text=(m.get("content") or ""), out="",
                    rng=_range(cmd) if kind == "read" else None,
                    script_body=_script_body(cmd),
                )
            )
        elif role == "tool" and steps:
            steps[-1].out = m.get("content") or ""
            steps[-1].rc = (m.get("extra") or {}).get("returncode")
        elif role == "user" and steps:
            steps[-1].out += m.get("content") or ""
    patch = (d.get("info") or {}).get("submission") or ""
    pf = re.findall(r"^diff --git a/(\S+) b/", patch, flags=re.M)
    return Traj(id=d.get("instance_id") or os.path.basename(path), prefix_tok=prefix, steps=steps, patch_files=pf)


# ---------- Claude Code ~/.claude/projects/<proj>/<session>.jsonl (best effort) ----------
# Entry format is internal to Claude Code and changes between versions; this parser is tolerant:
# it looks for message.content blocks of type tool_use / tool_result and maps tool names to kinds.
CC_KIND = {
    "Read": "read", "Grep": "search", "Glob": "search", "LS": "search",
    "Edit": "edit", "Write": "edit", "MultiEdit": "edit", "NotebookEdit": "edit",
    "Bash": None,  # decided by classify() on the command
    "WebFetch": "other", "WebSearch": "other", "Agent": "other", "Task": "other",
}


def _cc_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            (b.get("text") or "") if isinstance(b, dict) else str(b) for b in content
        )
    return str(content or "")


def parse_claude_code(path: str) -> Traj:
    steps: List[Step] = []
    pending = {}  # tool_use_id -> step
    prefix = 0.0
    first_user_seen = False
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = e.get("message") or {}
        role = msg.get("role") or e.get("type")
        content = msg.get("content")
        if role == "user" and not first_user_seen and isinstance(content, str):
            prefix += tok(content)
            first_user_seen = True
            continue
        if not isinstance(content, list):
            continue
        if role == "assistant":
            prose = "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
            for b in content:
                if not isinstance(b, dict) or b.get("type") != "tool_use":
                    continue
                name = b.get("name", "")
                inp = b.get("input") or {}
                if name == "Bash":
                    cmd = inp.get("command", "")
                    kind, keys = classify(cmd) if cmd else ("other", [])
                    rng = _range(cmd) if kind == "read" else None
                elif name == "Read":
                    cmd = f"Read {inp.get('file_path','')}"
                    kind, keys = "read", [inp.get("file_path", "")]
                    off = inp.get("offset") or 1
                    lim = inp.get("limit")
                    rng = (off, off + lim - 1) if lim else (1, None)
                elif name in ("Edit", "Write", "MultiEdit", "NotebookEdit"):
                    cmd = f"{name} {inp.get('file_path') or inp.get('notebook_path','')}"
                    kind, keys, rng = "edit", [inp.get("file_path") or inp.get("notebook_path", "")], None
                elif name in ("Grep", "Glob", "LS"):
                    cmd = f"{name} {inp.get('pattern','')} {inp.get('path','')}"
                    kind, keys, rng = "search", [inp.get("path", "")] if inp.get("path") else [], None
                else:
                    cmd = f"{name} {json.dumps(inp)[:200]}"
                    kind, keys, rng = CC_KIND.get(name, "other") or "other", [], None
                keys = [norm(k) for k in keys if k]
                s = Step(i=len(steps), kind=kind, keys=keys, cmd=cmd, text=prose, out="", rng=rng,
                         script_body=_script_body(cmd) if name == "Bash" else (inp.get("content", "") if name == "Write" else ""))
                prose = ""  # attach prose to first tool call of the message only
                steps.append(s)
                pending[b.get("id")] = s
        elif role == "user":
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    s = pending.pop(b.get("tool_use_id"), None)
                    if s is not None:
                        s.out = _cc_text(b.get("content"))
    return Traj(id=os.path.basename(path), prefix_tok=prefix, steps=steps, patch_files=[])


def load(path: str) -> Traj:
    return parse_claude_code(path) if path.endswith(".jsonl") else parse_mini(path)
