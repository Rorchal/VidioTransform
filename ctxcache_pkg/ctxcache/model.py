"""Step model + parsers for two trajectory formats.

A Step is one agent action and its observation:
    kind   : read | search | run | edit | gitdiff | submit | other
    keys   : file paths (or other identifiers) the action touches, canonicalized by canon()
    cmd    : the command / tool call rendered as text (incl. edit bodies for Claude Code)
    text   : the agent's own prose for that step
    out    : the raw observation text
    rc     : return code if known
    rng    : (lo, hi) line range for reads, (1, None) = whole file
    cwd    : directory the action ran in (relative paths in cmd/out resolve against it)
    call   : index of the model call that issued the step (several steps share a call when the
             model makes parallel tool calls)

Token costs. By default they come from the API usage recorded in the trajectory: the context the
model received on call k+1 minus the context on call k is what step(s) k added. Of that, the model's
own output (usage output/completion tokens) is the agent side; the rest is the observation (tool
results plus anything the harness injected). Claude's output also contains thinking, which is
resent within a tool loop but is not visible in the transcript; it is carried as think_tok. Steps
without usage (the last call, or a call where the context shrank) fall back to chars/4 scaled by
the trajectory's median real/estimated ratio. set_token_source(False) restores plain chars/4.
"""
import json
import os
import re
import sys
from dataclasses import dataclass, field
from statistics import median
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from agent_parse import classify, split_prefix  # noqa: E402

CH = 4.0  # chars per token
USE_REAL = True


def set_token_source(real: bool):
    global USE_REAL
    USE_REAL = real


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
    script_body: str = ""  # inline script / edit body inside the call (python -c / heredoc / Edit / Write)
    cwd: str = ""
    call: int = -1
    obs_tok: Optional[float] = None    # real tokens of the observation (None: use chars/4)
    agent_scale: float = 1.0           # real / estimated tokens of the visible agent side
    think_tok: float = 0.0             # thinking tokens resent with this step (first step of a call)
    ctx_next: Optional[float] = None   # real context of the next model call (for calibration)


@dataclass
class Traj:
    id: str
    prefix_tok: float  # system prompt + tool definitions + task description
    steps: List[Step] = field(default_factory=list)
    patch_files: List[str] = field(default_factory=list)
    root: str = ""                 # paths inside root are keyed relative to it
    real_ctx: List[float] = field(default_factory=list)  # real context per model call, if recorded


SED_RANGE = re.compile(r"sed\s+-n\s+'?(\d+),(\d+)p'?\s+(\S+)")
HEAD_N = re.compile(r"head\s+-n?\s*(\d+)\s+(\S+)")
PY_INLINE = re.compile(r"python3?\s+-c\s+([\"'])(.*?)\1\s*$", re.S)
HEREDOC = re.compile(r"<<\s*-?\s*['\"]?(\w+)['\"]?\s*>\s*\S+\n(.*?)\n\1", re.S)


def canon(p: str, cwd: str = "", root: str = "") -> str:
    """One spelling per file: resolve against cwd, then key relative to root when inside it.
    Without a cwd, falls back to the old behaviour (strip leading ./)."""
    if not p or p.startswith("<"):
        return p
    p = p.strip("\"'")
    if not cwd and not os.path.isabs(p):
        return p.lstrip("./")
    p = os.path.normpath(os.path.join(cwd, p))
    if root and (p == root or p.startswith(root.rstrip("/") + "/")):
        return os.path.relpath(p, root)
    return p


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


def _int(v):
    """Tool inputs are model output: offset/limit occasionally arrive as a list or string."""
    if isinstance(v, list):
        v = v[0] if v else None
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _apply_real(steps: List[Step], calls: List[dict]):
    """calls: [{ctx, out, steps:[Step], visible}] in order. Fills obs_tok / agent_scale / think_tok."""
    for k, c in enumerate(calls[:-1]):
        for s in c["steps"]:
            s.ctx_next = calls[k + 1]["ctx"]
        add = calls[k + 1]["ctx"] - c["ctx"]
        obs = add - c["out"]
        if add <= 0 or obs < 0 or not c["steps"]:
            continue
        chars = [len(s.out) for s in c["steps"]]
        tot = sum(chars)
        for s, n in zip(c["steps"], chars):
            s.obs_tok = obs * (n / tot if tot else 1 / len(chars))
        vis = c["visible"]
        if c.get("thinking"):
            c["steps"][0].think_tok = max(0.0, c["out"] - vis)
        elif vis > 0:
            for s in c["steps"]:
                s.agent_scale = c["out"] / vis
    ratios = [s.obs_tok / tok(s.out) for s in steps if s.obs_tok is not None and len(s.out) > 80]
    r = median(ratios) if ratios else 1.0
    for s in steps:
        if s.obs_tok is None:
            s.obs_tok = tok(s.out) * r


# ---------- mini-SWE-agent .traj.json ----------
MINI_ROOT = "/testbed"


def parse_mini(path: str) -> Traj:
    d = json.load(open(path))
    msgs = d["messages"]
    prefix = 0.0
    steps: List[Step] = []
    calls = []
    for m in msgs:
        role = m["role"]
        if role == "system":
            prefix += tok(m.get("content") or "")
        elif role == "user" and not steps:
            prefix += tok(m.get("content") or "")
        elif role == "assistant":
            ex = m.get("extra") or {}
            acts = ex.get("actions", [])
            cmd = acts[0]["command"] if acts else ""
            kind, keys = classify(cmd) if cmd else ("other", [])
            _, cd = split_prefix(cmd)
            cwd = os.path.normpath(os.path.join(MINI_ROOT, cd)) if cd else MINI_ROOT
            s = Step(
                i=len(steps), kind=kind, keys=[canon(k, cwd, MINI_ROOT) for k in keys], cmd=cmd,
                text=(m.get("content") or ""), out="",
                rng=_range(cmd) if kind == "read" else None,
                script_body=_script_body(cmd), cwd=cwd, call=len(steps),
            )
            steps.append(s)
            u = (ex.get("response") or {}).get("usage") or {}
            if u.get("prompt_tokens") is not None:
                # Gemini reasoning tokens are not resent by mini, so completion_tokens is the visible message
                calls.append(dict(ctx=u["prompt_tokens"], out=u.get("completion_tokens") or 0, steps=[s],
                                  visible=tok(cmd) + tok(s.text)))
        elif role == "tool" and steps:
            steps[-1].out = m.get("content") or ""
            steps[-1].rc = (m.get("extra") or {}).get("returncode")
        elif role == "user" and steps:
            steps[-1].out += m.get("content") or ""
    patch = (d.get("info") or {}).get("submission") or ""
    pf = re.findall(r"^diff --git a/(\S+) b/", patch, flags=re.M)
    t = Traj(id=d.get("instance_id") or os.path.basename(path), prefix_tok=prefix, steps=steps, patch_files=pf,
             root=MINI_ROOT)
    if USE_REAL and len(calls) == len(steps) and calls:
        _apply_real(steps, calls)
        t.prefix_tok = calls[0]["ctx"]
        t.real_ctx = [c["ctx"] for c in calls]
    return t


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


def _edit_body(name, inp):
    if name == "Write":
        return inp.get("content") or ""
    if name == "Edit":
        return f"{inp.get('old_string') or ''}\n{inp.get('new_string') or ''}"
    if name == "MultiEdit":
        return "\n".join(f"{e.get('old_string') or ''}\n{e.get('new_string') or ''}" for e in inp.get("edits") or [])
    if name == "NotebookEdit":
        return inp.get("new_source") or ""
    return ""


def parse_claude_code(path: str) -> Traj:
    steps: List[Step] = []
    pending = {}  # tool_use_id -> step
    prefix = 0.0
    first_user_seen = False
    root = ""
    calls, call_of = [], {}  # message.id -> call dict
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(e, dict):
            continue
        root = root or (e.get("cwd") or "")
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
            mid = msg.get("id") or f"_{len(calls)}"
            c = call_of.get(mid)
            if c is None:
                c = call_of[mid] = dict(ctx=None, out=0, steps=[], visible=0.0, thinking=False, prose="")
                calls.append(c)
            u = msg.get("usage") or {}
            if u:
                c["ctx"] = u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0) + u.get("cache_creation_input_tokens", 0)
                c["out"] = max(c["out"], u.get("output_tokens") or 0)
            ecwd = e.get("cwd") or root
            # one API message is often written as several entries (text, then each tool_use): buffer
            # the prose per message and attach it to the message's first tool call
            prose = "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
            c["prose"] = "\n".join(x for x in (c["prose"], prose) if x)
            c["visible"] += tok(prose)
            c["thinking"] |= any(isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking") for b in content)
            for b in content:
                if not isinstance(b, dict) or b.get("type") != "tool_use":
                    continue
                name = b.get("name", "")
                inp = b.get("input") or {}
                cwd, body = ecwd, ""
                if name == "Bash":
                    cmd = inp.get("command", "")
                    kind, keys = classify(cmd) if cmd else ("other", [])
                    _, cd = split_prefix(cmd)
                    cwd = os.path.normpath(os.path.join(ecwd, cd)) if cd else ecwd
                    rng = _range(cmd) if kind == "read" else None
                    body = _script_body(cmd)
                elif name == "Read":
                    cmd = f"Read {inp.get('file_path','')}"
                    kind, keys = "read", [inp.get("file_path", "")]
                    off = _int(inp.get("offset")) or 1
                    lim = _int(inp.get("limit"))
                    rng = (off, off + lim - 1) if lim else (1, None)
                elif name in ("Edit", "Write", "MultiEdit", "NotebookEdit"):
                    body = _edit_body(name, inp)
                    cmd = f"{name} {inp.get('file_path') or inp.get('notebook_path','')}\n{body}"
                    kind, keys, rng = "edit", [inp.get("file_path") or inp.get("notebook_path", "")], None
                elif name in ("Grep", "Glob", "LS"):
                    cmd = f"{name} {inp.get('pattern','')} {inp.get('path','')}"
                    kind, keys, rng = "search", [inp.get("path", "")] if inp.get("path") else [], None
                else:
                    cmd = f"{name} {json.dumps(inp)[:200]}"
                    kind, keys, rng = CC_KIND.get(name, "other") or "other", [], None
                keys = [canon(k, cwd, root) for k in keys if k]
                s = Step(i=len(steps), kind=kind, keys=keys, cmd=cmd, text=c["prose"], out="", rng=rng,
                         script_body=body, cwd=cwd, call=calls.index(c))
                c["visible"] += tok(cmd)
                c["prose"] = ""
                c["steps"].append(s)
                steps.append(s)
                pending[b.get("id")] = s
        elif role == "user":
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    s = pending.pop(b.get("tool_use_id"), None)
                    if s is not None:
                        s.out = _cc_text(b.get("content"))
    for c in calls:  # prose written after the message's last tool call
        if c["prose"] and c["steps"]:
            c["steps"][-1].text = "\n".join(x for x in (c["steps"][-1].text, c["prose"]) if x)
    t = Traj(id=os.path.basename(path), prefix_tok=prefix, steps=steps, patch_files=[], root=root)
    calls = [c for c in calls if c["ctx"] is not None]
    if USE_REAL and calls:
        _apply_real(steps, calls)
        t.prefix_tok = calls[0]["ctx"]
        t.real_ctx = [c["ctx"] for c in calls]
    return t


def load(path: str) -> Traj:
    return parse_claude_code(path) if path.endswith(".jsonl") else parse_mini(path)
