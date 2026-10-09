"""Context policies replayed over a trajectory.

Policies
  raw      keep everything
  tail10   keep only the last 10 observations (agent text kept)
  compact  structural proxy for an LLM /compact: at 90% of budget, drop everything older than the
           last 8 steps, insert a fixed 1200-token summary, keep bodies of the 5 most recently read files
  slim     our write-time slimming only (run tail / grep -> file names / read range merge / scripts on disk)
  ours     slim + write-invalidation + eviction by the 'active' rule when above 60% of budget, down to 40%

Every policy reports, per step, the context size the model would be sent, and simulates a
'miss': the agent edits a file whose content it read earlier but which the policy has removed
and the agent has not re-read since -> we charge one extra read (re-insert the body) and count it.
"""
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .model import Step, Traj, is_file, tok

REF_TOK = 25
TAIL_LINES = 15
PATHISH = re.compile(r"(?:\./)?[\w.\-]+(?:/[\w.\-]+)+\.\w+")
NARR = re.compile(r"^\s*(I will|I'll|Let me|Let's|Now I|Now,? let|Next,? I|First,? I|I need to (check|look|see|run|find|examine|verify|read)|I should (check|look|run|see))", re.I)
INFER = re.compile(r"\b(because|since|the (issue|bug|problem|root cause|error) (is|was|lies|occurs|comes)|caused by|this (confirms|means|suggests|indicates|shows|explains)|therefore|it (seems|appears|looks like)|the fix (is|should)|fails because|wrong|incorrect)\b", re.I)


def is_repo_file(k: str) -> bool:
    b = k.split("/")[-1]
    return is_file(k) and not (b.startswith("test_") or "repro" in b or k.startswith("/tmp") or b.endswith((".txt", ".bak", ".sh", ".log")))


@dataclass
class Node:
    step: int
    kind: str
    key: str                  # file path for reads/edits, cmd for runs, pattern for searches
    full: float               # tokens of the raw observation
    cost: float               # tokens currently charged to context (0 when evicted)
    paths: Set[str] = field(default_factory=set)   # files named in a search result
    ranges: List[Tuple[int, Optional[int]]] = field(default_factory=list)
    stale: bool = False
    evicted: bool = False
    last_touch: int = 0


def _tail_tok(out: str, rc) -> float:
    lines = out.splitlines()
    return tok("\n".join(lines[-TAIL_LINES:])) + 4


def _paths_in(out: str) -> Set[str]:
    return {p.lstrip("./") for p in PATHISH.findall(out)}


def _overlap_new_fraction(ranges, rng, total_lines):
    """fraction of rng's lines not covered by ranges. None hi = whole file."""
    lo, hi = rng
    if hi is None:
        hi = lo + max(total_lines, 1) - 1
    covered = 0
    span = hi - lo + 1
    # merge covered intervals clipped to [lo,hi]
    ivs = []
    for a, b in ranges:
        if b is None:
            return 0.0  # whole file already present
        a2, b2 = max(a, lo), min(b, hi)
        if a2 <= b2:
            ivs.append((a2, b2))
    ivs.sort()
    cur = None
    for a, b in ivs:
        if cur is None or a > cur[1] + 1:
            if cur:
                covered += cur[1] - cur[0] + 1
            cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    if cur:
        covered += cur[1] - cur[0] + 1
    return max(0.0, 1 - covered / span)


class Replay:
    def __init__(self, policy: str, budget: float, traj: Traj, hi: float = 0.8, lo: float = 0.5):
        assert policy in ("raw", "tail10", "compact", "slim", "ours")
        self.p = policy
        self.budget = budget
        self.hi, self.lo = hi, lo
        self.t = traj
        self.fresh_misses = 0           # body was still valid when the policy removed it
        self.compaction_calls = 0.0     # tokens spent on summarization requests (compact proxy only)
        self.nodes: List[Node] = []
        self.agent: List[float] = []       # kept agent-side tokens per step
        self.summary = 0.0
        self.compactions = 0
        self.misses = 0
        self.extra_tokens = 0.0            # tokens re-inserted by simulated re-reads
        self.refs: Set[str] = set()
        self.freq: Dict[str, int] = {}
        self.sizes: List[float] = []
        self.over_budget = 0

    # ---------- accounting ----------
    def size(self) -> float:
        s = self.t.prefix_tok + sum(n.cost for n in self.nodes) + sum(self.agent) + self.summary
        if self.p in ("slim", "ours"):
            s += REF_TOK * len(self.refs)
        return s

    def _agent_cost(self, s: Step) -> float:
        if self.p in ("slim", "ours"):
            cmd_tok = tok(s.cmd) - tok(s.script_body) + (15 if s.script_body else 0)
            text_tok = tok(s.text) if (INFER.search(s.text) and not NARR.search(s.text)) else 0
            return cmd_tok + text_tok
        return tok(s.cmd) + tok(s.text)

    def _obs_node(self, s: Step) -> Node:
        full = tok(s.out)
        key = s.keys[0] if s.keys else s.cmd[:80]
        if s.kind == "run":
            key = s.cmd
        n = Node(step=s.i, kind=s.kind, key=key, full=full, cost=full, last_touch=s.i)
        if s.kind == "search":
            n.paths = _paths_in(s.out)
        if self.p not in ("slim", "ours"):
            return n
        # write-time slimming
        if s.kind in ("run", "other"):
            n.cost = min(full, _tail_tok(s.out, s.rc))
        elif s.kind == "search":
            if n.paths:
                n.cost = min(full, tok(", ".join(sorted(n.paths)[:30])) + 6)
            else:
                n.cost = min(full, _tail_tok(s.out, s.rc))
        elif s.kind == "read" and s.keys:
            prev = [m for m in self.nodes if m.kind == "read" and m.key == key and not m.stale and not m.evicted]
            ranges = [r for m in prev for r in m.ranges]
            rng = s.rng or (1, None)
            nlines = max(1, s.out.count("\n"))
            frac = _overlap_new_fraction(ranges, rng, nlines) if ranges else 1.0
            n.cost = full * frac
            n.ranges = [rng]
        elif s.kind == "submit":
            n.cost = 0
        return n

    # ---------- invalidation (ours) ----------
    def _invalidate(self, s: Step):
        edited = {k for k in s.keys}
        for m in self.nodes:
            if m.evicted:
                continue
            if m.kind == "run" or m.kind == "gitdiff":
                m.stale = True
            elif m.kind == "read" and m.key in edited:
                m.stale = True
            elif m.kind == "search" and (m.paths & edited):
                m.stale = True

    # ---------- eviction ----------
    def _named_recent(self, key: str, upto: int) -> bool:
        base = key.split("/")[-1]
        for s in self.t.steps[max(0, upto - 2): upto + 1]:
            if base and (base in s.text or base in s.cmd):
                return True
        return False

    def _evict(self, m: Node):
        m.cost = 0
        m.evicted = True
        self.refs.add(f"{m.kind}:{m.key}")

    def _compact_ours(self, t: int):
        hi, lo = self.hi * self.budget, self.lo * self.budget
        if self.size() <= hi:
            return
        self.compactions += 1
        live = [m for m in self.nodes if not m.evicted and m.cost > 0]
        stale = sorted([m for m in live if m.stale], key=lambda m: m.step)
        for m in stale:
            if self.size() <= lo:
                return
            self._evict(m)
        inactive = [m for m in live if not m.stale and not m.evicted and
                    (t - m.last_touch > 10) and not self._named_recent(m.key, t) and self.freq.get(m.key, 0) < 3]
        for m in sorted(inactive, key=lambda m: m.last_touch):
            if self.size() <= lo:
                return
            self._evict(m)
        for m in sorted([m for m in live if not m.evicted], key=lambda m: m.last_touch):
            if self.size() <= lo:
                return
            self._evict(m)

    def _compact_proxy(self, t: int):
        if self.size() <= 0.9 * self.budget:
            return
        self.compactions += 1
        self.compaction_calls += self.size()   # the summarization request itself reads the full context once
        keep_from = max(0, t - 7)
        recent_reads = []
        for m in sorted([m for m in self.nodes if m.kind == "read" and not m.evicted], key=lambda m: -m.step):
            if m.key not in [r.key for r in recent_reads]:
                recent_reads.append(m)
            if len(recent_reads) == 5:
                break
        keep_ids = {id(m) for m in recent_reads}
        for m in self.nodes:
            if m.step < keep_from and id(m) not in keep_ids and not m.evicted:
                m.cost = 0
                m.evicted = True
        for i in range(keep_from):
            self.agent[i] = 0
        self.summary = 1200
        # a compact that still does not fit: drop the protected files, then older recent steps (harness truncation)
        if self.size() > 0.9 * self.budget:
            for m in recent_reads:
                if m.step < keep_from:
                    m.cost = 0
                    m.evicted = True
        if self.size() > 0.9 * self.budget:
            for m in self.nodes:
                if not m.evicted and m.step < t - 2:
                    m.cost = 0
                    m.evicted = True

    def _tail(self, t: int):
        for m in self.nodes:
            if not m.evicted and m.step < t - 9:
                m.cost = 0
                m.evicted = True

    # ---------- miss simulation ----------
    def _check_miss(self, s: Step):
        if s.kind != "edit":
            return
        head = s.cmd.strip().split()[:2]
        if head and (head[0] == "rm" or head == ["git", "checkout"] or head == ["git", "stash"]):
            return  # reverting / deleting does not need the file content
        for k in s.keys:
            if not is_repo_file(k):
                continue
            reads = [m for m in self.nodes if m.kind == "read" and m.key == k]
            if not reads:
                continue
            last = reads[-1]
            if last.evicted:
                # agent would have to read the file again before editing
                self.misses += 1
                edited_since = any(e.kind == "edit" and k in e.keys for e in self.t.steps[last.step + 1: s.i])
                if not edited_since:
                    self.fresh_misses += 1
                re_cost = last.full
                n = Node(step=s.i, kind="read", key=k, full=last.full, cost=re_cost, ranges=list(last.ranges), last_touch=s.i)
                self.nodes.append(n)
                self.extra_tokens += re_cost

    # ---------- main loop ----------
    def run(self):
        for s in self.t.steps:
            t = s.i
            for k in s.keys:
                self.freq[k] = self.freq.get(k, 0) + 1
                for m in self.nodes:
                    if m.key == k and not m.evicted:
                        m.last_touch = t
            self._check_miss(s)
            if self.p == "ours" and s.kind == "edit":
                self._invalidate(s)
            self.agent.append(self._agent_cost(s))
            n = self._obs_node(s)
            self.nodes.append(n)
            if self.p in ("slim", "ours") and is_file(n.key):
                self.refs.add(f"file:{n.key}")
            if self.p == "tail10":
                self._tail(t)
            elif self.p == "compact":
                self._compact_proxy(t)
            elif self.p == "ours":
                self._compact_ours(t)
            sz = self.size()
            self.sizes.append(sz)
            if sz > self.budget:
                self.over_budget += 1
        return {
            "policy": self.p, "budget": self.budget, "id": self.t.id, "steps": len(self.t.steps),
            "peak": max(self.sizes) if self.sizes else 0,
            "sent": sum(self.sizes) + self.compaction_calls,
            "compactions": self.compactions,
            "misses": self.misses,
            "fresh_misses": self.fresh_misses,
            "hi": self.hi, "lo": self.lo,
            "over_budget_steps": self.over_budget,
            "sizes": self.sizes,
        }
