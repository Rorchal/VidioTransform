"""Cheap, model-free edge sources: shared identifiers, value supersession,
proposal rejection. These approximate what an LLM edge-inferrer would output;
the gap between them and the oracle edges is measured by the eval."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from .model import Edge, EdgeType, Graph, Role

IDENT_RE = re.compile(
    r"""
    (?:[\w./-]+/)?[\w-]+\.(?:py|js|ts|sql|md|ya?ml|json|toml|txt|log|cfg|ini)\b   # file paths
  | \b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b                                              # CONSTANT_NAME
  | \b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b                                              # snake_case
  | \b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+(?:\.\w+)?(?:\(\))?                         # CamelCase(.method)
  | \b[a-z][A-Za-z0-9]*\(\)                                                        # func()
  | \b\d+(?:\.\d+)?(?:ms|s|sec|m|h|MB|GB|KB|%)\b                                   # number + unit
  | \breq-[a-f0-9]+\b                                                              # request ids
  | \bE\d{3,5}\b                                                                   # error codes
    """,
    re.X,
)

ASSIGN_RE = re.compile(
    r"\b(?:set|change|update|switch|bump|lower|raise|configure|keep)\s+([A-Za-z_][\w.]*)\s+(?:to|at|=|:)\s*([^\s,.;()]+)"
    r"|\b([A-Z][A-Z0-9_]{2,})\s*(?:=|:|\bis(?: now)?)\s*([^\s,.;()]+)",
    re.I,
)

REJECT_RE = re.compile(
    r"\b(?:no,?\s|don'?t|do not|not going to|won'?t|reject(?:ed)?|skip|drop|rather not|instead of|scrap|forget)\b",
    re.I,
)
STOPWORDS = {"this", "that", "with", "from", "into", "have", "want", "will", "just", "then",
             "than", "them", "they", "your", "please", "instead", "about", "should", "would"}


def identifiers(text: str) -> set[str]:
    out = set()
    for m in IDENT_RE.finditer(text):
        ident = m.group(0).removesuffix("()")     # loadProfile() and loadProfile are the same thing
        if len(ident) >= 3:
            out.add(ident)
    return out


def add_symbolic_edges(g: Graph, max_df_ratio: float = 0.3) -> int:
    """Link nodes that share a distinctive identifier.
    later -> earlier: REFERS (0.5); earlier -> later: MENTIONS (0.3).
    Identifiers that appear in too many nodes carry no signal and are skipped."""
    nodes = g.ordered()
    n = len(nodes)
    max_df = max(2, int(max_df_ratio * n))
    where: dict[str, list[int]] = defaultdict(list)
    for i, node in enumerate(nodes):
        for ident in identifiers(node.text):
            where[ident].append(i)
    added = 0
    for ident, idxs in where.items():
        if len(idxs) < 2 or len(idxs) > max_df:
            continue
        for ai in range(len(idxs)):
            for bi in range(ai + 1, len(idxs)):
                a, b = nodes[idxs[ai]], nodes[idxs[bi]]
                both_tool = a.role == Role.TOOL_RESULT and b.role == Role.TOOL_RESULT
                refers = 0.25 if both_tool else 0.5
                mentions = 0.15 if both_tool else 0.3
                g.add_edge(Edge(b.id, a.id, EdgeType.REFERS, refers, source="symbolic"))
                g.add_edge(Edge(a.id, b.id, EdgeType.MENTIONS, mentions, source="symbolic"))
                added += 2
    return added


def _is_key(key: str) -> bool:
    return "_" in key or "." in key or (key.isupper() and len(key) >= 3)


def extract_assignments(text: str) -> list[tuple[str, str]]:
    out = []
    for m in ASSIGN_RE.finditer(text):
        key = m.group(1) or m.group(3)
        val = m.group(2) or m.group(4)
        if key and val and _is_key(key):
            out.append((key.upper(), val.strip("\"'`").lower()))
    return out


def detect_supersedes(g: Graph) -> int:
    """A later assignment to the same key with a different value supersedes the
    earlier one. Tool outputs are skipped (a config dump is not a decision)."""
    latest: dict[str, tuple[str, str]] = {}
    added = 0
    for node in g.ordered():
        if node.role in (Role.TOOL_RESULT, Role.TOOL_CALL):
            continue
        for key, val in extract_assignments(node.text):
            prev = latest.get(key)
            if prev and prev[1] != val and prev[0] != node.id:
                old = g.nodes[prev[0]]
                if old.superseded_by is None:
                    old.superseded_by = node.id
                    g.add_edge(Edge(node.id, old.id, EdgeType.SUPERSEDES, 0.3, source="symbolic"))
                    added += 1
            latest[key] = (node.id, val)
    return added


def _content_words(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9]{3,}", text)} - STOPWORDS


def detect_rejects(g: Graph, lookback: int = 12) -> int:
    """A user message with rejection wording that shares a content word with a
    recent proposal rejects that proposal."""
    nodes = g.ordered()
    added = 0
    for i, node in enumerate(nodes):
        if node.role not in (Role.USER, Role.CONSTRAINT, Role.GOAL) or not REJECT_RE.search(node.text):
            continue
        words = _content_words(node.text)
        for j in range(i - 1, max(-1, i - lookback - 1), -1):
            cand = nodes[j]
            if cand.meta.get("proposal") and cand.rejected_by is None and words & _content_words(cand.text):
                cand.rejected_by = node.id
                g.add_edge(Edge(node.id, cand.id, EdgeType.REJECTS, 0.3, source="symbolic"))
                added += 1
                break
    return added


# --------------------------------------------------------------------------- lexical edges

LEX_STOP = set("""a an the and or but if then than so of to in on at by for from with without into onto over under
about as is are was were be been being am do does did done have has had having will would shall should can could may
might must i me my mine you your yours he him his she her hers it its we us our ours they them their theirs this that
these those there here what which who whom whose when where why how all any both each few more most other some such no
nor not only own same too very just also ever never now again further once because while during before after above
below between through up down out off again please tell know think like want need get got make made say said one two
new old much many lot lots thing things time times way ways day days yes okay ok hi hello thanks thank""".split())
LEX_WORD_RE = re.compile(r"[a-z][a-z0-9'\-]{2,}")


def content_words(text: str) -> set[str]:
    return {w for w in LEX_WORD_RE.findall(text.lower()) if w not in LEX_STOP}


def add_lexical_edges(g: Graph, query_ids: list[str], top_k: int = 8, min_score: float = 0.1,
                      max_strength: float = 0.9) -> int:
    """Retrieval-style edges for prose: from a query node (the latest ask) to the
    earlier nodes sharing its rarer content words. Score = idf-weighted share of
    the query's words found in the node; the top_k nodes above min_score get a
    REFERS edge of strength 0.3 + score (capped). Model-free; what the symbolic
    identifier edges are for code, this is for chat."""
    nodes = g.ordered()
    words = {n.id: content_words(n.text) for n in nodes}
    df: Counter = Counter()
    for ws in words.values():
        df.update(ws)
    n_docs = len(nodes)

    def idf(w: str) -> float:
        return math.log((n_docs + 1) / (df[w] + 0.5))

    added = 0
    for qid in query_ids:
        if qid not in g.nodes:
            continue
        qw = words[qid]
        qnorm = sum(idf(w) for w in qw)
        if not qw or qnorm <= 0:
            continue
        qseq = g.nodes[qid].seq
        scored = []
        for n in nodes:
            if n.seq >= qseq:
                continue
            common = qw & words[n.id]
            if not common:
                continue
            score = sum(idf(w) for w in common) / qnorm
            if score >= min_score:
                scored.append((score, n.id))
        for score, nid in sorted(scored, reverse=True)[:top_k]:
            g.add_edge(Edge(qid, nid, EdgeType.REFERS, min(max_strength, 0.3 + score), source="lexical"))
            added += 1
    return added
