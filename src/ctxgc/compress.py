"""Pipeline entry points and baselines."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .allocate import allocate
from .edges import add_symbolic_edges, detect_rejects, detect_supersedes
from .ingest import assign_roots, ingest
from .model import L0, L3, LEVELS, Edge, EdgeType, Graph, Role
from .propagate import propagate, score as score_nodes
from .render import render
from .summarize import ExtractiveSummarizer, Summarizer, conversation_hints
from .tokens import count


@dataclass
class Oracle:
    """Ground truth a perfect edge-inferrer would produce (from synthetic data or
    hand labelling): node roles, dependency edges, supersessions, rejections."""
    roles: dict[str, str] = field(default_factory=dict)
    edges: list[tuple[str, str, float]] = field(default_factory=list)   # (src, dst, strength)
    supersedes: list[tuple[str, str]] = field(default_factory=list)    # (new, old)
    rejects: list[tuple[str, str]] = field(default_factory=list)       # (rejection, proposal)


@dataclass
class Result:
    method: str
    text: str
    tokens: int
    level: dict[str, int]
    dropped: set[str]
    score: dict[str, float]
    graph: Graph


def build_graph(
    messages: list[dict],
    *,
    split_facts: bool = True,
    symbolic: bool = True,
    chain: bool = True,
    tombstones: bool = True,
    oracle: Oracle | None = None,
    llm=None,
    llm_heuristics: bool = False,
    llm_workers: int = 8,
) -> Graph:
    if llm is not None:
        # model-based mark phase (see llm.infer_graph). By default the model
        # replaces the heuristic edge sources (chain/serves/symbolic/regex
        # tombstones) the way the oracle does; llm_heuristics=True keeps them
        # too and the model's edges are added on top.
        from .llm import infer_graph
        g = ingest(messages, split_facts=split_facts, chain=llm_heuristics and chain, serves=llm_heuristics)
        if llm_heuristics and symbolic:
            add_symbolic_edges(g)
        if llm_heuristics and tombstones:
            detect_supersedes(g)
            detect_rejects(g)
        infer_graph(g, llm, workers=llm_workers)
        if not tombstones:
            for n in g.nodes.values():
                n.superseded_by = n.rejected_by = None
        assign_roots(g)
        return g

    if oracle is not None:
        # perfect edge inference: structural TOOL/READS from the format + oracle edges
        g = ingest(messages, split_facts=split_facts, chain=False, serves=False)
        for nid, role in oracle.roles.items():
            if nid in g.nodes:
                g.nodes[nid].role = Role(role)
        for src, dst, s in oracle.edges:
            g.add_edge(Edge(src, dst, EdgeType.ORACLE, s, source="oracle"))
        if tombstones:
            for new, old in oracle.supersedes:
                if new in g.nodes and old in g.nodes:
                    g.nodes[old].superseded_by = new
                    g.add_edge(Edge(new, old, EdgeType.SUPERSEDES, 0.3, source="oracle"))
            for rej, prop in oracle.rejects:
                if rej in g.nodes and prop in g.nodes:
                    g.nodes[prop].rejected_by = rej
                    g.add_edge(Edge(rej, prop, EdgeType.REJECTS, 0.3, source="oracle"))
        assign_roots(g)
        return g

    g = ingest(messages, split_facts=split_facts, chain=chain, serves=True)
    if symbolic:
        add_symbolic_edges(g)
    if tombstones:
        detect_supersedes(g)
        detect_rejects(g)
    assign_roots(g)
    return g


def _versions(g: Graph, summarizer: Summarizer) -> dict[str, dict[int, str]]:
    return {nid: summarizer.versions(n) for nid, n in g.nodes.items()}


def compress(
    g: Graph,
    budget: int,
    method: str = "graded",
    *,
    summarizer: Summarizer | None = None,
    seed: int = 0,
    frame_pop: bool = True,
    cascade: bool = True,
    pin_roots: bool = True,
) -> Result:
    summarizer = summarizer or ExtractiveSummarizer()
    if isinstance(summarizer, ExtractiveSummarizer):
        summarizer.hints = conversation_hints(g.ordered())
    versions = _versions(g, summarizer)
    nodes = g.ordered()
    # baselines budget against what they will actually render (tags included)
    rcost = {n.id: {lvl: count(_plain_node(n, lvl, versions[n.id])) for lvl in LEVELS} for n in nodes}
    cost0 = {n.id: rcost[n.id][L0] for n in nodes}
    level = {n.id: L3 for n in nodes}
    dropped: set[str] = set()
    sc: dict[str, float] = {}

    if method == "full":
        level = {n.id: L0 for n in nodes}

    elif method == "truncate":
        # keep the most recent messages verbatim; drop the rest entirely
        used = 0
        for n in reversed(nodes):
            if used + cost0[n.id] <= budget:
                level[n.id] = L0
                used += cost0[n.id]
            else:
                dropped.add(n.id)

    elif method == "random":
        rng = random.Random(seed)
        order = list(nodes)
        rng.shuffle(order)
        used = 0
        for n in order:
            if used + cost0[n.id] <= budget:
                level[n.id] = L0
                used += cost0[n.id]
            else:
                dropped.add(n.id)

    elif method == "uniform":
        # same resolution for everything: the finest level that fits
        chosen = None
        for lvl in LEVELS:
            total = sum(rcost[n.id][lvl] for n in nodes)
            if total <= budget:
                chosen = lvl
                break
        if chosen is None:
            chosen = L3
            used = sum(rcost[n.id][L3] for n in nodes)
            for n in nodes:               # even stubs don't fit: drop oldest first
                if used <= budget:
                    break
                dropped.add(n.id)
                used -= rcost[n.id][L3]
        level = {n.id: chosen for n in nodes}

    elif method == "gc_binary":
        # pure GC analogue: reachable -> verbatim (most recent first), unreachable -> gone
        reach = propagate(g)
        sc = reach
        used = 0
        for n in reversed(nodes):
            if reach.get(n.id, 0.0) <= 0.0:
                dropped.add(n.id)
            elif used + cost0[n.id] <= budget:
                level[n.id] = L0
                used += cost0[n.id]
            else:
                dropped.add(n.id)

    elif method == "graded":
        sc = score_nodes(g, frame_pop=frame_pop)
        level, _ = allocate(g, sc, versions, budget, pin_roots=pin_roots, cascade=cascade)

    else:
        raise ValueError(f"unknown method {method!r}")

    # tombstones are a graded-pipeline concept; baselines render every node plainly
    text = render(g, level, versions, dropped) if method == "graded" else _render_plain(g, level, versions, dropped)
    return Result(method, text, count(text), level, dropped, sc, g)


def _plain_node(n, lvl: int, v: dict[int, str]) -> str:
    """Node rendering that ignores tombstones (baselines have no notion of them)."""
    from .render import TAG
    return v[L3] if lvl == L3 else f"[#{n.id} {n.role.value}{TAG[lvl]}] {v[lvl]}"


def _render_plain(g: Graph, level: dict[str, int], versions: dict[str, dict[int, str]], dropped: set[str]) -> str:
    return "\n".join(_plain_node(n, level.get(n.id, L3), versions[n.id]) for n in g.ordered() if n.id not in dropped)
