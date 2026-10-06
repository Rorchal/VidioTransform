"""Relevance propagation from roots.

Strength along a path is the product of edge strengths; a node's strength is the
max over paths (Dijkstra on -log s, done directly in strength space). Nodes not
reached have strength 0 - the GC "unreachable" case, which here means L3 stub,
not deletion.
"""

from __future__ import annotations

import heapq

from .model import Graph, Role

POPPABLE = (Role.ASSISTANT, Role.TOOL_CALL, Role.TOOL_RESULT)


def propagate(g: Graph) -> dict[str, float]:
    best: dict[str, float] = dict(g.roots)
    heap = [(-s, nid) for nid, s in g.roots.items()]
    heapq.heapify(heap)
    while heap:
        neg, nid = heapq.heappop(heap)
        s = -neg
        if s < best.get(nid, 0.0):
            continue
        for e in g.out_edges(nid):
            ns = s * e.strength
            if ns > best.get(e.dst, 0.0):
                best[e.dst] = ns
                heapq.heappush(heap, (-ns, e.dst))
    return best


def score(
    g: Graph,
    reach: dict[str, float] | None = None,
    *,
    frame_pop: bool = True,
    frame_decay: float = 0.5,
    recency_floor: float = 0.05,
) -> dict[str, float]:
    """Turn raw reachability into allocation scores.

    frame_pop: internals of closed sub-task frames (tool calls/results, interim
    assistant text) are halved - the frame's decision is its return value and keeps
    its strength. recency_floor: a weak prior so that with a generous budget the
    most recent unreachable material is still promoted before older material.
    """
    reach = propagate(g) if reach is None else reach
    nodes = g.ordered()
    cur = g.current_frame()
    out: dict[str, float] = {}
    n = len(nodes)
    for i, node in enumerate(nodes):
        s = reach.get(node.id, 0.0)
        if frame_pop and node.frame < cur and node.role in POPPABLE and node.id not in g.roots:
            s *= frame_decay
        floor = recency_floor * (i + 1) / n
        out[node.id] = max(s, floor)
    return out
