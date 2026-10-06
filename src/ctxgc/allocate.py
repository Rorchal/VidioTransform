"""Budget-driven level allocation.

Start with every node at L3 (stub). Goals/constraints are pinned at L0.
Repeatedly promote the node with the best (score x gain / token delta), pulling
its strong dependencies up to within one level of it so a verbatim conclusion
never points at an evicted premise. Tombstoned nodes are fixed. Nodes the goal
does not reach are only promoted with whatever budget is left after the
reachable ones. Stop when no promotion fits the budget.

Costs are what render() will actually emit, including the collapsing of
consecutive stubs into one line, so `used` equals the rendered size.
"""

from __future__ import annotations

from .model import L0, L3, LEVELS, Graph, Node, Role
from .render import group_stub, is_stub, render_node
from .tokens import count

# marginal value of promoting FROM this level (3->2 is worth more than 1->0)
GAIN = {3: 1.0, 2: 0.6, 1: 0.4}
PINNED_ROLES = (Role.GOAL, Role.CONSTRAINT)


def allocate(
    g: Graph,
    score: dict[str, float],
    versions: dict[str, dict[int, str]],
    budget: int,
    *,
    pin_roots: bool = True,
    cascade: bool = True,
    cascade_min_strength: float = 0.6,
    reachable_min: float = 0.1,
) -> tuple[dict[str, int], int]:
    order: list[Node] = g.ordered()
    pos = {n.id: i for i, n in enumerate(order)}
    cost = {nid: {lvl: count(render_node(g.nodes[nid], lvl, v)) for lvl in LEVELS} for nid, v in versions.items()}
    group_cache: dict[tuple[str, ...], int] = {}

    def run_cost(run: list[Node]) -> int:
        if len(run) == 1:
            return cost[run[0].id][L3]
        key = tuple(n.id for n in run)
        if key not in group_cache:
            group_cache[key] = count(group_stub(run))
        return group_cache[key]

    def seg_cost(lo: int, hi: int, lv: dict[str, int]) -> int:
        total = 0
        run: list[Node] = []
        for i in range(lo, hi + 1):
            n = order[i]
            if is_stub(n, lv[n.id]):
                run.append(n)
            else:
                if run:
                    total += run_cost(run)
                    run = []
                total += cost[n.id][lv[n.id]]
        if run:
            total += run_cost(run)
        return total

    level = {nid: L3 for nid in g.nodes}
    fixed = set()
    for nid, node in g.nodes.items():
        if node.tombstoned:
            fixed.add(nid)
        elif pin_roots and nid in g.roots and node.role in PINNED_ROLES:
            level[nid] = L0
            fixed.add(nid)
    used = seg_cost(0, len(order) - 1, level)

    def plan_for(nid: str) -> dict[str, int]:
        plan = {nid: level[nid] - 1}
        if not cascade:
            return plan
        stack = [nid]
        while stack:
            cur = stack.pop()
            need = plan[cur] + 1
            for e in g.out_edges(cur):
                if e.strength < cascade_min_strength or e.dst in fixed:
                    continue
                if plan.get(e.dst, level[e.dst]) > need:
                    plan[e.dst] = need
                    stack.append(e.dst)
        return plan

    def delta_of(plan: dict[str, int]) -> int:
        # only the stub runs touched by the plan can change cost
        segs = []
        for nid in plan:
            i = pos[nid]
            lo = hi = i
            if is_stub(order[i], level[nid]):
                while lo > 0 and is_stub(order[lo - 1], level[order[lo - 1].id]):
                    lo -= 1
                while hi < len(order) - 1 and is_stub(order[hi + 1], level[order[hi + 1].id]):
                    hi += 1
            segs.append((lo, hi))
        segs.sort()
        merged = [list(segs[0])]
        for lo, hi in segs[1:]:
            if lo <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        new_level = {**level, **plan}
        return sum(seg_cost(lo, hi, new_level) - seg_cost(lo, hi, level) for lo, hi in merged)

    def promote_phase(candidates: set[str]) -> None:
        nonlocal used
        while True:
            best = None
            for nid in candidates:
                if nid in fixed or level[nid] == L0:
                    continue
                plan = plan_for(nid)
                d = delta_of(plan)
                if used + d > budget:
                    continue
                prio = score.get(nid, 0.0) * GAIN[level[nid]] / max(d, 1)
                key = (prio, g.nodes[nid].seq)
                if best is None or key > best[0]:
                    best = (key, plan, d)
            if best is None:
                return
            _, plan, d = best
            level.update(plan)
            used += d

    # phase 1: nodes the goal reaches, by value density; phase 2: leftover budget
    # goes to unreachable nodes (recency floor only), most recent first
    reachable = {nid for nid in g.nodes if score.get(nid, 0.0) >= reachable_min}
    promote_phase(reachable)
    promote_phase(set(g.nodes) - reachable)
    return level, used
