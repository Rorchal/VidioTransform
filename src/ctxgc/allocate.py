"""Budget-driven level allocation.

Start with every node at L3 (stub). Goals/constraints are floored at the finest
level that fits a share of the budget.
Repeatedly promote the node with the best (score x gain / token delta), pulling
its strong dependencies up to within one level of it so a verbatim conclusion
never points at an evicted premise. Tombstoned nodes are fixed. Nodes the goal
does not reach are only promoted with whatever budget is left after the
reachable ones. Stop when no promotion fits the budget.

Costs are what render() will actually emit, including the collapsing of
consecutive stubs into one line, so `used` equals the rendered size.
"""

from __future__ import annotations

from .model import L0, L1, L3, LEVELS, Graph, Node, Role
from .render import group_stub, is_stub, render_node
from .tokens import count

# marginal value of promoting FROM this level (3->2 is worth more than 1->0)
GAIN = {3: 1.0, 2: 0.6, 1: 0.4}
PINNED_ROLES = (Role.GOAL, Role.CONSTRAINT)

# How the value of a promotion is measured (prio = score x gain / token delta):
#   "level"   the fixed table above: breadth first, every node gets a one-liner
#             before anything gets its paragraph back
#   "tokens"  gain = tokens added, so prio = score: resolution follows distance
#             and nothing else (the one-line statement of the design)
#   "idents"  gain = concrete referents (identifiers) the finer version adds:
#             a file view earns its verbatim copy, chatter does not
#   "hybrid"  the level table times (1 + identifiers added): the table's
#             breadth-first shape, but a chunk dense in referents earns more
GAIN_MODES = ("level", "tokens", "idents", "hybrid")


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
    gain: str = "level",
    pin_fraction: float | None = 0.3,
    tool_l1: bool = True,
    pin_promote: str = "last",
) -> tuple[dict[str, int], int]:
    """pin_fraction: goals/constraints are guaranteed the finest uniform level
    that fits this share of the budget and stay promotable like any other node
    (a long opening request must not eat the whole budget: on real SWE-agent
    transcripts the issue text alone exceeded a 10% budget). None fixes them at
    L0 whatever they cost.
    pin_promote: how pinned nodes may rise above their floor. "last" (default):
    only after every other reachable node has had its turn - on real
    trajectories the opening request, scoring 1.0 as a root, otherwise took a
    third of a 20% budget for its own text while the tool output the agent was
    about to act on sat at one line. "always": compete like any node. "never":
    the floor is all the task statement gets. Roots drive propagation at full
    strength either way; this only concerns their own resolution.
    tool_l1: False skips the L1 (extractive paragraph) rung for tool results, so
    they are promoted straight from one line to verbatim. An extractive excerpt
    of a file view or log cannot know which name the agent will act on next; on
    real trajectories almost every identifier lost at a generous budget sat in a
    tool result parked at L1."""
    if gain not in GAIN_MODES:
        raise ValueError(f"gain must be one of {GAIN_MODES}")
    if pin_promote not in ("always", "last", "never"):
        raise ValueError("pin_promote must be always | last | never")
    order: list[Node] = g.ordered()
    pos = {n.id: i for i, n in enumerate(order)}
    cost = {nid: {lvl: count(render_node(g.nodes[nid], lvl, v)) for lvl in LEVELS} for nid, v in versions.items()}
    if gain in ("idents", "hybrid"):
        from .edges import identifiers
        n_idents = {nid: {lvl: len(identifiers(v[lvl])) for lvl in LEVELS} for nid, v in versions.items()}

    def gain_of(nid: str, lvl: int) -> float:
        if gain == "level":
            return GAIN[lvl]
        if gain == "tokens":
            return max(1, cost[nid][lvl - 1] - cost[nid][lvl])
        added = 1 + max(0, n_idents[nid][lvl - 1] - n_idents[nid][lvl])
        return GAIN[lvl] * added if gain == "hybrid" else added
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
    pinned = [nid for nid, node in g.nodes.items()
              if pin_roots and nid in g.roots and node.role in PINNED_ROLES and not node.tombstoned]
    for nid, node in g.nodes.items():
        if node.tombstoned:
            fixed.add(nid)
    if pin_fraction is None:
        for nid in pinned:
            level[nid] = L0
            fixed.add(nid)
    elif pinned:
        floor = L3
        for lvl in LEVELS:
            if sum(cost[nid][lvl] for nid in pinned) <= pin_fraction * budget:
                floor = lvl
                break
        for nid in pinned:
            level[nid] = floor
            if pin_promote == "never":
                fixed.add(nid)
    used = seg_cost(0, len(order) - 1, level)

    def next_level(nid: str) -> int:
        nxt = level[nid] - 1
        if nxt == L1 and not tool_l1 and g.nodes[nid].role == Role.TOOL_RESULT:
            nxt = L0
        return nxt

    def plan_for(nid: str) -> dict[str, int]:
        plan = {nid: next_level(nid)}
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

    def delta_of(plan: dict[str, int]) -> tuple[int, list[list[int]]]:
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
        return sum(seg_cost(lo, hi, new_level) - seg_cost(lo, hi, level) for lo, hi in merged), merged

    # A candidate's plan, token delta and priority stay valid until a promotion
    # touches a node in its plan or a stub run its delta was computed over, so
    # they are cached and only those candidates are re-evaluated. Same choices as
    # re-evaluating everything every round (the tests check that), far fewer
    # evaluations on graphs with hundreds of nodes.
    cache: dict[str, tuple] = {}

    def evaluate(nid: str) -> tuple:
        plan = plan_for(nid)
        d, segs = delta_of(plan)
        prio = score.get(nid, 0.0) * gain_of(nid, level[nid]) / max(d, 1)
        return (prio, g.nodes[nid].seq), plan, d, segs

    def promote_phase(candidates: set[str]) -> None:
        nonlocal used
        while True:
            best = None
            for nid in candidates:
                if nid in fixed or level[nid] == L0:
                    continue
                entry = cache.get(nid)
                if entry is None:
                    entry = cache[nid] = evaluate(nid)
                key, plan, d, segs = entry
                if used + d > budget:
                    continue
                if best is None or key > best[0]:
                    best = entry
            if best is None:
                return
            _, plan, d, _ = best
            changed = set(plan)
            changed_pos = {pos[n] for n in changed}
            level.update(plan)
            used += d
            for cid in [c for c, (_, cplan, _, csegs) in cache.items()
                        if changed & cplan.keys() or any(lo <= p <= hi for lo, hi in csegs for p in changed_pos)]:
                del cache[cid]

    # phase 1: nodes the goal reaches, by value density; phase 2: leftover budget
    # goes to unreachable nodes (recency floor only), most recent first
    reachable = {nid for nid in g.nodes if score.get(nid, 0.0) >= reachable_min}
    deferred = set(pinned) if (pin_promote == "last" and pin_fraction is not None) else set()
    promote_phase(reachable - deferred)
    promote_phase(deferred)
    promote_phase(set(g.nodes) - reachable - deferred)
    return level, used
