from ctxgc.allocate import allocate
from ctxgc.model import L0, L1, L2, L3, Edge, EdgeType, Graph, Node, Role
from ctxgc.summarize import ExtractiveSummarizer
from ctxgc.tokens import count


def build():
    g = Graph()
    long_text = "\n".join(f"2026-10-06 line {i} query=users id={1000+i} took {i % 9}ms" for i in range(80))
    g.add_node(Node("goal", Role.GOAL, "Fix the login timeout.", 0))
    g.add_node(Node("c", Role.CONSTRAINT, "Do not change the schema.", 1))
    g.add_node(Node("dec", Role.DECISION, "Root cause: N+1 in loadProfile(); 50 queries per login.", 2))
    g.add_node(Node("log", Role.TOOL_RESULT, long_text, 3, tool_name="grep"))
    g.add_node(Node("old", Role.DECISION, "Decision: keep TIMEOUT = 30s for now.", 4))
    g.add_node(Node("new", Role.USER, "Change TIMEOUT to 10s (was 30s).", 5))
    g.add_node(Node("chat", Role.ASSISTANT, "Teal works well for the dashboard.", 6))
    g.add_edge(Edge("goal", "dec", EdgeType.SERVES, 0.9))
    g.add_edge(Edge("dec", "log", EdgeType.READS, 1.0))
    g.nodes["old"].superseded_by = "new"
    g.roots = {"goal": 1.0, "c": 1.0, "new": 1.0}
    return g


def versions_for(g):
    s = ExtractiveSummarizer()
    return {nid: s.versions(n) for nid, n in g.nodes.items()}


def test_roots_pinned_budget_respected_and_graded_levels():
    g = build()
    v = versions_for(g)
    score = {"goal": 1.0, "c": 1.0, "new": 1.0, "dec": 0.9, "log": 0.9, "old": 0.3, "chat": 0.02}
    _, base = allocate(g, score, v, budget=0)          # pinned roots + stubs + tombstone, nothing promoted
    budget = base + 170
    level, used = allocate(g, score, v, budget)
    assert level["goal"] == L0 and level["c"] == L0
    assert used <= budget
    assert level["dec"] == L0                         # near the root, cheap: verbatim
    assert level["log"] in (L1, L2)                   # evidence: summarized; verbatim (80 lines) doesn't fit
    # what allocate reports is exactly what render emits (stub runs collapsed and all)
    from ctxgc.render import parts_cost, render_parts
    assert parts_cost(render_parts(g, level, v)) == used
    # phase separation: a node the goal does not reach (score below reachable_min)
    # gets nothing while a reachable node still has a promotion that fits, even
    # when the unreachable node's promotion is the cheaper one
    from ctxgc.render import render_node
    d_dec = count(render_node(g.nodes["dec"], L2, v["dec"])) - count(render_node(g.nodes["dec"], L3, v["dec"]))
    d_chat = count(render_node(g.nodes["chat"], L2, v["chat"])) - count(render_node(g.nodes["chat"], L3, v["chat"]))
    assert d_chat <= d_dec, "test setup: chatter promotion should be the cheaper one"
    first_dec = None
    prev = None
    for b in range(base, base + 80):
        lv, _ = allocate(g, score, v, b)
        if first_dec is None and lv["dec"] < L3:
            first_dec = b
            # the cheaper chatter promotion would have fit too; the reachable node won
            assert lv["chat"] == L3
        if prev is not None:   # more budget never demotes a reachable node
            assert lv["dec"] <= prev["dec"] and lv["log"] <= prev["log"] and lv["new"] <= prev["new"]
        prev = lv
    assert first_dec is not None


def test_cascade_keeps_premises_within_one_level_of_conclusions():
    g = build()
    v = versions_for(g)
    score = {"goal": 1.0, "c": 1.0, "new": 1.0, "dec": 0.95, "log": 0.2, "old": 0.3, "chat": 0.02}
    level, _ = allocate(g, score, v, budget=400, cascade=True)
    assert level["dec"] == L0
    assert level["log"] <= level["dec"] + 1           # dec -> log is a strong edge: log pulled to <= L1
    level_nc, _ = allocate(g, score, v, budget=400, cascade=False)
    assert level_nc["dec"] == L0


def test_everything_verbatim_with_a_huge_budget_and_stubs_only_with_a_tiny_one():
    g = build()
    v = versions_for(g)
    score = {nid: 0.5 for nid in g.nodes}
    level, _ = allocate(g, score, v, budget=10**6)
    assert all(level[nid] == L0 for nid, n in g.nodes.items() if not n.tombstoned)
    level, used = allocate(g, score, v, budget=1, pin_roots=False)
    assert all(level[nid] == L3 for nid, n in g.nodes.items() if not n.tombstoned)


def test_versions_are_monotone_in_tokens():
    g = build()
    for nid, vv in versions_for(g).items():
        assert count(vv[L0]) >= count(vv[L1]) >= count(vv[L2]) >= count(vv[L3]), nid
