"""End-to-end on synthetic cases: ground truth is intact at full size, the graded
compressor beats the dumb baselines at a mid budget, and the pipeline's
invariants hold on real-sized graphs."""

from ctxgc.compress import build_graph, compress
from ctxgc.model import L0, L3
from evalkit.metrics import present, score_question
from evalkit.synth import make_case


def test_full_context_contains_every_answer():
    for seed in range(5):
        case = make_case(seed)
        g = build_graph(case.messages)
        r = compress(g, 10**9, method="full")
        for q in case.questions:
            assert any(present(r.text, a) for a in q.answers), (seed, q.question)
        # oracle ids all resolve to real nodes
        for nid in case.oracle.roles:
            assert nid in g.nodes, nid
        for s, d, _ in case.oracle.edges:
            assert s in g.nodes and d in g.nodes, (s, d)


def test_graded_beats_truncate_and_uniform_at_20_percent():
    kept = {"graded": 0, "graded+oracle": 0, "truncate": 0, "uniform": 0}
    total = 0
    for seed in range(8):
        case = make_case(seed)
        g = build_graph(case.messages)
        go = build_graph(case.messages, oracle=case.oracle)
        full = compress(g, 10**9, method="full").tokens
        budget = int(0.2 * full)
        runs = {
            "graded": compress(g, budget, method="graded"),
            "graded+oracle": compress(go, budget, method="graded"),
            "truncate": compress(g, budget, method="truncate"),
            "uniform": compress(g, budget, method="uniform"),
        }
        for name, r in runs.items():
            # rendered output must fit the budget (plus newline slack) for every method
            assert r.tokens <= budget + len(g.nodes) // 4 + 2, (name, r.tokens, budget)
            for q in case.questions:
                kept[name] += score_question(q, r)["kept"]
        total += len(case.questions)
    assert kept["graded"] > kept["truncate"]
    assert kept["graded"] > kept["uniform"]
    # perfect edges should not do materially worse than the model-free heuristics
    assert kept["graded+oracle"] >= kept["graded"] - 0.05 * total


def test_graded_pins_constraints_and_never_deletes():
    case = make_case(3)
    g = build_graph(case.messages)
    full = compress(g, 10**9, method="full").tokens
    r = compress(g, int(0.1 * full), method="graded")
    assert not r.dropped
    for nid, node in g.nodes.items():
        if node.role.value in ("goal", "constraint"):
            assert r.level[nid] == L0
    # every node is either rendered at some level or as a tombstone; stubs name
    # their node, literally or inside a stub id range
    from ctxgc.render import stub_names
    order = [n.id for n in g.ordered()]
    for nid, node in g.nodes.items():
        if r.level[nid] == L3 and not node.tombstoned:
            assert stub_names(r.text, nid, order)
