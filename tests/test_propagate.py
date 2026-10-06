from ctxgc.model import Edge, EdgeType, Graph, Node, Role
from ctxgc.propagate import propagate, score


def g_with(nodes, edges, roots):
    g = Graph()
    for i, (nid, role) in enumerate(nodes):
        g.add_node(Node(nid, role, f"text {nid}", seq=i))
    for s, d, w in edges:
        g.add_edge(Edge(s, d, EdgeType.ORACLE, w))
    g.roots = dict(roots)
    return g


def test_strength_is_product_along_path_and_max_over_paths():
    g = g_with(
        [("R", Role.GOAL), ("A", Role.CONSTRAINT), ("B", Role.DECISION), ("C", Role.TOOL_RESULT),
         ("F", Role.TOOL_CALL), ("G", Role.ASSISTANT), ("D", Role.ASSISTANT)],
        [("R", "B", 0.8), ("B", "C", 0.5), ("C", "F", 0.5), ("B", "G", 0.5), ("A", "G", 0.25)],
        {"R": 1.0, "A": 1.0},
    )
    s = propagate(g)
    assert s["B"] == 0.8
    assert abs(s["C"] - 0.4) < 1e-9
    assert abs(s["F"] - 0.2) < 1e-9
    assert abs(s["G"] - 0.4) < 1e-9          # via R->B->G (0.4) beats A->G (0.25)
    assert "D" not in s                      # unreachable


def test_unreachable_gets_only_recency_floor_and_frame_pop_halves_closed_frames():
    g = g_with([("R", Role.GOAL), ("X", Role.TOOL_RESULT), ("D", Role.ASSISTANT)],
               [("R", "X", 1.0)], {"R": 1.0})
    g.nodes["X"].frame = 0
    g.nodes["D"].frame = 1
    g.nodes["R"].frame = 1
    sc = score(g, frame_pop=True, frame_decay=0.5, recency_floor=0.06)
    assert sc["R"] == 1.0
    assert abs(sc["X"] - 0.5) < 1e-9         # reachable at 1.0, halved: its frame is closed
    assert 0 < sc["D"] <= 0.06               # unreachable: floor only


def test_keep_strongest_edge_per_pair_and_type():
    g = g_with([("a", Role.GOAL), ("b", Role.ASSISTANT)], [("a", "b", 0.3), ("a", "b", 0.9), ("a", "b", 0.5)], {"a": 1.0})
    assert len(g.edges) == 1 and g.edges[0].strength == 0.9
