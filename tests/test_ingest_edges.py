from ctxgc.compress import build_graph
from ctxgc.edges import detect_rejects, detect_supersedes, extract_assignments
from ctxgc.ingest import ingest
from ctxgc.model import EdgeType, Role

MESSAGES = [
    {"role": "user", "content": "Login is timing out. Please find the root cause and fix it. Do not change the database schema."},
    {"role": "assistant", "content": "Let me look at the log.",
     "tool_calls": [{"id": "c1", "name": "grep", "args": {"pattern": "LoginService"}}]},
    {"role": "tool", "tool_call_id": "c1", "content": "LoginService.loadProfile executed 50 queries\nother line 2ms"},
    {"role": "assistant", "content": "Root cause: N+1 in LoginService.loadProfile(); 50 queries per login."},
    {"role": "assistant", "content": "Proposal: add a Redis cache in front of users lookups."},
    {"role": "user", "content": "No, don't add Redis — we don't want a new dependency. Use a batched query instead."},
    {"role": "assistant", "content": "Decision: keep REQUEST_TIMEOUT = 30s for now."},
    {"role": "user", "content": "Actually, change REQUEST_TIMEOUT to 10s (was 30s)."},
    {"role": "user", "content": "Great. Now write the PR description."},
]


def test_user_messages_split_into_facts_with_roles_and_roots():
    g = ingest(MESSAGES)
    assert g.nodes["m0.s0"].role == Role.GOAL
    assert g.nodes["m0.s1"].role == Role.GOAL
    assert g.nodes["m0.s2"].role == Role.CONSTRAINT
    assert g.nodes["m5.s0"].role == Role.CONSTRAINT          # "don't add Redis"
    assert g.nodes["m5.s1"].role == Role.USER                # "Use a batched query instead."
    assert g.roots["m0.s2"] == 1.0
    assert g.roots["m5.s1"] == 0.6                           # earlier plain user text
    assert g.roots["m8.s1"] == 1.0                           # latest turn is the active ask


def test_structural_edges_tool_reads_chain_serves():
    g = ingest(MESSAGES)
    assert g.has_edge("m2", "m1.c0")                         # result -> call
    assert g.has_edge("m3", "m2")                            # decision reads the result
    assert g.has_edge("m4", "m3")                            # chain: proposal -> previous text
    assert g.has_edge("m0.s1", "m3")                         # goal serves decision
    assert g.nodes["m3"].role == Role.DECISION
    assert g.nodes["m4"].meta["proposal"] is True
    assert g.nodes["m3"].frame < g.nodes["m4"].frame         # decision closes a frame


def test_supersedes_and_rejects_heuristics():
    assert extract_assignments("Decision: keep REQUEST_TIMEOUT = 30s for now.") == [("REQUEST_TIMEOUT", "30s")]
    assert extract_assignments("Actually, change REQUEST_TIMEOUT to 10s (was 30s).") == [("REQUEST_TIMEOUT", "10s")]
    g = ingest(MESSAGES)
    assert detect_supersedes(g) == 1
    assert g.nodes["m6"].superseded_by == "m7"
    assert detect_rejects(g) == 1
    assert g.nodes["m4"].rejected_by == "m5.s0"
    assert any(e.type == EdgeType.SUPERSEDES for e in g.edges)


def test_symbolic_edges_link_shared_identifiers():
    g = build_graph(MESSAGES)
    # "LoginService.loadProfile" appears in the log (m2) and in the decision (m3)
    assert any(e.src == "m3" and e.dst == "m2" and e.type == EdgeType.REFERS for e in g.edges)
