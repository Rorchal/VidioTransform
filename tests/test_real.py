"""Real-trajectory eval pieces: the two transcript converters, the label-free
needs (what the agent refers to later), cut points, and the allocator options
the agent profile relies on (gain modes, budget-aware pins, compact stubs)."""

import json

import pytest

from ctxgc.allocate import allocate
from ctxgc.compress import build_graph, compress
from ctxgc.model import L0, L1, L2, L3, Role
from ctxgc.render import group_stub
from ctxgc.summarize import ExtractiveSummarizer
from evalkit.real import (
    agent_text, constraints_in, cut_points, from_claude_code_jsonl, from_swe_agent, goal_needs, needs, run,
)

SWE_TRAJ = [
    {"role": "system", "text": "", "system_prompt": "SETTING: ..."},
    {"role": "user", "text": "We're currently solving the following issue:\nISSUE:\nLoad system default SSL cert in sendmail_client.py\nINSTRUCTIONS: fix it."},
    {"role": "ai", "text": "Let's list files.\n```\nls -F\n```"},
    {"role": "user", "text": "README.md\nmailmerge/\n\n(Open file: n/a)\n(Current directory: /mailmerge)\nbash-$"},
    {"role": "ai", "text": "Open the client.\n```\nopen mailmerge/sendmail_client.py\n```"},
    {"role": "user", "text": "[File: /mailmerge/mailmerge/sendmail_client.py (168 lines total)]\n1:import smtplib\n40:    def sendmail_ssltls(self):\n41:        ctx = ssl.SSLContext()\n\n(Open file: /mailmerge/mailmerge/sendmail_client.py)\n(Current directory: /mailmerge)\nbash-$"},
    {"role": "ai", "text": "Edit it.\n```\nedit 41:41\n        ctx = ssl.create_default_context()\nend_of_edit\n```"},
    {"role": "user", "text": "[File: /mailmerge/mailmerge/sendmail_client.py (168 lines total)]\n41:        ctx = ssl.create_default_context()\n\n(Open file: x)\n(Current directory: /mailmerge)\nbash-$"},
    {"role": "ai", "text": "Done with sendmail_ssltls.\n```\nsubmit\n```"},
]


def test_swe_agent_converter_pairs_commands_with_observations():
    msgs = from_swe_agent(SWE_TRAJ)
    roles = [m["role"] for m in msgs]
    assert roles == ["user", "assistant", "tool", "assistant", "tool", "assistant", "tool", "assistant"]
    assert msgs[0]["content"].startswith("We're currently solving")
    call = msgs[1]["tool_calls"][0]
    assert call["name"] == "ls" and call["args"]["command"] == "ls -F" and msgs[1]["content"] == "Let's list files."
    assert msgs[2]["tool_call_id"] == call["id"]
    assert "bash-$" not in msgs[2]["content"] and "(Open file" not in msgs[2]["content"]   # interface trailer stripped
    assert msgs[5]["tool_calls"][0]["name"] == "edit" and "end_of_edit" in msgs[5]["tool_calls"][0]["args"]["command"]
    # the whole thing ingests with format-given edges intact
    g = build_graph(msgs)
    assert any(n.role == Role.TOOL_RESULT for n in g.nodes.values())


def test_claude_code_converter_merges_streamed_blocks(tmp_path):
    events = [
        {"type": "user", "message": {"role": "user", "content": "Fix the bug. <system-reminder>noise</system-reminder>"}},
        {"type": "assistant", "message": {"id": "msg1", "role": "assistant", "content": [{"type": "thinking", "thinking": "hmm"}]}},
        {"type": "assistant", "message": {"id": "msg1", "role": "assistant", "content": [{"type": "text", "text": "Looking."}]}},
        {"type": "assistant", "message": {"id": "msg1", "role": "assistant",
                                          "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}}]}},
        {"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1",
                                                                  "content": [{"type": "text", "text": "a.py\nb.py"}]}]}},
        {"type": "attachment", "attachment": {}},
        {"type": "assistant", "isSidechain": True, "message": {"id": "side", "role": "assistant", "content": [{"type": "text", "text": "ignored"}]}},
        {"type": "assistant", "message": {"id": "msg2", "role": "assistant", "content": [{"type": "text", "text": "Found it in a.py."}]}},
        {"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": "Great, don't touch b.py."}]}},
    ]
    path = tmp_path / "s.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in events) + "\nnot json\n")
    msgs = from_claude_code_jsonl(path)
    assert [m["role"] for m in msgs] == ["user", "assistant", "tool", "assistant", "user"]
    assert msgs[0]["content"] == "Fix the bug."                       # reminder stripped
    assert msgs[1]["content"] == "Looking." and msgs[1]["tool_calls"] == [{"id": "t1", "name": "Bash", "args": {"command": "ls"}}]
    assert msgs[2] == {"role": "tool", "tool_call_id": "t1", "content": "a.py\nb.py"}
    assert "ignored" not in json.dumps(msgs)
    assert msgs[4]["content"] == "Great, don't touch b.py."


def test_needs_cuts_and_goal_identifiers():
    msgs = from_swe_agent(SWE_TRAJ)
    # cut right after the second tool result (prefix = first 5 messages)
    cut = 5
    g = build_graph(msgs[:cut])
    carried, reread = needs(g, msgs, cut)
    # the agent later edits sendmail_ssltls (seen in the file view) and names the path in a command
    assert "sendmail_ssltls" in carried
    assert "sendmail_client.py" not in carried          # in the opening request: excluded from carried
    assert all(nid in g.nodes for nids in carried.values() for nid in nids)
    assert reread == set()                             # no later command names a carried path
    assert "ssl.create_default_context" not in carried  # appears only after the cut
    assert goal_needs(msgs, cut) == set()              # after the cut the agent never names the file again
    # one step earlier the agent still opens the file named in the issue
    # (paths compare by basename: "mailmerge/sendmail_client.py" is the same referent)
    assert "sendmail_client.py" in goal_needs(msgs, 3)
    g3 = build_graph(msgs[:3])
    assert needs(g3, msgs, 3)[0] == {}                 # nothing discovered yet that is reused later
    # agent_text sees raw command text, not JSON escapes
    assert "nend_of_edit" not in agent_text(msgs[5]) and "end_of_edit" in agent_text(msgs[5])
    assert constraints_in(msgs, cut) == []
    # cut points need enough steps; this toy trajectory has 3
    assert cut_points(msgs) == []
    assert cut_points(msgs, min_prefix_steps=1) == [3, 5]


def test_run_on_toy_trajectory_full_keeps_everything():
    msgs = from_swe_agent(SWE_TRAJ)
    import evalkit.real as real
    old = real.cut_points
    real.cut_points = lambda m, **kw: old(m, min_prefix_steps=1)
    try:
        s = run([("toy", msgs)], methods={"full": {"method": "full"}, "graded": {"method": "graded"}}, budgets=[0.5, 1.0])
    finally:
        real.cut_points = old
    assert s["n_cuts"] == 1                            # the cut at 3 has no carried identifiers and is skipped
    assert s["table"]["full@1.0"]["carried"] == 1.0 and s["table"]["full@0.5"]["carried"] == 1.0
    assert s["table"]["graded@1.0"]["retrievable"] == 1.0


# --------------------------------------------------------------------------- allocator options

def _graph():
    return build_graph([
        {"role": "user", "content": "Fix the login timeout. " + "The issue text goes on and on about many unrelated things. " * 20},
        {"role": "assistant", "content": "Let me read the file.",
         "tool_calls": [{"id": "c1", "name": "read_file", "args": {"path": "src/login_service.py"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "\n".join(f"    def helper_{i}(self, payload_{i}):\n        return self.db.query_{i}(payload_{i})" for i in range(30))},
        {"role": "assistant", "content": "Now I will edit helper_7 and query_7."},
    ])


def test_pin_fraction_bounds_the_opening_request():
    g = _graph()
    sm = ExtractiveSummarizer()
    v = {n.id: sm.versions(n) for n in g.nodes.values()}
    score = {nid: 1.0 for nid in g.nodes}
    pinned = [nid for nid, n in g.nodes.items() if n.role == Role.GOAL]
    from ctxgc.tokens import count
    pin_cost_l0 = sum(count(v[nid][L0]) for nid in pinned)
    budget = pin_cost_l0 // 2
    level_old, used_old = allocate(g, score, v, budget, pin_fraction=None)
    assert all(level_old[nid] == L0 for nid in pinned) and used_old > budget     # old rule: pins blow the budget
    level, used = allocate(g, score, v, budget, pin_fraction=0.3)
    assert used <= budget
    assert any(level[nid] > L0 for nid in pinned)                                   # floored below L0 ...
    assert all(level[nid] <= L2 for nid in pinned) or sum(count(v[n][L2]) for n in pinned) > 0.3 * budget
    big, _ = allocate(g, score, v, 10**6, pin_fraction=0.3)
    assert all(big[nid] == L0 for nid in pinned)                                    # ... but promotable


def test_gain_modes_differ_and_idents_favours_the_file_view():
    g = _graph()
    sm = ExtractiveSummarizer()
    v = {n.id: sm.versions(n) for n in g.nodes.values()}
    score = {nid: 0.5 for nid in g.nodes}
    file_node = next(nid for nid, n in g.nodes.items() if n.role == Role.TOOL_RESULT)
    full = sum(len(n.text) // 4 for n in g.nodes.values())
    budget = int(0.5 * full)
    lv_level, _ = allocate(g, score, v, budget, gain="level", pin_fraction=0.3)
    lv_idents, _ = allocate(g, score, v, budget, gain="idents", pin_fraction=0.3)
    assert lv_idents[file_node] <= lv_level[file_node]
    with pytest.raises(ValueError):
        allocate(g, score, v, budget, gain="bogus")
    for mode in ("tokens", "hybrid"):
        lv, used = allocate(g, score, v, budget, gain=mode, pin_fraction=0.3)
        assert used <= budget and set(lv) == set(g.nodes)


def test_group_stub_is_compact_and_keeps_every_id():
    g = _graph()
    nodes = g.ordered()
    text = group_stub(nodes)
    for n in nodes:
        assert f"#{n.id}" in text
    assert "read_file result" in text and "goal" in text
    assert "payload" not in text                   # no content previews
    assert len(text) < 40 * len(nodes)


def test_compress_default_profile_and_agent_profile_fit_the_budget():
    g = _graph()
    full = compress(g, 10**9, method="full").tokens
    for kw in ({}, {"gain": "idents"}, {"pin_fraction": None}):
        r = compress(g, int(0.2 * full), method="graded", **kw)
        assert r.tokens <= int(0.2 * full) + len(g.nodes) // 4 + 2 or kw.get("pin_fraction", 0.3) is None
        assert not r.dropped


def test_pin_promote_policies():
    g = _graph()
    sm = ExtractiveSummarizer()
    v = {n.id: sm.versions(n) for n in g.nodes.values()}
    score = {nid: (1.0 if n.role == Role.GOAL else 0.5) for nid, n in g.nodes.items()}
    pinned = [nid for nid, n in g.nodes.items() if n.role == Role.GOAL]
    file_node = next(nid for nid, n in g.nodes.items() if n.role == Role.TOOL_RESULT)
    from ctxgc.tokens import count
    budget = int(0.5 * sum(count(vv[L0]) for vv in v.values()))
    always, _ = allocate(g, score, v, budget, gain="idents", pin_promote="always")
    last, _ = allocate(g, score, v, budget, gain="idents", pin_promote="last")
    never, _ = allocate(g, score, v, budget, gain="idents", pin_promote="never")
    # deferring the pins leaves at least as much budget for the file view
    assert last[file_node] <= always[file_node]
    assert sum(last[n] for n in pinned) >= sum(always[n] for n in pinned)
    # "never" holds the pins at their floor even with a huge budget
    huge, _ = allocate(g, score, v, 10**6, gain="idents", pin_promote="never")
    floor_lvls = {huge[n] for n in pinned}
    assert floor_lvls == {L0}          # the floor itself is L0 when 30% of the budget fits the pins
    with pytest.raises(ValueError):
        allocate(g, score, v, budget, pin_promote="sometimes")


def test_llm_summarizer_role_and_size_filters():
    from ctxgc.llm import LLMSummarizer

    class Counting:
        name = "counting"

        def __init__(self):
            self.calls = 0

        def json(self, system, user, schema):
            self.calls += 1
            return {"l1": "model l1", "l2": "model l2"}

    g = _graph()
    llm = Counting()
    sm = LLMSummarizer(llm, roles=(Role.TOOL_RESULT,), min_tokens=80)
    sm.prefetch(g.ordered(), workers=2)
    big_results = [n for n in g.nodes.values() if n.role == Role.TOOL_RESULT and len(n.text) // 4 >= 80]
    assert llm.calls == len(big_results) >= 1
    assert sm.skipped == len(g.nodes) - len(big_results)
    for n in g.nodes.values():
        v = sm.versions(n)
        assert (v[L1] == "model l1") == (n in big_results)
