"""Model-backed stages with a fake model: the DeepSeek JSON transport
(validation + retry), the graph-level mark phase (roles, edge direction,
tombstones, frames) and the LLM summarizer's fallbacks. No network."""

import json

import pytest

from ctxgc.compress import build_graph, compress
from ctxgc.ingest import ingest
from ctxgc.llm import (
    EDGE_SCHEMA, SUMMARY_SCHEMA, CachedLLM, DeepSeekLLM, LLMSummarizer, graph_context, infer_graph, validate,
)
from ctxgc.model import L0, L1, L2, L3, EdgeType, Role
from evalkit.metrics import edge_quality
from evalkit.synth import make_case

from .test_ingest_edges import MESSAGES


class FakeLLM:
    """Answers edge queries from a table keyed by the new chunk id; summaries
    are derived from the chunk text."""
    name = "fake"

    def __init__(self, table: dict[str, dict]):
        self.table = table
        self.calls: list[str] = []

    def json(self, system: str, user: str, schema: dict) -> dict:
        self.calls.append(user)
        if schema is EDGE_SCHEMA:
            nid = user.split("NEW CHUNK #")[1].split(" ")[0]
            return json.loads(json.dumps(self.table.get(nid, {"role": "assistant", "closes_frame": False, "edges": []})))
        chunk = user.split("CHUNK (")[1].split("):\n", 1)[1]
        return {"l1": "L1 " + chunk[:40], "l2": "L2 " + chunk[:12]}


TABLE = {
    "m0.s0": {"role": "goal", "closes_frame": False, "edges": []},
    "m0.s1": {"role": "goal", "closes_frame": False, "edges": []},
    "m0.s2": {"role": "constraint", "closes_frame": False, "edges": []},
    "m3": {"role": "decision", "closes_frame": True,
           "edges": [{"to": "#m2", "type": "derived_from", "confidence": 0.9},
                     {"to": "m0.s1", "type": "supports", "confidence": 1.0}]},
    "m4": {"role": "decision", "closes_frame": False,
           "edges": [{"to": "m3", "type": "derived_from", "confidence": 0.8},
                     {"to": "m99", "type": "mentions", "confidence": 0.9},        # unknown node: ignored
                     {"to": "m5.s0", "type": "mentions", "confidence": 0.9}]},   # later node: ignored
    "m5.s0": {"role": "constraint", "closes_frame": False,
              "edges": [{"to": "m4", "type": "rejects", "confidence": 1.0}]},
    "m5.s1": {"role": "user", "closes_frame": False, "edges": []},
    "m6": {"role": "decision", "closes_frame": True, "edges": []},
    "m7": {"role": "user", "closes_frame": False,
           "edges": [{"to": "m6", "type": "supersedes", "confidence": 0.9}]},
    "m8.s0": {"role": "chatter", "closes_frame": False, "edges": []},
    "m8.s1": {"role": "goal", "closes_frame": False,
              "edges": [{"to": "m6", "type": "derived_from", "confidence": 0.7}]},
}


def test_infer_graph_roles_edges_tombstones_frames():
    g = ingest(MESSAGES, chain=False, serves=False)
    llm = FakeLLM(TABLE)
    raw = infer_graph(g, llm, workers=2)
    # only prose nodes are sent to the model; tool call / result keep format-given edges
    assert set(raw) == {nid for nid, n in g.nodes.items() if n.role not in (Role.TOOL_CALL, Role.TOOL_RESULT)}
    assert g.has_edge("m2", "m1.c0") and g.has_edge("m3", "m2")
    # roles relabelled within the speaker's family only
    assert g.nodes["m8.s0"].role == Role.CHATTER and g.roots["m8.s0"] == 0.3
    assert g.nodes["m4"].role == Role.DECISION
    # derived_from: new -> old; supports is reversed (goal -> chunk), like SERVES
    assert any(e.src == "m4" and e.dst == "m3" and e.source == "llm" and e.strength == pytest.approx(0.8) for e in g.edges)
    assert any(e.src == "m0.s1" and e.dst == "m3" and e.type == EdgeType.SERVES for e in g.edges)
    assert not g.has_edge("m3", "m0.s1")
    # edges to unknown or later nodes are dropped
    assert not any(e.dst == "m99" for e in g.edges) and not g.has_edge("m4", "m5.s0")
    # tombstones from rejects / supersedes
    assert g.nodes["m4"].rejected_by == "m5.s0"
    assert g.nodes["m6"].superseded_by == "m7"
    # frames: new at each user turn, closed after each decision
    assert g.nodes["m3"].frame < g.nodes["m4"].frame < g.nodes["m5.s0"].frame
    assert g.nodes["m8.s1"].role == Role.GOAL and g.roots["m8.s1"] == 1.0


def test_build_graph_with_llm_and_edge_quality_on_synthetic_case():
    case = make_case(1)
    # a fake that answers like the oracle, so quality must come out perfect
    table = {}
    for nid, role in case.oracle.roles.items():
        table[nid] = {"role": role, "closes_frame": role == "decision", "edges": []}
    order = {nid: i for i, nid in enumerate(ingest(case.messages).nodes)}
    for src, dst, s in case.oracle.edges:
        if order[src] > order[dst]:       # dependency on an earlier chunk
            table[src]["edges"].append({"to": dst, "type": "derived_from", "confidence": s})
        else:                             # goal -> later decision: the decision "supports" the goal
            table[dst]["edges"].append({"to": src, "type": "supports", "confidence": s / 0.8})
    for new, old in case.oracle.supersedes:
        table.setdefault(new, {"role": "user", "closes_frame": False, "edges": []})["edges"].append(
            {"to": old, "type": "supersedes", "confidence": 1.0})
    for rej, prop in case.oracle.rejects:
        table.setdefault(rej, {"role": "constraint", "closes_frame": False, "edges": []})["edges"].append(
            {"to": prop, "type": "rejects", "confidence": 1.0})
    g = build_graph(case.messages, llm=FakeLLM(table))
    q = edge_quality(case, g)
    assert q["recall"] == 1.0 and q["precision"] == 1.0
    assert q["user_role_acc"] == 1.0 and q["supersedes_strict"] == 1.0 and q["rejects_strict"] == 1.0
    assert q["false_tombstones"] == 0
    # and the graph compresses like any other
    full = compress(g, 10**9, method="full").tokens
    r = compress(g, int(0.2 * full), method="graded")
    assert r.tokens <= int(0.2 * full) + len(g.nodes) // 4 + 2
    # llm_heuristics=True keeps the model-free edges as well
    gh = build_graph(case.messages, llm=FakeLLM(table), llm_heuristics=True)
    assert len(gh.edges) > len(g.edges)
    assert any(e.source == "symbolic" for e in gh.edges)


def test_llm_summarizer_overrides_l1_l2_and_falls_back():
    g = ingest(MESSAGES)
    ctx = graph_context(g)
    assert "Login is timing out" in ctx and "Redis" not in ctx     # opening turn only
    sm = LLMSummarizer(FakeLLM({}), context=ctx)
    n = g.nodes["m3"]
    v = sm.versions(n)
    assert v[L0] == n.text and v[L1].startswith("L1 ") and v[L2].startswith("L2 ")
    assert v[L3].startswith(f"[#{n.id}")
    # cached: second call does not hit the model
    calls = len(sm.llm.calls)
    assert sm.versions(n) is v and len(sm.llm.calls) == calls

    class Broken:
        name = "broken"

        def json(self, *a):
            raise RuntimeError("down")

    sm2 = LLMSummarizer(Broken())
    v2 = sm2.versions(n)
    assert v2[L1] and v2[L2] and sm2.failures == 1          # extractive fallback

    class TooLong:
        name = "long"

        def json(self, *a):
            return {"l1": "x" * 4000, "l2": "y" * 4000}

    v3 = LLMSummarizer(TooLong()).versions(n)
    assert len(v3[L1]) <= len(v3[L0]) and len(v3[L2]) <= len(v3[L1])   # never grows coarser-but-bigger
    assert v3[L3].startswith(f"[#{n.id}")                                # the stub always keeps its id


def test_deepseek_transport_validates_and_retries(tmp_path):
    replies = iter([
        {"choices": [{"message": {"content": "not json"}}], "usage": {"prompt_tokens": 5, "completion_tokens": 1}},
        {"choices": [{"message": {"content": json.dumps({"role": "bogus", "closes_frame": False, "edges": []})}}]},
        {"choices": [{"message": {"content": json.dumps(
            {"role": "decision", "closes_frame": True, "extra": 1,
             "edges": [{"to": "m1", "type": "derived_from", "confidence": 0.5}]})}}],
         "usage": {"prompt_tokens": 7, "completion_tokens": 3, "completion_tokens_details": {"reasoning_tokens": 2}}},
    ])
    bodies = []

    def transport(body):
        bodies.append(body)
        return next(replies)

    llm = DeepSeekLLM(effort="off", transport=transport, api_key="x")
    out = llm.json("sys", "user", EDGE_SCHEMA)
    assert out == {"role": "decision", "closes_frame": True,
                   "edges": [{"to": "m1", "type": "derived_from", "confidence": 0.5}]}   # extra key dropped
    assert len(bodies) == 3 and llm.usage["retries"] == 2 and llm.usage["reasoning_tokens"] == 2
    assert bodies[0]["response_format"] == {"type": "json_object"} and bodies[0]["thinking"] == {"type": "disabled"}
    assert "JSON schema" in bodies[0]["messages"][0]["content"]
    assert bodies[2]["messages"][-1]["role"] == "user" and "invalid" in bodies[2]["messages"][-1]["content"]

    with pytest.raises(RuntimeError):
        DeepSeekLLM(transport=lambda b: {"choices": [{"message": {"content": "{}"}}]}, api_key="x",
                    retries=2).json("s", "u", SUMMARY_SCHEMA)

    # reasoning that eats the whole max_tokens budget leaves an empty answer with
    # finish_reason "length": retry with a doubled budget, up to the cap
    seen = []

    def truncating(body):
        seen.append(body["max_tokens"])
        if body["max_tokens"] < 4000:
            return {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}
        return {"choices": [{"message": {"content": json.dumps({"l1": "a", "l2": "b"})}, "finish_reason": "stop"}]}

    llm = DeepSeekLLM(transport=truncating, api_key="x", max_tokens=1000, max_tokens_cap=8000, retries=1)
    assert llm.json("s", "u", SUMMARY_SCHEMA) == {"l1": "a", "l2": "b"}
    assert seen == [1000, 2000, 4000] and llm.usage["truncated"] == 2 and llm.usage["retries"] == 0
    with pytest.raises(RuntimeError):      # cap reached and still no answer
        DeepSeekLLM(transport=lambda b: {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]},
                    api_key="x", max_tokens=1000, max_tokens_cap=2000, retries=1).json("s", "u", SUMMARY_SCHEMA)

    # disk cache: second identical call never reaches the transport
    hits = []
    cached = CachedLLM(DeepSeekLLM(transport=lambda b: hits.append(1) or {
        "choices": [{"message": {"content": json.dumps({"l1": "a", "l2": "b"})}}]}, api_key="x"), tmp_path / "c.jsonl")
    assert cached.json("s", "u", SUMMARY_SCHEMA) == {"l1": "a", "l2": "b"}
    assert cached.json("s", "u", SUMMARY_SCHEMA) == {"l1": "a", "l2": "b"}
    assert len(hits) == 1 and cached.hits == 1 and cached.misses == 1
    assert cached.cached("s", "other", SUMMARY_SCHEMA) is None
    # a fresh wrapper over the same file sees the stored reply
    again = CachedLLM(DeepSeekLLM(transport=lambda b: 1 / 0, api_key="x"), tmp_path / "c.jsonl")
    assert again.json("s", "u", SUMMARY_SCHEMA) == {"l1": "a", "l2": "b"} and again.hits == 1


def test_validate_schema_rules():
    validate({"l1": "a", "l2": "b"}, SUMMARY_SCHEMA)
    with pytest.raises(ValueError):
        validate({"l1": "a"}, SUMMARY_SCHEMA)
    with pytest.raises(TypeError):
        validate({"role": "goal", "closes_frame": "yes", "edges": []}, EDGE_SCHEMA)
    with pytest.raises(TypeError):
        validate({"role": "goal", "closes_frame": True, "edges": [{"to": 3, "type": "mentions", "confidence": 1}]}, EDGE_SCHEMA)


def test_claude_cli_backend_parses_validates_and_accounts():
    from ctxgc.llm import ClaudeCLILLM, make_llm
    replies = iter([
        {"result": "```json\n{\"l1\": \"a\", \"l2\": \"b\"}\n```", "usage": {"input_tokens": 2, "cache_creation_input_tokens": 100,
                                                                      "cache_read_input_tokens": 5, "output_tokens": 7}, "total_cost_usd": 0.001},
    ])
    seen = []

    def runner(system, user):
        seen.append((system, user))
        return next(replies)

    llm = ClaudeCLILLM(model="haiku", runner=runner)
    assert llm.json("sys", "user text", SUMMARY_SCHEMA) == {"l1": "a", "l2": "b"}      # code fence tolerated
    assert "JSON schema" in seen[0][0] and seen[0][1].startswith("user text") and "JSON" in seen[0][1]
    assert llm.usage == {"calls": 1, "input_tokens": 107, "output_tokens": 7, "cost_usd": 0.001, "retries": 0}
    bad = ClaudeCLILLM(runner=lambda s, u: {"result": "nope"}, retries=2)
    with pytest.raises(RuntimeError):
        bad.json("s", "u", SUMMARY_SCHEMA)
    assert bad.usage["retries"] == 2
    assert make_llm("claude-cli:haiku").name == "claude-cli:haiku"


def test_coerce_json_tolerates_prose_and_maps_single_field_replies():
    from ctxgc.llm import coerce_json
    answer_schema = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}
    judge_schema = {"type": "object", "properties": {"correct": {"type": "boolean"}}, "required": ["correct"]}
    assert coerce_json('{"answer": "x"}', answer_schema) == {"answer": "x"}
    assert coerce_json('```json\n{"answer": "x"}\n```', answer_schema) == {"answer": "x"}
    assert coerce_json('Sure! Here it is: {"answer": "x"} hope that helps', answer_schema) == {"answer": "x"}
    assert coerce_json("Based on the history, Kansas City Masterpiece.", answer_schema) == {"answer": "Based on the history, Kansas City Masterpiece."}
    assert coerce_json("**Yes**, the answer matches.", judge_schema) == {"correct": True}
    assert coerce_json("No. The model said 25 minutes.", judge_schema) == {"correct": False}
    with pytest.raises(ValueError):
        coerce_json("I am not sure.", judge_schema)
    with pytest.raises(ValueError):
        coerce_json("", answer_schema)


def test_claude_cli_bare_mode_moves_instructions_into_the_user_message():
    from ctxgc.llm import ClaudeCLILLM, make_llm
    seen = []

    def runner(system, user):
        seen.append((system, user))
        return {"result": '{"l1": "a", "l2": "b"}', "usage": {}, "total_cost_usd": 0}

    llm = ClaudeCLILLM(model="haiku", runner=runner, bare=True)
    llm.json("TASK RULES", "the chunk", SUMMARY_SCHEMA)
    system, user = seen[0]
    assert system == ClaudeCLILLM.NEUTRAL_SYSTEM
    assert user.startswith("TASK RULES") and "the chunk" in user and user.rstrip().endswith("}")
    assert make_llm("claude-cli-bare:claude-haiku-4-5-20251001").name == "claude-cli-bare:claude-haiku-4-5-20251001"
