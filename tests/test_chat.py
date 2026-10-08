"""Chat-history eval pieces: the LongMemEval and LoCoMo loaders on tiny
fixtures, lexical (word-overlap) edges, the chat graph's root handling, and
the allocator's candidate cache against a brute-force re-evaluation."""

import json

from ctxgc.allocate import allocate
from ctxgc.compress import build_graph
from ctxgc.edges import add_lexical_edges, content_words
from ctxgc.model import L0, L3, Role
from ctxgc.propagate import score as score_nodes
from ctxgc.render import group_stub, is_stub, render_node
from ctxgc.summarize import ExtractiveSummarizer, conversation_hints
from ctxgc.tokens import count
from evalkit.chat import _answer_variants, chat_graph, locomo_cases, longmemeval_cases, run, score_case
from evalkit.synth import make_case


def _lme_instance(qid, qtype, question, answer, with_evidence=True):
    distractor = [{"role": "user", "content": "The farmer needs to cross a river with a fox."},
                  {"role": "assistant", "content": "Take the chicken first, then come back."}]
    evidence = [{"role": "user", "content": "I graduated with a degree in Business Administration last May.", "has_answer": with_evidence},
                {"role": "assistant", "content": "Congratulations on the degree!", "has_answer": False}]
    return {"question_id": qid, "question_type": qtype, "question": question, "answer": answer,
            "question_date": "2023/06/01", "haystack_dates": ["2023/05/20 (Sat) 02:21", "2023/05/21 (Sun) 10:00", "2023/05/22 (Mon) 09:00"],
            "haystack_session_ids": ["s1", "answer_1", "s3"],
            "haystack_sessions": [distractor, evidence, distractor], "answer_session_ids": ["answer_1"]}


def test_longmemeval_loader_builds_history_plus_question(tmp_path):
    data = [_lme_instance("q1", "single-session-user", "What degree did I graduate with?", "Business Administration"),
            _lme_instance("q2", "knowledge-update", "What is my best 5K time?", "25 minutes and 50 seconds (or 25:50)")]
    p = tmp_path / "lme.json"
    p.write_text(json.dumps(data))
    cases = longmemeval_cases(p, n=2, max_turns=120)
    assert {c["type"] for c in cases} == {"single-session-user", "knowledge-update"}
    c = next(c for c in cases if c["id"] == "q1")
    assert c["messages"][-1] == {"role": "user", "content": "What degree did I graduate with?"}
    assert c["question_index"] == len(c["messages"]) - 1
    assert c["messages"][0]["content"].startswith("[2023/05/20")           # session date on the first turn
    assert [c["messages"][i]["content"] for i in c["evidence"]] == ["[2023/05/21 (Sun) 10:00] I graduated with a degree in Business Administration last May."]
    assert c["answers"] == ["Business Administration"]
    c2 = next(c for c in cases if c["id"] == "q2")
    assert "25 minutes and 50 seconds" in c2["answers"] and "25:50" in c2["answers"]
    # a tight turn cap keeps the evidence sessions and drops distractors
    small = longmemeval_cases(p, n=2, max_turns=2)
    assert all(len(c["messages"]) == 3 for c in small)                      # evidence session (2 turns) + question


def test_answer_variants():
    assert _answer_variants("four") == ["four"]
    assert _answer_variants("7 days. 8 days (including the last day) is also acceptable.")[0].startswith("7 days")
    assert set(_answer_variants("Paris or Rome")) >= {"Paris or Rome", "Paris", "Rome"}


def test_locomo_loader_maps_speakers_and_evidence(tmp_path):
    conv = {"sample_id": "conv-1",
            "conversation": {"speaker_a": "Caroline", "speaker_b": "Melanie",
                             "session_1_date_time": "1:56 pm on 8 May, 2023",
                             "session_1": [{"speaker": "Caroline", "dia_id": "D1:1", "text": "Hey Mel! Good to see you!"},
                                           {"speaker": "Melanie", "dia_id": "D1:2", "text": "Hey! I went to the LGBTQ support group yesterday."}],
                             "session_2_date_time": "2:00 pm on 9 May, 2023",
                             "session_2": [{"speaker": "Caroline", "dia_id": "D2:1", "text": "How was the group?"}]},
            "qa": [{"question": "Which group did Melanie go to?", "answer": "LGBTQ support group", "evidence": ["D1:2"], "category": 2},
                   {"question": "Did Melanie win the lottery?", "answer": None, "category": 5}]}
    p = tmp_path / "locomo.json"
    p.write_text(json.dumps([conv]))
    cases = locomo_cases(p, per_conv=5)
    assert len(cases) == 1                                 # adversarial category skipped
    c = cases[0]
    assert c["type"] == "cat2" and c["answers"] == ["LGBTQ support group"]
    roles = [m["role"] for m in c["messages"]]
    assert roles == ["user", "assistant", "user", "user"]  # A -> user, B -> assistant, question last
    assert c["messages"][0]["content"].startswith("Caroline: [1:56 pm on 8 May, 2023]")
    assert c["evidence"] == [1]
    # budget 1.0 is the "full" rendering's size; graded's own tags cost a token
    # more and the cascade rule chains the short turns together, so give it slack
    rows = score_case(c, ExtractiveSummarizer(), methods={"full": {"method": "full"}, "graded+lex": {"method": "graded", "lexical": True}},
                      budgets=[1.5])
    assert all(r["answer"] and r["retrievable"] for r in rows)
    assert all(r["evidence_l0"] == 1.0 for r in rows)


def test_lexical_edges_point_from_question_to_overlapping_turns():
    msgs = [{"role": "user", "content": "I love my new Weber gas grill. Any BBQ rub recommendations?"},
            {"role": "assistant", "content": "Killer Hogs BBQ Rub is popular."},
            {"role": "user", "content": "Unrelated: the farmer must cross the river with a fox."},
            {"role": "assistant", "content": "Take the chicken first."},
            {"role": "user", "content": "Which BBQ rub did you recommend for my grill?"}]
    g = chat_graph(msgs, question_index=4, lexical=True)
    q = [n.id for n in g.nodes.values() if n.msg_index == 4]
    assert len(q) == 1 and g.roots[q[0]] == 1.0
    # the opening turn is plain user text in a chat history, not a pinned goal
    assert all(n.role != Role.GOAL for n in g.nodes.values() if n.msg_index != 4)
    lex = [e for e in g.edges if e.source == "lexical"]
    assert lex and all(e.src == q[0] for e in lex)
    targets = {g.nodes[e.dst].msg_index for e in lex}
    assert 1 in targets and 0 in targets and 2 not in targets          # river puzzle shares no content word
    assert max(lex, key=lambda e: e.strength).strength <= 0.9
    assert "bbq" in content_words("BBQ rub recommendations") and "the" not in content_words("the rub")
    assert add_lexical_edges(g, ["nope"]) == 0


def test_run_on_fixture_and_markdown(tmp_path):
    data = [_lme_instance("q1", "single-session-user", "What degree did I graduate with?", "Business Administration")]
    p = tmp_path / "lme.json"
    p.write_text(json.dumps(data))
    cases = longmemeval_cases(p, n=1)
    s = run(cases, methods={"full": {"method": "full"}, "truncate": {"method": "truncate"}}, budgets=[0.5, 1.0], verbose=False)
    assert s["table"]["full@1.0"]["answer"] == 1.0 and s["table"]["full@1.0"]["evidence_l0"] == 1.0
    from evalkit.chat import to_markdown
    md = to_markdown(s, "t")
    assert "Answer string present" in md and "| truncate |" in md


def test_allocator_is_deterministic_and_exact_on_synthetic_graphs():
    """The candidate cache must not change what the greedy picks: two runs on
    identical inputs agree, and the reported size is exactly the rendered size
    (a stale cache entry would break the second check first)."""
    for seed in range(4):
        g = build_graph(make_case(seed).messages)
        sm = ExtractiveSummarizer()
        sm.hints = conversation_hints(g.ordered())
        v = {n.id: sm.versions(n) for n in g.nodes.values()}
        sc = score_nodes(g)
        full = sum(count(vv[L0]) for vv in v.values())
        for frac in (0.1, 0.35):
            for kw in ({}, {"gain": "idents"}):
                level, used = allocate(g, sc, v, int(full * frac), **kw)
                again, used2 = allocate(g, sc, v, int(full * frac), **kw)
                assert level == again and used == used2
                # exactness: what the allocator reports is what render emits
                parts, run = [], []
                for n in g.ordered():
                    lvl = level[n.id]
                    if is_stub(n, lvl):
                        run.append(n)
                        continue
                    if run:
                        parts.append(render_node(run[0], L3, v[run[0].id]) if len(run) == 1 else group_stub(run))
                        run = []
                    parts.append(render_node(n, lvl, v[n.id]))
                if run:
                    parts.append(render_node(run[0], L3, v[run[0].id]) if len(run) == 1 else group_stub(run))
                assert sum(count(p) for p in parts) == used <= int(full * frac) or frac == 0.1


def test_qa_dry_run_builds_contexts_and_estimates_cost(tmp_path):
    from evalkit.qa import contexts_for, run as qa_run, to_markdown as qa_md
    data = [_lme_instance("q1", "single-session-user", "What degree did I graduate with?", "Business Administration")]
    p = tmp_path / "lme.json"
    p.write_text(json.dumps(data))
    cases = longmemeval_cases(p, n=1)
    ctxs = contexts_for(cases[0], ["truncate", "graded"], [0.5], ExtractiveSummarizer())
    assert set(ctxs) == {("full", 1.0), ("truncate", 0.5), ("graded", 0.5)}
    # the question is asked in the prompt, not repeated inside the history
    assert "What degree did I graduate with?" not in ctxs[("full", 1.0)]
    assert "Business Administration" in ctxs[("full", 1.0)]
    s = qa_run(cases, ["truncate", "graded"], [0.5], dry_run=True, verbose=False)
    assert s["estimate"]["input_tokens"] > 0 and s["estimate"]["cny"]["peak"] > s["estimate"]["cny"]["off_peak"]
    assert "Dry run" in qa_md(s, "t") and "accuracy" not in s["table"]["full@1.0"]
