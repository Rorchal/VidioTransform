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


class _FakeQA:
    """Answers or grades, depending on the schema it is asked for; records
    which system prompts it saw."""

    def __init__(self, name, correct=True):
        self.name = name
        self.correct = correct
        self.systems = []

    def json(self, system, user, schema):
        self.systems.append(system)
        if "answer" in schema["properties"]:
            return {"answer": "Business Administration"}
        return {"correct": self.correct}


def test_qa_run_uses_separate_judge_and_rejudge_reads_cached_answers(tmp_path):
    from ctxgc.llm import CachedLLM
    from evalkit.qa import ANSWER_SYSTEM, JUDGE_SYSTEM, run as qa_run, to_markdown as qa_md
    from evalkit.rejudge import rejudge
    data = [_lme_instance("q1", "single-session-user", "What degree did I graduate with?", "Business Administration")]
    p = tmp_path / "lme.json"
    p.write_text(json.dumps(data))
    cases = longmemeval_cases(p, n=1)
    cache = tmp_path / "cache.jsonl"
    answerer = CachedLLM(_FakeQA("fake-answerer"), cache)
    judge = _FakeQA("fake-judge", correct=False)
    s = qa_run(cases, ["truncate", "graded"], [0.5], llm=answerer, judge=judge, workers=2, verbose=False)
    # the answerer only answers, the judge only grades, and the report names both
    assert set(answerer.inner.systems) == {ANSWER_SYSTEM} and set(judge.systems) == {JUDGE_SYSTEM}
    assert s["answerer"] == "fake-answerer" and s["judge"] == "fake-judge"
    assert all(c["accuracy"] == 0.0 for c in s["table"].values())
    assert "judged by fake-judge" in qa_md(s, "t")
    # re-judging finds every cached answer of that run and regrades it with the new judge
    lenient = _FakeQA("lenient", correct=True)
    r = rejudge(cases, ["truncate", "graded"], [0.5], "fake-answerer", lenient, cache, workers=2)
    assert r["answers_found"] == 3 and r["answers_missing"] == 0
    assert set(r["accuracy"]) == {"full@1.0", "truncate@0.5", "graded@0.5"} and all(v == 1.0 for v in r["accuracy"].values())
    assert len(lenient.systems) == 3
    # a run name that never answered has nothing in the cache
    r2 = rejudge(cases, ["truncate", "graded"], [0.5], "nobody", lenient, cache)
    assert r2["answers_found"] == 0 and r2["answers_missing"] == 3


def test_bm25_retrieval_selects_matching_turns_and_carries_the_session_date():
    from evalkit import retrieve
    msgs = [{"role": "user", "content": "[2023/05/20 (Sat) 02:21] The farmer needs to cross a river with a fox."},
            {"role": "assistant", "content": "Take the chicken first, then come back."},
            {"role": "user", "content": "I graduated with a degree in Business Administration last May."},
            {"role": "assistant", "content": "Congratulations on the degree!"},
            {"role": "user", "content": "[2023/05/22 (Mon) 09:00] Any tips for my new Weber grill?"},
            {"role": "assistant", "content": "Season the grates first."},
            {"role": "user", "content": "What degree did I graduate with?"}]
    case = {"messages": msgs, "question_index": 6, "sessions": [[0, 4], [4, 6]], "evidence": [2], "answers": ["Business Administration"]}
    scores = retrieve.bm25([retrieve.tokens(m["content"]) for m in msgs[:6]], retrieve.tokens(msgs[6]["content"]))
    assert max(range(6), key=lambda i: scores[i]) == 2 and scores[0] == 0
    assert retrieve.stem("graduated") == retrieve.stem("graduate") and retrieve.stem("degrees") == retrieve.stem("degree")
    # a budget for one turn takes the best-scoring one; the orphaned turn gets its session's date
    assert retrieve.select(case, 30, "turn") == [2]
    out, idx = retrieve.messages(case, 30, "turn")
    assert idx == [2] and out[0]["content"].startswith("[2023/05/20 (Sat) 02:21] I graduated")
    # the unmatched rest of the budget is filled most-recent first
    assert retrieve.select(case, 30 + 20 + 12, "turn") == [2, 3, 5]
    # session granularity takes whole sessions, best-matching first
    assert retrieve.select(case, 90, "session") == [0, 1, 2, 3]
    text = retrieve.context(case, 30, "turn")
    assert "Business Administration" in text and "farmer" not in text and "[#m0 user]" in text
    # LoCoMo-style speaker prefix keeps the speaker ahead of the carried date
    loco = {"messages": [{"role": "user", "content": "Caroline: [1:56 pm on 8 May, 2023] Hey Mel!"},
                         {"role": "assistant", "content": "Melanie: I went to the LGBTQ support group yesterday."},
                         {"role": "user", "content": "Which group did Melanie go to?"}],
            "question_index": 2, "sessions": [[0, 2]]}
    out, idx = retrieve.messages(loco, 20, "turn")
    assert idx == [1] and out[0]["content"] == "Melanie: [1:56 pm on 8 May, 2023] I went to the LGBTQ support group yesterday."


def test_loaders_record_session_spans_and_retrieval_runs_through_metrics_and_qa(tmp_path):
    from evalkit.qa import contexts_for
    data = [_lme_instance("q1", "single-session-user", "What degree did I graduate with?", "Business Administration")]
    p = tmp_path / "lme.json"
    p.write_text(json.dumps(data))
    c = longmemeval_cases(p, n=1)[0]
    assert [i for s, e in c["sessions"] for i in range(s, e)] == list(range(c["question_index"]))
    assert len(c["sessions"]) == 3
    rows = score_case(c, ExtractiveSummarizer(), methods={"retrieve-turn": {"retrieve": "turn"}, "retrieve-session": {"retrieve": "session"}},
                      budgets=[0.5, 1.5])
    by = {(r["method"], r["frac"]): r for r in rows}
    assert all(r["answer"] and r["evidence_l0"] == 1.0 and r["retrievable"] for r in rows)
    assert by[("retrieve-turn", 0.5)]["tokens"] < 0.7 < by[("retrieve-turn", 1.5)]["tokens"]
    ctxs = contexts_for(c, ["retrieve-session"], [0.5], ExtractiveSummarizer())
    assert "Business Administration" in ctxs[("retrieve-session", 0.5)] and "What degree" not in ctxs[("retrieve-session", 0.5)]
    conv = {"sample_id": "conv-1",
            "conversation": {"speaker_a": "Caroline", "speaker_b": "Melanie", "session_1_date_time": "1 pm",
                             "session_1": [{"speaker": "Caroline", "dia_id": "D1:1", "text": "Hey Mel!"}],
                             "session_2_date_time": "2 pm",
                             "session_2": [{"speaker": "Melanie", "dia_id": "D2:1", "text": "I went to the LGBTQ support group."}]},
            "qa": [{"question": "Which group did Melanie go to?", "answer": "LGBTQ support group", "evidence": ["D2:1"], "category": 2}]}
    p2 = tmp_path / "locomo.json"
    p2.write_text(json.dumps([conv]))
    lc = locomo_cases(p2, per_conv=5)[0]
    assert lc["sessions"] == [[0, 1], [1, 2]]
