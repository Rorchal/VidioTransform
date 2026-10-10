"""BM25 retrieval baseline, the kind of baseline the LongMemEval paper runs.

Turns (or whole sessions) are ranked by BM25 against the question; the token
budget is filled in rank order, and whatever room is left goes to the most
recent unranked pieces so every method spends the same budget. The chosen
pieces are shown in chronological order, rendered exactly like the full
history. A turn retrieved without the first turn of its session gets that
session's date prepended, so date questions are not handicapped by the
granularity."""

from __future__ import annotations

import math
import re

from ctxgc.compress import compress
from ctxgc.edges import LEX_STOP, LEX_WORD_RE
from ctxgc.tokens import count

DATE_RE = re.compile(r"^(?:[^:\[\]]{1,40}: )?\[(?P<date>[^\]]+)\] ")
SPEAKER_RE = re.compile(r"^([^:\[\]]{1,40}: )(.*)$", re.S)
LINE_OVERHEAD = 6                       # "[#m123 user] " prefix, in tokens


def stem(w: str) -> str:
    """Crude suffix stripping so "graduated" meets "graduate" and "degrees"
    meets "degree"; applied to query and documents alike."""
    if w.endswith("'s"):
        w = w[:-2]
    for suf in ("ing", "ed", "ly", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[: -len(suf)]
            break
    if w.endswith("e") and len(w) >= 4:
        w = w[:-1]
    return w


def tokens(text: str) -> list[str]:
    return [stem(w) for w in LEX_WORD_RE.findall(text.lower()) if w not in LEX_STOP]


def bm25(docs: list[list[str]], query: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    n = len(docs)
    if not n:
        return []
    avg = sum(map(len, docs)) / n or 1.0
    df: dict[str, int] = {}
    for d in docs:
        for w in set(d):
            df[w] = df.get(w, 0) + 1
    out = []
    for d in docs:
        tf: dict[str, int] = {}
        for w in d:
            tf[w] = tf.get(w, 0) + 1
        s = 0.0
        for w in set(query):
            if w in tf:
                idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(d) / avg))
        out.append(s)
    return out


def _sessions(case: dict) -> list[tuple[int, int]]:
    hist = case["question_index"]
    return [tuple(s) for s in case.get("sessions") or [(0, hist)]]


def select(case: dict, budget_tokens: int, unit: str = "turn") -> list[int]:
    """Indices of the history messages to show, chronological."""
    hist = case["question_index"]
    msgs = case["messages"][:hist]
    if unit == "session":
        units = [list(range(s, e)) for s, e in _sessions(case)]
    else:
        units = [[i] for i in range(hist)]
    docs = [tokens(" ".join(msgs[i]["content"] for i in u)) for u in units]
    scores = bm25(docs, tokens(case["messages"][hist]["content"]))
    ranked = [i for i in sorted(range(len(units)), key=lambda i: -scores[i]) if scores[i] > 0]
    rest = [i for i in range(len(units) - 1, -1, -1) if scores[i] <= 0]          # most recent first
    chosen, used = [], 0
    for i in ranked + rest:
        cost = sum(count(msgs[j]["content"]) + LINE_OVERHEAD for j in units[i])
        if used + cost > budget_tokens:
            continue
        chosen.append(i)
        used += cost
    return sorted(j for i in chosen for j in units[i])


def messages(case: dict, budget_tokens: int, unit: str = "turn") -> tuple[list[dict], list[int]]:
    """The retrieved history as messages (dates carried over to orphaned turns)
    plus the indices they came from."""
    idx = select(case, budget_tokens, unit)
    first_of = {j: s for s, e in _sessions(case) for j in range(s, e)}
    picked = set(idx)
    out = []
    for j in idx:
        m = case["messages"][j]
        text = m["content"]
        s = first_of.get(j, j)
        if s != j and s not in picked and not DATE_RE.match(text):
            md = DATE_RE.match(case["messages"][s]["content"])
            if md:
                sp = SPEAKER_RE.match(text)
                text = f"{sp.group(1)}[{md.group('date')}] {sp.group(2)}" if sp else f"[{md.group('date')}] {text}"
        out.append({"role": m["role"], "content": text})
    return out, idx


def context(case: dict, budget_tokens: int, unit: str = "turn", summarizer=None) -> str:
    """The retrieved history rendered like the full history (question included
    as the last turn, as the compression methods render it)."""
    from .chat import chat_graph

    msgs, _ = messages(case, budget_tokens, unit)
    g = chat_graph(msgs + [case["messages"][case["question_index"]]], len(msgs))
    return compress(g, 10**9, method="full", summarizer=summarizer).text
