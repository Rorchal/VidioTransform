"""Per-node resolution versions (the "mipmap"): L0 verbatim, L1 paragraph,
L2 one line, L3 stub. The default summarizer is extractive and deterministic so
that every method in the eval is compared with the same summaries; an LLM
summarizer can be plugged in (see llm.py)."""

from __future__ import annotations

import re
from typing import Protocol

from .model import L0, L1, L2, L3, Node, Role
from .tokens import count

SALIENT_RE = re.compile(
    r"\d|[A-Z]{2,}|_|\(\)|\.(?:py|js|sql|ya?ml)\b|error|fail|root cause|decision|because|must|don'?t|TODO",
    re.I,
)
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


class Summarizer(Protocol):
    def versions(self, node: Node) -> dict[int, str]: ...


def monotonic(v: dict[int, str]) -> dict[int, str]:
    """Resolutions must not grow as the level gets coarser. The L3 stub is
    exempt: it carries the node id (a stub without an id cannot be asked for),
    and it is a few words anyway."""
    for lvl in (L1, L2):
        if count(v[lvl]) > count(v[lvl - 1]):
            v[lvl] = v[lvl - 1]
    return v


def conversation_hints(nodes: list[Node], min_df: int = 2, max_df_ratio: float = 0.5) -> set[str]:
    """Identifiers mentioned in at least two prose nodes (user/assistant text):
    the things the conversation is about."""
    from .edges import identifiers
    prose = [n for n in nodes if n.role not in (Role.TOOL_RESULT, Role.TOOL_CALL)]
    df: dict[str, int] = {}
    for n in prose:
        for ident in identifiers(n.text):
            df[ident] = df.get(ident, 0) + 1
    max_df = max(min_df, int(max_df_ratio * len(prose)))
    return {i for i, c in df.items() if min_df <= c <= max_df and len(i) >= 4}


def truncate_words(text: str, n: int) -> str:
    words = text.split()
    return text if len(words) <= n else " ".join(words[:n]) + " …"


def fit_tokens(text: str, budget: int) -> str:
    if count(text) <= budget:
        return text
    return text[: max(0, budget * 4 - 2)].rstrip() + " …"


class ExtractiveSummarizer:
    """hints: identifiers the conversation itself talks about (set by the pipeline).
    Lines of a tool result that mention one are kept first - task-relative
    salience, the cheap stand-in for what an LLM summarizer would do."""

    def __init__(self, l1_tokens: int = 60, l2_tokens: int = 16, stub_words: int = 6):
        self.l1_tokens = l1_tokens
        self.l2_tokens = l2_tokens
        self.stub_words = stub_words
        self.hints: set[str] = set()

    def versions(self, node: Node) -> dict[int, str]:
        text = node.text.strip()
        if node.role == Role.TOOL_RESULT:
            v = self._tool_result(node, text)
        elif node.role == Role.TOOL_CALL:
            v = {L0: text, L1: text, L2: fit_tokens(text, self.l2_tokens),
                 L3: f"[#{node.id} call {node.tool_name or 'tool'}]"}
        else:
            v = self._prose(node, text)
        return monotonic(v)

    def _prose(self, node: Node, text: str) -> dict[int, str]:
        sentences = [s for s in SENT_SPLIT_RE.split(text) if s.strip()] or [text]
        l1_parts: list[str] = []
        used = 0
        # first sentence always, then salient sentences in order
        for i, s in enumerate(sentences):
            if i == 0 or SALIENT_RE.search(s):
                c = count(s)
                if used + c > self.l1_tokens and l1_parts:
                    break
                l1_parts.append(s)
                used += c
        l1 = " ".join(l1_parts)
        l2 = fit_tokens(sentences[0], self.l2_tokens)
        l3 = f"[#{node.id} {node.role.value}: {truncate_words(sentences[0], self.stub_words)}]"
        return {L0: text, L1: l1, L2: l2, L3: l3}

    def _tool_result(self, node: Node, text: str) -> dict[int, str]:
        lines = [ln for ln in text.splitlines() if ln.strip()] or [text]
        hinted = [ln for ln in lines if any(h in ln for h in self.hints)] if self.hints else []
        salient = hinted + [ln for ln in lines if ln not in hinted and SALIENT_RE.search(ln)] or lines
        picked: list[str] = []
        used = 0
        for ln in salient:
            c = count(ln)
            if used + c > self.l1_tokens and picked:
                break
            picked.append(fit_tokens(ln, self.l1_tokens))
            used += count(picked[-1])
        tool = node.tool_name or "tool"
        l1 = f"{tool} result ({len(lines)} lines), key lines:\n" + "\n".join(picked)
        l2 = f"{tool} → {len(lines)} lines; {fit_tokens(salient[0].strip(), self.l2_tokens)}"
        l3 = f"[#{node.id} {tool} result, {len(lines)} lines]"
        return {L0: text, L1: l1, L2: l2, L3: l3}
