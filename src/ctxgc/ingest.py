"""Ingest a transcript into nodes + structural edges.

Transcript format (list of dicts):
    {"role": "user", "content": str}
    {"role": "assistant", "content": str, "tool_calls": [{"id": str, "name": str, "args": dict}]}
    {"role": "tool", "tool_call_id": str, "content": str}

Node ids are deterministic so that external ground truth can refer to them:
    m{i}        whole message i (assistant text / tool result / unsplit user message)
    m{i}.s{k}   k-th sentence ("fact") of user message i when split_facts is on
    m{i}.c{j}   j-th tool call in assistant message i

Granularity: messages are the base unit (free, exact structural edges); user
messages are split down into sentence-level facts so a constraint buried in a
long message becomes its own root; tool steps and frames are grouped up.
"""

from __future__ import annotations

import json
import re

from .model import BASE_STRENGTH, Edge, EdgeType, Graph, Node, Role

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])")
CONSTRAINT_RE = re.compile(
    r"\b(?:don'?t|do not|must not|mustn'?t|never|not allowed|no changes? to|no new\b|"
    r"without (?:changing|touching|modifying)|keep [^.]* unchanged|must\b|should not|shouldn'?t|avoid)",
    re.I,
)
DECISION_RE = re.compile(
    r"^\s*(?:root cause|decision|conclusion|diagnosis|verdict|proposal|plan|fix applied)\b\s*[:\-]",
    re.I | re.M,
)
PROPOSAL_RE = re.compile(r"^\s*proposal\b", re.I)

ROOT_STRENGTH = {
    Role.GOAL: 1.0,
    Role.CONSTRAINT: 1.0,
    Role.USER: 0.6,       # 1.0 when in the latest turn (the active ask)
    Role.CHATTER: 0.3,
}


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in SENT_SPLIT_RE.split(text.strip()) if p.strip()]
    return parts or [text.strip()]


def classify_user_sentence(sentence: str, first_turn: bool) -> Role:
    if CONSTRAINT_RE.search(sentence):
        return Role.CONSTRAINT
    return Role.GOAL if first_turn else Role.USER


def ingest(messages: list[dict], *, split_facts: bool = True, chain: bool = True, serves: bool = True) -> Graph:
    g = Graph()
    seq = 0
    turn = 0
    frame = 0
    step = 0
    pending_results: list[str] = []      # tool results not yet read by an assistant text
    last_text: str | None = None         # previous assistant text (for CHAIN)
    openers: list[str] = []              # user facts of the current turn
    call_nodes: dict[str, str] = {}      # tool_call_id -> node id
    goal_ids: list[str] = []

    for mi, m in enumerate(messages):
        role = m["role"]
        if role == "user":
            turn += 1
            frame += 1
            openers = []
            content = (m.get("content") or "").strip()
            sentences = split_sentences(content) if split_facts else [content]
            for si, s in enumerate(sentences):
                nid = f"m{mi}" if len(sentences) == 1 else f"m{mi}.s{si}"
                r = classify_user_sentence(s, first_turn=(turn == 1))
                if not split_facts and turn == 1:
                    r = Role.GOAL
                node = Node(nid, r, s, seq, turn=turn, frame=frame, msg_index=mi)
                seq += 1
                g.add_node(node)
                openers.append(nid)
                if r == Role.GOAL:
                    goal_ids.append(nid)

        elif role == "assistant":
            text = (m.get("content") or "").strip()
            text_id: str | None = None
            if text:
                r = Role.DECISION if DECISION_RE.search(text) else Role.ASSISTANT
                text_id = f"m{mi}"
                node = Node(text_id, r, text, seq, turn=turn, frame=frame, step=step or None, msg_index=mi)
                node.meta["proposal"] = bool(PROPOSAL_RE.match(text))
                seq += 1
                g.add_node(node)
                for rid in pending_results:
                    g.add_edge(Edge(text_id, rid, EdgeType.READS, BASE_STRENGTH[EdgeType.READS]))
                pending_results = []
                if chain and last_text is not None:
                    prev = g.nodes[last_text]
                    s = BASE_STRENGTH[EdgeType.CHAIN] if prev.turn == turn else 0.3
                    g.add_edge(Edge(text_id, last_text, EdgeType.CHAIN, s))
                if serves:
                    s_local = 0.9 if r == Role.DECISION else 0.7
                    s_goal = 0.9 if r == Role.DECISION else 0.4
                    for o in openers:
                        g.add_edge(Edge(o, text_id, EdgeType.SERVES, s_local))
                    for gid in goal_ids:
                        if gid not in openers:
                            g.add_edge(Edge(gid, text_id, EdgeType.SERVES, s_goal))
                last_text = text_id
                if r == Role.DECISION:
                    frame += 1   # a decision closes the current sub-task frame
            for ci, call in enumerate(m.get("tool_calls") or []):
                step += 1
                cid = f"m{mi}.c{ci}"
                args = json.dumps(call.get("args", {}), sort_keys=True, ensure_ascii=False)
                node = Node(cid, Role.TOOL_CALL, f"{call['name']}({args})", seq,
                            turn=turn, frame=frame, step=step, msg_index=mi, tool_name=call["name"])
                seq += 1
                g.add_node(node)
                call_nodes[call["id"]] = cid
                if text_id is not None:
                    g.add_edge(Edge(cid, text_id, EdgeType.CHAIN, 0.5))

        elif role == "tool":
            rid = f"m{mi}"
            cid = call_nodes.get(m.get("tool_call_id", ""))
            call = g.nodes[cid] if cid else None
            node = Node(rid, Role.TOOL_RESULT, m.get("content") or "", seq, turn=turn, frame=frame,
                        step=call.step if call else step, msg_index=mi,
                        tool_name=call.tool_name if call else None)
            seq += 1
            g.add_node(node)
            pending_results.append(rid)
            if cid:
                g.add_edge(Edge(rid, cid, EdgeType.TOOL, BASE_STRENGTH[EdgeType.TOOL]))
        else:
            raise ValueError(f"unknown role {role!r} at message {mi}")

    assign_roots(g)
    return g


def assign_roots(g: Graph) -> None:
    """Roots: every user fact. Goals/constraints at 1.0, the latest turn's asks at
    1.0, earlier plain user text at 0.6, chatter (oracle/LLM-labelled) at 0.3."""
    g.roots.clear()
    latest_turn = max((n.turn for n in g.nodes.values()), default=0)
    for n in g.nodes.values():
        if n.role in ROOT_STRENGTH:
            s = ROOT_STRENGTH[n.role]
            if n.role == Role.USER and n.turn == latest_turn:
                s = 1.0
            g.roots[n.id] = s


def reassign_frames(g: Graph) -> None:
    """Recompute frame boundaries from roles: a new frame at every user turn and
    after every decision (used after an edge-inferrer relabels roles)."""
    frame = 0
    prev_turn = None
    for n in g.ordered():
        if n.turn != prev_turn:
            frame += 1
            prev_turn = n.turn
        n.frame = frame
        if n.role == Role.DECISION:
            frame += 1
