"""Data model: nodes (chunks), edges (relevance flow), graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Role(str, Enum):
    GOAL = "goal"              # what the user asked for; always a root
    CONSTRAINT = "constraint"  # hard rule from the user; always a root
    USER = "user"              # other user text
    CHATTER = "chatter"        # off-task user text (only known via oracle/LLM)
    DECISION = "decision"      # assistant conclusion / proposal; a frame's return value
    ASSISTANT = "assistant"    # other assistant text
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


class EdgeType(str, Enum):
    # Edge src -> dst means: "if src matters, dst matters x strength"
    # (equivalently: src depends on dst). Relevance flows from roots along edges.
    TOOL = "tool"              # tool_result -> tool_call that produced it
    READS = "reads"            # assistant text -> tool results it just read
    CHAIN = "chain"            # assistant text -> previous assistant text
    SERVES = "serves"          # user ask -> assistant text answering it
    REFERS = "refers"          # later node -> earlier node sharing an identifier
    MENTIONS = "mentions"      # earlier node -> later node sharing an identifier
    SUPERSEDES = "supersedes"  # new value -> old value (old becomes a tombstone)
    REJECTS = "rejects"        # rejection -> proposal (proposal becomes a tombstone)
    ORACLE = "oracle"          # ground-truth edge supplied by synthetic data


BASE_STRENGTH: dict[EdgeType, float] = {
    EdgeType.TOOL: 1.0,
    EdgeType.READS: 1.0,
    EdgeType.CHAIN: 0.5,
    EdgeType.SERVES: 0.9,
    EdgeType.REFERS: 0.5,
    EdgeType.MENTIONS: 0.3,
    EdgeType.SUPERSEDES: 0.3,
    EdgeType.REJECTS: 0.3,
    EdgeType.ORACLE: 1.0,
}

L0, L1, L2, L3 = 0, 1, 2, 3
LEVELS = (L0, L1, L2, L3)


@dataclass
class Node:
    id: str
    role: Role
    text: str
    seq: int                       # position in the conversation
    turn: int = 0                  # user-turn index
    frame: int = 0                 # sub-task frame (new at each user turn and after each decision)
    step: int | None = None        # tool-call step index
    msg_index: int = -1
    tool_name: str | None = None
    superseded_by: str | None = None
    rejected_by: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def tombstoned(self) -> bool:
        return self.superseded_by is not None or self.rejected_by is not None


@dataclass
class Edge:
    src: str
    dst: str
    type: EdgeType
    strength: float
    source: str = "structural"     # structural | symbolic | oracle | llm


@dataclass
class Graph:
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    roots: dict[str, float] = field(default_factory=dict)   # node id -> initial strength
    _out: dict[str, list[Edge]] = field(default_factory=dict, repr=False)
    _in: dict[str, list[Edge]] = field(default_factory=dict, repr=False)

    def add_node(self, node: Node) -> Node:
        self.nodes[node.id] = node
        return node

    def add_edge(self, edge: Edge) -> None:
        if edge.src not in self.nodes or edge.dst not in self.nodes or edge.src == edge.dst:
            return
        # keep the strongest edge per (src, dst, type)
        for e in self._out.get(edge.src, []):
            if e.dst == edge.dst and e.type == edge.type:
                if edge.strength > e.strength:
                    e.strength = edge.strength
                    e.source = edge.source
                return
        self.edges.append(edge)
        self._out.setdefault(edge.src, []).append(edge)
        self._in.setdefault(edge.dst, []).append(edge)

    def out_edges(self, nid: str) -> list[Edge]:
        return self._out.get(nid, [])

    def in_edges(self, nid: str) -> list[Edge]:
        return self._in.get(nid, [])

    def has_edge(self, src: str, dst: str) -> bool:
        return any(e.dst == dst for e in self._out.get(src, []))

    def ordered(self) -> list[Node]:
        return sorted(self.nodes.values(), key=lambda n: n.seq)

    def current_frame(self) -> int:
        return max((n.frame for n in self.nodes.values()), default=0)
