"""Assemble the compressed context from per-node levels.

Nothing is deleted: a node that gets no budget is rendered as a stub carrying its
id, so the model can ask for it back. Consecutive stubs are collapsed into one
line - with dozens of nodes, one bracket per stub would otherwise eat a tenth of
the budget before anything is promoted.
"""

from __future__ import annotations

from collections.abc import Iterable

from .model import L0, L1, L2, L3, Graph, Node, Role
from .tokens import count

TAG = {L0: "", L1: "|summary", L2: "|brief", L3: ""}


def tombstone_text(node: Node, versions: dict[int, str]) -> str:
    if node.superseded_by is not None:
        return f"[#{node.id} SUPERSEDED by #{node.superseded_by}: {versions[L2]}]"
    if node.rejected_by is not None:
        return f"[#{node.id} REJECTED, see #{node.rejected_by}: {versions[L2]}]"
    raise ValueError("not a tombstone")


def stub_label(node: Node) -> str:
    """Short label used inside a collapsed group of stubs."""
    if node.role == Role.TOOL_RESULT:
        return f"{node.tool_name or 'tool'} result {len(node.text.splitlines())} lines"
    if node.role == Role.TOOL_CALL:
        return f"{node.tool_name or 'tool'} call"
    words = node.text.split()
    head = " ".join(words[:3]) + ("…" if len(words) > 3 else "")
    return f'{node.role.value} "{head}"'


def group_stub(nodes: list[Node]) -> str:
    ids = " ".join(f"#{n.id}" for n in nodes)
    labels = "; ".join(stub_label(n) for n in nodes)
    return f"[{ids} — {len(nodes)} chunks: {labels}]"


def render_node(node: Node, level: int, versions: dict[int, str]) -> str:
    if node.tombstoned:
        return tombstone_text(node, versions)
    if level == L3:
        return versions[L3]
    return f"[#{node.id} {node.role.value}{TAG[level]}] {versions[level]}"


def is_stub(node: Node, level: int) -> bool:
    return level == L3 and not node.tombstoned


def render_parts(
    g: Graph,
    level: dict[str, int],
    versions: dict[str, dict[int, str]],
    dropped: Iterable[str] = (),
) -> list[str]:
    dropped = set(dropped)
    parts: list[str] = []
    run: list[Node] = []

    def flush() -> None:
        if len(run) == 1:
            parts.append(render_node(run[0], L3, versions[run[0].id]))
        elif run:
            parts.append(group_stub(run))
        run.clear()

    for n in g.ordered():
        if n.id in dropped:
            continue
        lvl = level.get(n.id, L3)
        if is_stub(n, lvl):
            run.append(n)
            continue
        flush()
        parts.append(render_node(n, lvl, versions[n.id]))
    flush()
    return parts


def render(g: Graph, level: dict[str, int], versions: dict[str, dict[int, str]], dropped: Iterable[str] = ()) -> str:
    return "\n".join(render_parts(g, level, versions, dropped))


def parts_cost(parts: list[str]) -> int:
    return sum(count(p) for p in parts)
