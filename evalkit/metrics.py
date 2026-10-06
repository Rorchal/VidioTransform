"""Scoring: answer containment, retrievability, stale exposure, edge P/R."""

from __future__ import annotations

import re

from ctxgc.compress import Result, build_graph
from ctxgc.model import EdgeType, L2

from .synth import Case, Question

FORMAT_EDGES = {EdgeType.TOOL, EdgeType.READS}   # given by the transcript format, not inferred


def present(text: str, answer: str) -> bool:
    pat = r"(?<![\w.])" + re.escape(answer) + r"(?![\w])"
    return re.search(pat, text, re.I) is not None


def stub_present(text: str, node_id: str) -> bool:
    return re.search(r"#" + re.escape(node_id) + r"(?![\w.])", text) is not None


def score_question(q: Question, r: Result) -> dict:
    kept = any(present(r.text, a) for a in q.answers)
    retrievable = kept or any(stub_present(r.text, nid) for nid in q.node_ids)
    out = {"category": q.category, "kept": kept, "retrievable": retrievable}
    if q.stale_node is not None:
        node = r.graph.nodes.get(q.stale_node)
        lvl = r.level.get(q.stale_node, 3)
        tomb = node.tombstoned if (node is not None and r.method == "graded") else False
        out["stale_exposed"] = (node is not None and q.stale_node not in r.dropped and lvl <= L2 and not tomb)
    return out


def edge_quality(case: Case) -> dict:
    """Heuristic (structural+symbolic, model-free) edges vs the oracle's."""
    g = build_graph(case.messages)
    truth = {(s, d) for s, d, _ in case.oracle.edges}
    truth |= set(case.oracle.supersedes) | set(case.oracle.rejects)
    pred = {(e.src, e.dst) for e in g.edges if e.type not in FORMAT_EDGES}
    hit = pred & truth
    # tombstone correctness: was the outdated/rejected node tombstoned at all (lenient),
    # and was it attributed to the exact node the oracle names (strict)?
    sup_hit = sum(1 for _, old in case.oracle.supersedes if g.nodes[old].superseded_by is not None)
    sup_strict = sum(1 for new, old in case.oracle.supersedes if g.nodes[old].superseded_by == new)
    rej_hit = sum(1 for _, prop in case.oracle.rejects if g.nodes[prop].rejected_by is not None)
    rej_strict = sum(1 for rej, prop in case.oracle.rejects if g.nodes[prop].rejected_by == rej)
    false_tomb = sum(1 for n in g.nodes.values() if n.tombstoned) - sup_hit - rej_hit
    # role classification of user facts (goal/constraint/user/chatter)
    role_total = role_hit = 0
    for nid, role in case.oracle.roles.items():
        if role in ("goal", "constraint", "user", "chatter") and nid in g.nodes:
            role_total += 1
            role_hit += g.nodes[nid].role.value == role
    return {
        "precision": len(hit) / len(pred) if pred else 0.0,
        "recall": len(hit) / len(truth) if truth else 0.0,
        "n_pred": len(pred), "n_truth": len(truth),
        "supersedes_recall": sup_hit / max(1, len(case.oracle.supersedes)),
        "supersedes_strict": sup_strict / max(1, len(case.oracle.supersedes)),
        "rejects_recall": rej_hit / max(1, len(case.oracle.rejects)),
        "rejects_strict": rej_strict / max(1, len(case.oracle.rejects)),
        "false_tombstones": false_tomb,
        "user_role_acc": role_hit / max(1, role_total),
    }
