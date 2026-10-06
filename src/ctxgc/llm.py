"""Model-backed stages: edge inference at ingest time and L1/L2 summaries.

These are the only two places the pipeline calls a model. Neither is exercised
by the offline eval (evalkit/) - that eval uses oracle edges and an extractive
summarizer so the selection mechanism can be measured on its own. Requires the
`anthropic` package and credentials (ANTHROPIC_API_KEY or an `ant auth login`
profile).
"""

from __future__ import annotations

import json
from typing import Protocol

from .model import L1, L2, Node
from .summarize import ExtractiveSummarizer

MODEL = "claude-opus-5-5"


class LLM(Protocol):
    def json(self, system: str, user: str, schema: dict) -> dict: ...


class AnthropicLLM:
    def __init__(self, model: str = MODEL, effort: str = "low", max_tokens: int = 2048):
        import anthropic  # optional dependency

        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens

    def json(self, system: str, user: str, schema: dict) -> dict:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
        )
        if response.stop_reason == "refusal":
            raise RuntimeError(f"refused: {response.stop_details}")
        text = next(b.text for b in response.content if b.type == "text")
        return json.loads(text)


EDGE_SYSTEM = """You maintain a dependency graph over an agent conversation for context compression.
Given one NEW chunk and an INDEX of earlier chunks (one line each), output:
- role: goal | constraint | user | chatter | decision | assistant | tool_call | tool_result
- edges: which earlier chunks the new chunk depends on. Types:
    derived_from (new chunk is reasoned from it), supports (new chunk serves that goal/decision),
    supersedes (new chunk replaces that chunk's value/decision), rejects (new chunk rejects that proposal),
    mentions (only mentions it).
Connect at most 4 chunks. Prefer no edge over a doubtful one. confidence in [0,1]."""

EDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "role": {"type": "string",
                 "enum": ["goal", "constraint", "user", "chatter", "decision", "assistant", "tool_call", "tool_result"]},
        "closes_frame": {"type": "boolean"},
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "type": {"type": "string",
                             "enum": ["derived_from", "supports", "supersedes", "rejects", "mentions"]},
                    "confidence": {"type": "number"},
                },
                "required": ["to", "type", "confidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["role", "closes_frame", "edges"],
    "additionalProperties": False,
}

TYPE_STRENGTH = {"derived_from": 1.0, "supports": 0.8, "supersedes": 0.3, "rejects": 0.3, "mentions": 0.3}


def infer_edges(llm: LLM, new_node: Node, index: list[tuple[str, str]]) -> dict:
    """index: [(node_id, one-line L2 summary), ...] for existing nodes."""
    idx = "\n".join(f"#{nid}: {line}" for nid, line in index) or "(empty)"
    user = f"INDEX:\n{idx}\n\nNEW CHUNK #{new_node.id} ({new_node.role.value}):\n{new_node.text}"
    out = llm.json(EDGE_SYSTEM, user, EDGE_SCHEMA)
    for e in out.get("edges", []):
        e["strength"] = TYPE_STRENGTH[e["type"]] * max(0.0, min(1.0, float(e["confidence"])))
    return out


SUMMARY_SYSTEM = """Produce two compressed versions of a chunk from an agent conversation.
l1: one short paragraph keeping every concrete number, identifier, file name, decision and reason.
l2: one sentence with only the conclusion."""

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {"l1": {"type": "string"}, "l2": {"type": "string"}},
    "required": ["l1", "l2"],
    "additionalProperties": False,
}


class LLMSummarizer(ExtractiveSummarizer):
    """L1/L2 from the model, L0/L3 as in the extractive summarizer. Falls back to
    extractive text on any error so compression never blocks on the model."""

    def __init__(self, llm: LLM, **kw):
        super().__init__(**kw)
        self.llm = llm
        self.cache: dict[str, dict[int, str]] = {}

    def versions(self, node: Node) -> dict[int, str]:
        key = f"{node.id}:{hash(node.text)}"
        if key in self.cache:
            return self.cache[key]
        v = super().versions(node)
        try:
            out = self.llm.json(SUMMARY_SYSTEM, node.text, SUMMARY_SCHEMA)
            v[L1], v[L2] = out["l1"], out["l2"]
        except Exception:
            pass
        self.cache[key] = v
        return v
