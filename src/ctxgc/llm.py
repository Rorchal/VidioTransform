"""Model-backed stages: edge inference at ingest time and L1/L2 summaries.

These are the only two places the pipeline calls a model. Everything else is
deterministic code. Two backends:

  AnthropicLLM   claude via the `anthropic` package (ANTHROPIC_API_KEY)
  ClaudeCLILLM   claude via the local `claude -p` command (the CLI's own sign-in)
  DeepSeekLLM    deepseek via its OpenAI-compatible HTTP API, stdlib only
                 (DEEPSEEK_API_KEY). JSON-object mode + schema in the prompt +
                 validation/retry, since that API does not enforce a schema.

Both sit behind the `LLM` protocol (one method: json(system, user, schema)) and
can be wrapped in `CachedLLM`, which memoises replies in a JSONL file keyed by
(model, system, user) so an eval run is reproducible and re-runs are free.

`infer_graph` is the ingest-time "mark" phase done by a model: one call per
prose chunk (tool calls/results keep their format-given edges and are not sent)
with the new chunk + a one-line index of every earlier chunk. The model returns
the chunk's role and up to a few typed edges to earlier chunks; `supersedes` /
`rejects` edges also set the tombstone on their target.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol

from .ingest import assign_roots, reassign_frames
from .model import L1, L2, Edge, EdgeType, Graph, Node, Role
from .summarize import ExtractiveSummarizer, monotonic
from .tokens import count

ANTHROPIC_MODEL = "claude-opus-5-5"
DEEPSEEK_MODEL = "deepseek-flash"


class LLM(Protocol):
    name: str

    def json(self, system: str, user: str, schema: dict) -> dict: ...


# --------------------------------------------------------------------------- backends

class AnthropicLLM:
    name: str

    def __init__(self, model: str = ANTHROPIC_MODEL, effort: str = "low", max_tokens: int = 2048):
        import anthropic  # optional dependency

        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.name = f"anthropic:{model}:{effort}"

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


class DeepSeekLLM:
    """OpenAI-compatible chat completions with `response_format: json_object`.

    effort: "off" disables thinking; "low" / "high" / "max" set the reasoning
    effort. The schema is pasted into the system prompt and the reply is
    validated against it (types, enums, required keys); a bad reply is retried
    with the validation error appended, up to `retries` times.
    """

    URL = "https://api.deepseek.com/chat/completions"
    name: str

    def __init__(self, model: str = DEEPSEEK_MODEL, effort: str = "low", max_tokens: int = 4096,
                 max_tokens_cap: int = 32768, retries: int = 3, timeout: float = 180.0,
                 api_key: str | None = None, transport=None):
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens          # reasoning tokens count against it: doubled on truncation
        self.max_tokens_cap = max_tokens_cap
        self.retries = retries
        self.timeout = timeout
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        # a missing key only matters on the first real request: behind CachedLLM
        # a fully cached run needs no credentials at all
        self.transport = transport or self._http       # transport(body: dict) -> response dict (for tests)
        self.name = f"deepseek:{model}:{effort}"
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0,
                      "retries": 0, "truncated": 0}
        self._lock = threading.Lock()

    def _http(self, body: dict) -> dict:
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")
        req = urllib.request.Request(
            self.URL, data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
        delay = 2.0
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < 4:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise RuntimeError(f"deepseek HTTP {e.code}: {e.read()[:300]!r}") from e
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt < 4:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise
        raise RuntimeError("unreachable")

    def json(self, system: str, user: str, schema: dict) -> dict:
        sys_prompt = (f"{system}\n\nAnswer with a single JSON object (no prose) matching this JSON schema:\n"
                      f"{json.dumps(schema)}")
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}]
        last_err: Exception | None = None
        max_tokens = self.max_tokens
        bad = 0
        while bad < self.retries:
            body: dict = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                          "response_format": {"type": "json_object"}}
            if self.effort == "off":
                body["thinking"] = {"type": "disabled"}
            else:
                body["reasoning_effort"] = self.effort
            resp = self.transport(body)
            self._account(resp)
            choice = resp["choices"][0]
            content = choice["message"].get("content") or ""
            if choice.get("finish_reason") == "length" and max_tokens < self.max_tokens_cap:
                # the model thought past the limit and never wrote the answer:
                # give it more room, same conversation
                max_tokens = min(max_tokens * 2, self.max_tokens_cap)
                with self._lock:
                    self.usage["truncated"] += 1
                continue
            bad += 1
            try:
                out = json.loads(content)
                validate(out, schema)
                return out
            except (ValueError, TypeError) as e:        # json.JSONDecodeError is a ValueError
                last_err = e
                with self._lock:
                    self.usage["retries"] += 1
                messages = messages[:2] + [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": f"That reply was invalid ({e}). Return only a JSON object matching the schema."}]
        raise RuntimeError(f"deepseek: no valid JSON after {self.retries} attempts: {last_err}")

    def _account(self, resp: dict) -> None:
        u = resp.get("usage") or {}
        with self._lock:
            self.usage["calls"] += 1
            self.usage["prompt_tokens"] += u.get("prompt_tokens", 0)
            self.usage["completion_tokens"] += u.get("completion_tokens", 0)
            self.usage["reasoning_tokens"] += (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)


class ClaudeCLILLM:
    """Claude through the local `claude` command in headless mode (`claude -p`),
    so inference is billed to the Claude subscription the CLI is signed in
    with rather than to an API key. One process per call, no tools, JSON
    output; the schema goes into the system prompt and the reply is validated
    like the DeepSeek backend's. `cwd` defaults to a scratch directory so the
    nested CLI picks up no project settings or hooks."""

    name: str

    def __init__(self, model: str = "haiku", retries: int = 3, timeout: float = 300.0, cwd: str | None = None,
                 runner=None):
        import tempfile

        self.model = model
        self.retries = retries
        self.timeout = timeout
        self.cwd = cwd or tempfile.mkdtemp(prefix="ctxgc-claude-")
        self.runner = runner or self._run        # runner(system, user) -> dict (for tests)
        self.name = f"claude-cli:{model}"
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "retries": 0}
        self._lock = threading.Lock()

    def _run(self, system: str, user: str) -> dict:
        """One `claude -p` process. A failed run (rate limit, usage window
        exhausted, transient error) is retried with a growing pause, up to about
        ten minutes in total, so a long batch survives a 5-hour-window reset."""
        import subprocess

        cmd = ["claude", "-p", "--model", self.model, "--output-format", "json", "--tools", "",
               "--system-prompt", system]
        delay = 30.0
        for attempt in range(6):
            proc = subprocess.run(cmd, input=user, capture_output=True, text=True, timeout=self.timeout, cwd=self.cwd)
            out = proc.stdout.strip()
            if out:
                try:
                    resp = json.loads(out)
                    if not resp.get("is_error"):
                        return resp
                    err = str(resp.get("result") or "")[:300]
                except json.JSONDecodeError:
                    err = out[-300:]
            else:
                err = proc.stderr[-300:]
            if attempt == 5:
                raise RuntimeError(f"claude -p failed ({proc.returncode}): {err}")
            time.sleep(delay)
            delay = min(delay * 2, 300.0)
        raise RuntimeError("unreachable")

    def json(self, system: str, user: str, schema: dict) -> dict:
        sys_prompt = (f"{system}\n\nAnswer with a single JSON object (no prose, no code fence) matching this JSON schema:\n"
                      f"{json.dumps(schema)}")
        last_err: Exception | None = None
        prompt = user
        for attempt in range(self.retries):
            resp = self.runner(sys_prompt, prompt)
            self._account(resp)
            text = str(resp.get("result") or "").strip()
            if text.startswith("```"):
                text = text.strip("`")
                text = text[text.find("{"):text.rfind("}") + 1]
            try:
                out = json.loads(text)
                validate(out, schema)
                return out
            except (ValueError, TypeError) as e:
                last_err = e
                with self._lock:
                    self.usage["retries"] += 1
                prompt = f"{user}\n\n(Your previous reply was invalid: {e}. Return only a JSON object matching the schema.)"
        raise RuntimeError(f"claude-cli: no valid JSON after {self.retries} attempts: {last_err}")

    def _account(self, resp: dict) -> None:
        u = resp.get("usage") or {}
        with self._lock:
            self.usage["calls"] += 1
            self.usage["input_tokens"] += (u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0)
                                           + u.get("cache_read_input_tokens", 0))
            self.usage["output_tokens"] += u.get("output_tokens", 0)
            self.usage["cost_usd"] += float(resp.get("total_cost_usd") or 0.0)


def validate(value, schema: dict, path: str = "$") -> None:
    """Minimal JSON-schema check: type, enum, required, properties, items,
    additionalProperties (extra keys are dropped rather than rejected)."""
    t = schema.get("type")
    if t == "object":
        if not isinstance(value, dict):
            raise TypeError(f"{path}: expected object")
        for k in schema.get("required", []):
            if k not in value:
                raise ValueError(f"{path}: missing {k!r}")
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            for k in [k for k in value if k not in props]:
                del value[k]
        for k, sub in props.items():
            if k in value:
                validate(value[k], sub, f"{path}.{k}")
    elif t == "array":
        if not isinstance(value, list):
            raise TypeError(f"{path}: expected array")
        for i, item in enumerate(value):
            validate(item, schema.get("items", {}), f"{path}[{i}]")
    elif t == "string":
        if not isinstance(value, str):
            raise TypeError(f"{path}: expected string")
        if "enum" in schema and value not in schema["enum"]:
            raise ValueError(f"{path}: {value!r} not in {schema['enum']}")
    elif t == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{path}: expected number")
    elif t == "boolean":
        if not isinstance(value, bool):
            raise TypeError(f"{path}: expected boolean")


class CachedLLM:
    """Disk-memoised wrapper: one JSON line per reply, keyed by a hash of
    (backend name, system, user, schema), in a single append-only file so an
    eval's model replies travel with the repo. Thread-safe; misses go to the
    wrapped backend."""

    def __init__(self, inner: LLM, cache_path: str | Path):
        self.inner = inner
        self.name = inner.name
        self.path = Path(cache_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.store: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.store[rec["k"]] = rec["v"]
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()

    def key(self, system: str, user: str, schema: dict) -> str:
        h = hashlib.sha256()
        for part in (self.name, system, user, json.dumps(schema, sort_keys=True)):
            h.update(part.encode())
            h.update(b"\0")
        return h.hexdigest()[:32]

    def cached(self, system: str, user: str, schema: dict) -> dict | None:
        v = self.store.get(self.key(system, user, schema))
        return json.loads(json.dumps(v)) if v is not None else None

    def json(self, system: str, user: str, schema: dict) -> dict:
        k = self.key(system, user, schema)
        with self._lock:
            if k in self.store:
                self.hits += 1
                return json.loads(json.dumps(self.store[k]))     # callers mutate replies
        out = self.inner.json(system, user, schema)
        with self._lock:
            self.store[k] = out
            self.misses += 1
            with self.path.open("a") as f:
                f.write(json.dumps({"k": k, "v": out}, ensure_ascii=False) + "\n")
        return json.loads(json.dumps(out))


def make_llm(spec: str, effort: str = "low", cache_path: str | Path | None = None) -> LLM:
    """spec: "deepseek" | "deepseek:<model>" | "anthropic" | "anthropic:<model>"."""
    backend, _, model = spec.partition(":")
    if backend == "deepseek":
        llm: LLM = DeepSeekLLM(model or DEEPSEEK_MODEL, effort=effort)
    elif backend == "anthropic":
        llm = AnthropicLLM(model or ANTHROPIC_MODEL, effort=effort)
    elif backend == "claude-cli":
        llm = ClaudeCLILLM(model or "haiku")
    else:
        raise ValueError(f"unknown backend {backend!r}")
    return CachedLLM(llm, cache_path) if cache_path else llm


# --------------------------------------------------------------------------- edges

EDGE_SYSTEM = """You maintain a dependency graph over an agent conversation for context compression.
Given one NEW chunk and an INDEX of earlier chunks (one line each), output:
- role of the new chunk:
    goal        the user's task / ask: each sentence stating what to do (also a follow-up ask like "now write the PR description")
    constraint  a hard rule from the user: something that must not be done / changed / added
    user        other on-task user text (an instruction on how to proceed, a value change, an acknowledgement)
    chatter     user text unrelated to the task (small talk, an aside about another project)
    decision    assistant conclusion, proposal, root cause, or applied fix - the result of a sub-task
    assistant   other assistant text (narration, "let me look at ...")
- closes_frame: true if the chunk ends a sub-task (a decision or a final answer)
- edges: which earlier chunks the new chunk depends on. Types:
    derived_from  the new chunk is reasoned from that chunk (evidence it read, a decision it builds on)
    supports      the new chunk serves that goal / instruction
    supersedes    the new chunk replaces that chunk's value or decision (the old one is now stale)
    rejects       the new chunk rejects that proposal
    mentions      only mentions it
Connect at most {max_edges} chunks. Prefer no edge over a doubtful one. confidence in [0,1].
A goal or constraint chunk that opens the conversation has no edges. Chatter has no edges to task chunks.
A follow-up ask from the user is derived_from the earlier decisions and results the assistant needs to carry it out."""

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
EDGE_TYPE = {"derived_from": EdgeType.ORACLE, "supports": EdgeType.SERVES, "mentions": EdgeType.MENTIONS,
             "supersedes": EdgeType.SUPERSEDES, "rejects": EdgeType.REJECTS}
# Graph edges point from the chunk that matters to the chunk that then matters
# too. "new derived_from old" => new -> old. "new supports goal" => the goal
# matters, so the chunk serving it matters: goal -> new (same direction as the
# structural SERVES edge). Tombstone types point new -> old like the oracle's.
REVERSED = {"supports"}
USER_ROLES = {Role.GOAL, Role.CONSTRAINT, Role.USER, Role.CHATTER}
ASSISTANT_ROLES = {Role.DECISION, Role.ASSISTANT}
PROSE_ROLES = USER_ROLES | ASSISTANT_ROLES


def edge_prompt(new_node: Node, index: list[tuple[str, str]]) -> str:
    idx = "\n".join(f"#{nid}: {line}" for nid, line in index) or "(empty)"
    speaker = "user" if new_node.role in USER_ROLES else "assistant"
    return f"INDEX:\n{idx}\n\nNEW CHUNK #{new_node.id} (said by the {speaker}):\n{new_node.text}"


MAX_EDGES = 4


def edge_system(max_edges: int = MAX_EDGES) -> str:
    return EDGE_SYSTEM.replace("{max_edges}", str(max_edges))


def infer_edges(llm: LLM, new_node: Node, index: list[tuple[str, str]], max_edges: int = MAX_EDGES) -> dict:
    """index: [(node_id, one-line summary), ...] for existing nodes."""
    out = llm.json(edge_system(max_edges), edge_prompt(new_node, index), EDGE_SCHEMA)
    for e in out.get("edges", []):
        e["to"] = str(e["to"]).lstrip("#")
        e["strength"] = TYPE_STRENGTH[e["type"]] * max(0.0, min(1.0, float(e["confidence"])))
    return out


def index_line(node: Node, versions: dict[int, str]) -> str:
    return f"[{node.role.value}] {versions[L2]}"


def infer_graph(g: Graph, llm: LLM, *, workers: int = 8, index_summarizer=None,
                max_edges: int = MAX_EDGES) -> dict[str, dict]:
    """Model-based mark phase over an ingested graph (structural TOOL/READS edges
    already present). For every prose node, in conversation order, ask the model
    for its role and edges to earlier nodes. Calls are independent (the index is
    built from deterministic one-line summaries), so they run in parallel.

    Applies: roles on user / assistant text nodes, typed edges (source "llm"),
    supersedes / rejects tombstones, frame boundaries (user turns + decisions),
    roots. Returns the raw model output per node id."""
    summarizer = index_summarizer or ExtractiveSummarizer()
    nodes = g.ordered()
    versions = {n.id: summarizer.versions(n) for n in nodes}
    index_lines = [(n.id, index_line(n, versions[n.id])) for n in nodes]
    targets = [(i, n) for i, n in enumerate(nodes) if n.role in PROSE_ROLES]

    def ask(item):
        i, n = item
        return n.id, infer_edges(llm, n, index_lines[:i], max_edges)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        raw = dict(ex.map(ask, targets))

    pos = {n.id: i for i, n in enumerate(nodes)}
    for nid, out in raw.items():
        node = g.nodes[nid]
        role = Role(out["role"])
        if (node.role in USER_ROLES and role in USER_ROLES) or (node.role in ASSISTANT_ROLES and role in ASSISTANT_ROLES):
            node.role = role
        if node.role == Role.DECISION:
            node.meta["proposal"] = node.meta.get("proposal") or node.text.lower().startswith("proposal")
        node.meta["closes_frame"] = bool(out.get("closes_frame"))
        for e in out.get("edges", []):
            dst = e["to"]
            if dst not in g.nodes or pos[dst] >= pos[nid] or e["strength"] <= 0:
                continue        # only edges to earlier, existing nodes
            etype = EDGE_TYPE[e["type"]]
            if e["type"] in REVERSED:
                g.add_edge(Edge(dst, nid, etype, e["strength"], source="llm"))
            else:
                g.add_edge(Edge(nid, dst, etype, e["strength"], source="llm"))
            target = g.nodes[dst]
            if etype == EdgeType.SUPERSEDES and target.superseded_by is None and target.rejected_by is None:
                target.superseded_by = nid
            elif etype == EdgeType.REJECTS and target.rejected_by is None and target.superseded_by is None:
                target.rejected_by = nid
    reassign_frames(g)
    assign_roots(g)
    return raw


# --------------------------------------------------------------------------- summaries

SUMMARY_SYSTEM = """Produce two compressed versions of one CHUNK from an agent conversation.
Summarize the CHUNK only. CONTEXT says what the conversation is about so you can judge what in the chunk matters; never restate or paraphrase the context itself.
l1: one short paragraph (at most ~60 words) keeping every concrete number, identifier, file name, decision and reason in the chunk that matters for the task. For long tool output (logs, source, configs, test runs) keep the few relevant lines verbatim, including odd details (comments, warnings, config values) about the code or service under discussion; drop the routine noise.
l2: one sentence with only the chunk's conclusion.
Never invent facts and add no interpretation or commentary beyond what the chunk states; copy numbers, identifiers and short quoted phrases exactly as written."""

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {"l1": {"type": "string"}, "l2": {"type": "string"}},
    "required": ["l1", "l2"],
    "additionalProperties": False,
}


def summary_prompt(node: Node, context: str) -> str:
    kind = node.role.value.replace("_", " ")
    return f"CONTEXT (what the conversation is about):\n{context or '(unknown)'}\n\nCHUNK ({kind}):\n{node.text}"


def graph_context(g: Graph, max_chars: int = 600) -> str:
    """Task context handed to the summarizer: the opening user turn (the ask).
    Independent of inferred roles, so every graph of a conversation shares it."""
    parts = [n.text for n in g.ordered() if n.turn == 1 and n.role in USER_ROLES]
    return " ".join(parts)[:max_chars]


class LLMSummarizer(ExtractiveSummarizer):
    """L1/L2 from the model, L0/L3 as in the extractive summarizer. Falls back to
    extractive text on any error so compression never blocks on the model.
    `context` is the task description the model should judge salience against
    (set per conversation, see graph_context)."""

    def __init__(self, llm: LLM, context: str = "", roles=None, min_tokens: int = 0, **kw):
        super().__init__(**kw)
        self.llm = llm
        self.context = context
        self.roles = set(roles) if roles else None     # None: every node goes to the model
        self.min_tokens = min_tokens                   # shorter nodes keep the extractive versions
        self.cache: dict[str, dict[int, str]] = {}
        self.failures = 0
        self.skipped = 0
        self._lock = threading.Lock()

    def _wants_model(self, node: Node) -> bool:
        if self.roles is not None and node.role not in self.roles:
            return False
        return count(node.text) >= self.min_tokens

    def _key(self, node: Node) -> str:
        return f"{node.id}:{hash((node.text, node.role.value, self.context))}"

    def versions(self, node: Node) -> dict[int, str]:
        key = self._key(node)
        with self._lock:
            if key in self.cache:
                return self.cache[key]
        v = super().versions(node)
        if not self._wants_model(node):
            with self._lock:
                self.skipped += 1
                self.cache[key] = v
            return v
        try:
            out = self.llm.json(SUMMARY_SYSTEM, summary_prompt(node, self.context), SUMMARY_SCHEMA)
            l1, l2 = out["l1"].strip(), out["l2"].strip()
            if l1 and l2:
                v[L1], v[L2] = l1, l2
        except Exception:
            with self._lock:
                self.failures += 1
        monotonic(v)      # re-check after the override
        with self._lock:
            self.cache[key] = v
        return v

    def prefetch(self, nodes: list[Node], workers: int = 8) -> None:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            list(ex.map(self.versions, nodes))
