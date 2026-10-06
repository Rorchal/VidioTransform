"""Synthetic debugging conversations with ground truth.

Each case is an agent transcript (user goal + constraint, tool steps with noisy
outputs, a decision, a rejected proposal, a value that gets updated, an off-task
chatter exchange, a fix, tests) plus:
  - oracle roles / edges / supersessions / rejections (what a perfect edge
    inferrer would output), and
  - questions with short answers and the node ids that hold them.

Noise is randomized so planted facts are not trivially extractable. Phrasings of
the update and rejection vary so the regex heuristics catch some and miss some -
the miss rate is one of the things the eval measures.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from ctxgc.compress import Oracle

SERVICES = [
    ("LoginService", "loadProfile", "users", "/api/login", "src/services/login_service.py"),
    ("CheckoutService", "fetchCart", "orders", "/api/checkout", "src/services/checkout_service.py"),
    ("SearchService", "buildIndex", "products", "/api/search", "src/services/search_service.py"),
    ("BillingService", "computeInvoice", "invoices", "/api/billing", "src/services/billing_service.py"),
]
OTHER_SERVICES = ["OrderService", "CartService", "AuthService", "MailService", "ReportService", "SyncService"]
OTHER_METHODS = ["refresh", "warmCache", "flush", "rotateKeys", "sendDigest", "reconcile", "pollQueue"]
OTHER_TABLES = ["sessions", "audit_log", "carts", "payments", "events", "tags"]
CACHES = ["Redis", "Memcached"]
COLORS = ["teal", "amber", "indigo", "crimson", "olive", "cobalt"]
CONSTRAINTS = [("Do not change the database schema.", "schema"),
               ("You must not modify the auth module.", "auth module")]
TIMEOUT_PAIRS = [("30s", "10s"), ("45s", "15s"), ("60s", "20s"), ("90s", "25s")]
TIMEOUT_KEYS = ["REQUEST_TIMEOUT", "UPSTREAM_TIMEOUT", "DB_QUERY_TIMEOUT"]
CONFIG_NOISE_KEYS = ["MAX_CONNECTIONS", "POOL_SIZE", "LOG_LEVEL", "RETRY_COUNT", "CACHE_TTL_MS",
                     "WORKER_THREADS", "QUEUE_DEPTH", "METRICS_PORT", "FEATURE_FLAGS", "REGION",
                     "BATCH_SIZE", "HEARTBEAT_MS", "GC_INTERVAL_MS", "TLS_MODE", "SHARD_COUNT"]


@dataclass
class Question:
    category: str
    question: str
    answers: list[str]                 # any of these present => answer retained
    node_ids: list[str]                # nodes that hold the answer
    stale_node: str | None = None      # outdated node (for the stale-exposure metric)


@dataclass
class Case:
    seed: int
    messages: list[dict]
    oracle: Oracle
    questions: list[Question]
    meta: dict = field(default_factory=dict)


class _Builder:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.messages: list[dict] = []
        self.oracle = Oracle()
        self._calls = 0

    def user(self, sentences: list[str], roles: list[str]) -> list[str]:
        assert len(sentences) == len(roles)
        mi = len(self.messages)
        self.messages.append({"role": "user", "content": " ".join(sentences)})
        if len(sentences) == 1:
            ids = [f"m{mi}"]
        else:
            ids = [f"m{mi}.s{k}" for k in range(len(sentences))]
        for nid, r in zip(ids, roles):
            self.oracle.roles[nid] = r
        return ids

    def assistant(self, text: str, role: str = "assistant", call: tuple[str, dict] | None = None) -> tuple[str, str | None]:
        mi = len(self.messages)
        msg: dict = {"role": "assistant", "content": text}
        call_id = None
        if call is not None:
            self._calls += 1
            call_id = f"call_{self._calls}"
            msg["tool_calls"] = [{"id": call_id, "name": call[0], "args": call[1]}]
        self.messages.append(msg)
        if text:
            self.oracle.roles[f"m{mi}"] = role
        return (f"m{mi}" if text else None), call_id

    def tool(self, call_id: str, content: str) -> str:
        mi = len(self.messages)
        self.messages.append({"role": "tool", "tool_call_id": call_id, "content": content})
        return f"m{mi}"

    def edge(self, src: str | None, dst: str | None, s: float) -> None:
        if src and dst:
            self.oracle.edges.append((src, dst, s))


def _ts(rng: random.Random, base_min: int) -> str:
    return f"2026-10-06 14:{base_min:02d}:{rng.randint(0, 59):02d}.{rng.randint(0, 999):03d}"


def _log(rng: random.Random, svc: str, meth: str, table: str, q: int, lat: str, code: str) -> str:
    n = rng.randint(70, 140)
    lines = []
    for _ in range(n):
        kind = rng.random()
        if kind < 0.6:
            lines.append(f"{_ts(rng, rng.randint(20, 23))} [INFO] {rng.choice(OTHER_SERVICES)}.{rng.choice(OTHER_METHODS)} "
                         f"query={rng.choice(OTHER_TABLES)} id={rng.randint(1000, 9999)} took {rng.randint(1, 9)}ms")
        elif kind < 0.85:
            lines.append(f"{_ts(rng, rng.randint(20, 23))} [INFO] GET /api/{rng.choice(['health', 'metrics', 'cart', 'me'])} "
                         f"200 {rng.randint(10, 400)}ms")
        else:
            lines.append(f"{_ts(rng, rng.randint(20, 23))} [WARN] pool wait {rng.randint(100, 900)}ms on {rng.choice(OTHER_TABLES)}")
    req = f"req-{rng.getrandbits(24):06x}"
    planted = [
        f"{_ts(rng, 22)} [WARN] {svc}.{meth} executed {q} queries for request {req} (SELECT * FROM {table} WHERE id = ?)",
        f"{_ts(rng, 22)} [INFO] {svc}.{meth} avg query latency {lat}ms over {q} queries",
        f"{_ts(rng, 23)} [ERROR] GET {{endpoint}} 504 code={code} after upstream timeout",
    ]
    for p in planted:
        lines.insert(rng.randint(len(lines) // 4, len(lines) - 1), p)
    return "\n".join(lines)


def _source(rng: random.Random, svc: str, meth: str, table: str) -> str:
    lines = [f"class {svc}:", "    def __init__(self, db, cache=None):", "        self.db = db", "        self.cache = cache", ""]
    for _ in range(rng.randint(5, 9)):
        name = rng.choice(["validate", "serialize", "audit", "retry", "normalize", "paginate", "hydrate", "emit"]) + f"_{rng.randint(1, 99)}"
        lines += [f"    def {name}(self, payload):",
                  f"        if not payload.get('{rng.choice(['id', 'token', 'cursor'])}'):",
                  "            raise ValueError('missing field')",
                  f"        return self.db.query('SELECT count(*) FROM {rng.choice(OTHER_TABLES)}')", ""]
    planted = [f"    def {meth}(self, ids):", "        rows = []", "        for id in ids:",
               f"            rows.append(self.db.query('SELECT * FROM {table} WHERE id = %s', id))",
               "        # TODO batch this", "        return rows", ""]
    pos = rng.randint(5, len(lines) - 1)
    lines[pos:pos] = planted
    return "\n".join(lines)


def _config(rng: random.Random, key: str, old: str) -> str:
    keys = rng.sample(CONFIG_NOISE_KEYS, rng.randint(10, 15))
    lines = []
    for k in keys:
        if k.endswith("_MS"):
            v = str(rng.randint(100, 9000))
        elif k == "LOG_LEVEL":
            v = rng.choice(["INFO", "DEBUG", "WARN"])
        elif k == "TLS_MODE":
            v = rng.choice(["strict", "relaxed"])
        elif k == "REGION":
            v = rng.choice(["us-east-1", "eu-west-2"])
        elif k == "FEATURE_FLAGS":
            v = "batch_reads,new_dashboard"
        else:
            v = str(rng.randint(2, 64))
        lines.append(f"{k}: {v}")
    lines.insert(rng.randint(2, len(lines) - 1), f"{key}: {old}")
    return "\n".join(lines)


def _diff(svc: str, meth: str, table: str, path: str) -> str:
    return "\n".join([
        f"--- a/{path}", f"+++ b/{path}", f"@@ -41,6 +41,4 @@ class {svc}:",
        f"     def {meth}(self, ids):",
        "-        rows = []", "-        for id in ids:",
        f"-            rows.append(self.db.query('SELECT * FROM {table} WHERE id = %s', id))",
        "-        # TODO batch this", "-        return rows",
        f"+        rows = self.db.query('SELECT * FROM {table} WHERE id IN %s', tuple(ids))",
        "+        return list(rows)",
    ])


def _tests(rng: random.Random, t: int) -> str:
    lines = [f"test_{rng.choice(['login', 'cart', 'auth', 'index', 'invoice'])}_{rng.randint(1, 500)} PASSED" for _ in range(rng.randint(20, 35))]
    lines.append(f"========== {t} passed, 0 failed in {rng.randint(8, 40)}.{rng.randint(0, 9)}s ==========")
    return "\n".join(lines)


def make_case(seed: int) -> Case:
    rng = random.Random(seed)
    svc, meth, table, endpoint, path = rng.choice(SERVICES)
    q = rng.randint(20, 80)
    lat = rng.choice(["1.7", "2.1", "2.4", "3.3", "4.6"])
    code = f"E{rng.randint(4000, 4999)}"
    cache = rng.choice(CACHES)
    color = rng.choice(COLORS)
    constraint_text, constraint_key = rng.choice(CONSTRAINTS)
    key = rng.choice(TIMEOUT_KEYS)
    old, new = rng.choice(TIMEOUT_PAIRS)
    t_pass = rng.randint(40, 200)
    chatter_slot = rng.choice(["after_rootcause", "after_reject", "after_config", "after_fix"])
    reject_variant = rng.randrange(3)
    update_variant = rng.randrange(3)

    b = _Builder(rng)
    meta = dict(svc=svc, meth=meth, table=table, path=path, q=q, lat=lat, code=code, cache=cache, color=color,
                key=key, old=old, new=new, tests=t_pass, chatter_slot=chatter_slot,
                reject_variant=reject_variant, update_variant=update_variant)

    # --- goal + constraint
    goal_ids = b.user(
        [f"{svc} on {endpoint} is timing out under load (error {code}).",
         "Please find the root cause and fix it.",
         constraint_text],
        ["goal", "goal", "constraint"])
    goal, constraint = goal_ids[1], goal_ids[2]

    chatter_ids: dict[str, str] = {}

    def maybe_chatter(slot: str) -> None:
        if slot != chatter_slot:
            return
        u = b.user([f"By the way, for the new metrics dashboard I like {color} as the primary color.", "Any name ideas?"],
                   ["chatter", "chatter"])
        a, _ = b.assistant(f"{color.capitalize()} reads well on dark backgrounds. Name ideas: Pulse, Lantern, Meridian.")
        b.edge(a, u[0], 0.9)
        chatter_ids["user"], chatter_ids["assistant"] = u[0], a

    # --- investigate
    a1, c1 = b.assistant("Let me look at the slow query log first.", call=("grep", {"pattern": svc, "path": "logs/slow.log"}))
    log = b.tool(c1, _log(rng, svc, meth, table, q, lat, code).replace("{endpoint}", endpoint))
    a2, c2 = b.assistant(f"The log shows {svc}.{meth} issuing dozens of SELECTs against {table} per request. Let me read the source.",
                         call=("read_file", {"path": path}))
    src = b.tool(c2, _source(rng, svc, meth, table))
    rootcause, _ = b.assistant(
        f"Root cause: {meth}() in {path} issues one SELECT per id against the {table} table (N+1); the log shows {q} queries per request.",
        role="decision")
    b.edge(goal, rootcause, 0.9)
    b.edge(rootcause, log, 1.0)
    b.edge(rootcause, src, 1.0)
    b.edge(a2, log, 1.0)
    maybe_chatter("after_rootcause")

    # --- proposal + rejection
    proposal, _ = b.assistant(f"Proposal: add a {cache} cache in front of {table} lookups to cut query volume.", role="decision")
    b.edge(goal, proposal, 0.9)
    b.edge(proposal, rootcause, 0.9)
    reject_sentences = [
        ([f"No, don't add {cache} — we don't want a new dependency.", "Use a single batched query instead."],
         ["constraint", "constraint", "user"]),
        ([f"Let's not do {cache}.", "No new dependencies please.", "Batch the query instead."],
         ["constraint", "constraint", "user"]),
        ([f"Skip the {cache} idea; we must not introduce new infrastructure.", f"Rewrite {meth}() to use one IN query."],
         ["constraint", "user"]),
    ][reject_variant]
    sentences, roles = reject_sentences
    roles = roles[: len(sentences)]
    rej_ids = b.user(sentences, roles)
    reject_head, instruction = rej_ids[0], rej_ids[-1]
    b.oracle.rejects.append((reject_head, proposal))
    maybe_chatter("after_reject")

    # --- config check (value that will be updated later)
    a3, c3 = b.assistant("Understood. Before changing code I'll check the current timeout config.",
                         call=("read_file", {"path": "config/app.yaml"}))
    b.edge(a3, rootcause, 0.5)
    cfg = b.tool(c3, _config(rng, key, old))
    keep, _ = b.assistant(f"Decision: keep {key} = {old} for now; the batched query should bring latency well under it.", role="decision")
    b.edge(goal, keep, 0.7)
    b.edge(keep, cfg, 1.0)
    maybe_chatter("after_config")

    # --- fix
    a4, c4 = b.assistant(f"Now applying the fix to {meth}() in {path}.",
                         call=("edit_file", {"path": path, "replace": "loop", "with": "IN query"}))
    b.edge(a4, rootcause, 1.0)
    b.edge(a4, instruction, 0.9)
    diff = b.tool(c4, _diff(svc, meth, table, path))
    a5, c5 = b.assistant("Running the test suite.", call=("run_tests", {"suite": "all"}))
    b.edge(a5, a4, 0.8)
    tests = b.tool(c5, _tests(rng, t_pass))
    fixed, _ = b.assistant(f"Decision: fix applied in {path}; {meth}() now runs a single IN query; {t_pass} tests pass.", role="decision")
    b.edge(goal, fixed, 0.9)
    b.edge(fixed, diff, 1.0)
    b.edge(fixed, tests, 1.0)
    b.edge(fixed, rootcause, 0.8)
    maybe_chatter("after_fix")

    # --- value update (supersedes the earlier decision)
    update_sentences = [
        [f"Actually, change {key} to {new} (was {old})."],
        [f"Update: set {key} to {new}.", "The old value was too generous."],
        [f"One more thing: {key} should be {new} now, not {old}."],
    ][update_variant]
    upd_ids = b.user(update_sentences, ["user"] * len(update_sentences))
    update = upd_ids[0]
    b.oracle.supersedes.append((update, keep))
    a6, c6 = b.assistant(f"Done, {key} is now {new}.", call=("edit_file", {"path": "config/app.yaml", "set": f"{key}={new}"}))
    b.edge(a6, update, 1.0)
    b.tool(c6, f"config/app.yaml updated: {key}: {new}")

    # --- final ask (compression happens here)
    final_ids = b.user(["Great.", "Now write the PR description."], ["goal", "goal"])
    final = final_ids[1]
    b.edge(final, fixed, 0.9)
    b.edge(final, rootcause, 0.9)
    b.edge(final, a6, 0.8)
    b.edge(final, a4, 0.6)

    questions = [
        Question("near", "What is the root cause?", ["N+1", "one SELECT per id"], [rootcause]),
        Question("deep", "What was the average per-query latency in the slow log?", [f"{lat}ms"], [log]),
        Question("deep", "What TODO comment was in the source?", ["TODO batch this"], [src]),
        Question("updated", f"What is the current value of {key}?", [new], [update], stale_node=keep),
        Question("rejected", f"Was the {cache} cache adopted?",
                 ["REJECTED", f"don't add {cache}", f"not do {cache}", f"Skip the {cache}"], [reject_head, proposal]),
        Question("constraint", "What must not be changed?", [constraint_key], [constraint]),
        Question("chatter", "Which color did the user want for the dashboard?", [color],
                 [chatter_ids["user"], chatter_ids["assistant"]]),
        Question("near", "Which file was edited?", [path], [fixed, a4]),
        Question("near", "How many tests passed?", [f"{t_pass} tests", f"{t_pass} passed"], [fixed, tests]),
        Question("instruction", "What approach replaced the cache idea?",
                 ["batched query", "Batch the query", "IN query"], [instruction, fixed]),
    ]
    return Case(seed, b.messages, b.oracle, questions, meta)
