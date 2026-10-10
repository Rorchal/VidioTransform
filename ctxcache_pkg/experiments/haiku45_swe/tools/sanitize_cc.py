"""Keep only what ctxcache needs from Claude Code transcripts, for sharing.

Usage: python3 -I sanitize_cc.py <in_dir> <out_dir> <local_string_to_mask> [...]
Per entry keeps: type, cwd, message.{role, model, id, content, usage}. Drops attachments (session,
environment, account/org metadata, system-prompt snapshots), every other entry field, and thinking
signatures (thinking text is already empty in these transcripts). Each local string (scratch path,
project-dir slug) is replaced by a placeholder of the same length, so text lengths, and with them
every slimming ratio in the replay, stay exactly the same.
"""
import glob
import json
import os
import sys

src, dst, masks = sys.argv[1], sys.argv[2], sys.argv[3:]
os.makedirs(dst, exist_ok=True)
subs = [(m, ("/scratch" if m.startswith("/") else "-scratch").ljust(len(m), "_")) for m in sorted(masks, key=len, reverse=True)]
for f in sorted(glob.glob(os.path.join(src, "*.jsonl"))):
    out = []
    for line in open(f):
        e = json.loads(line) if line.strip() else None
        if not isinstance(e, dict) or e.get("type") not in ("user", "assistant"):
            continue
        m = e.get("message") or {}
        content = m.get("content")
        if isinstance(content, list):
            content = [dict(b, signature="") if isinstance(b, dict) and b.get("type") == "thinking" else b for b in content]
        keep = {"type": e["type"], "cwd": e.get("cwd"),
                "message": {k: v for k, v in dict(m, content=content).items() if k in ("role", "model", "id", "content", "usage")}}
        line = json.dumps(keep, ensure_ascii=False)
        for old, new in subs:
            line = line.replace(old, new)
        out.append(line)
    open(os.path.join(dst, os.path.basename(f)), "w").write("\n".join(out) + "\n")
print(len(glob.glob(os.path.join(dst, "*.jsonl"))), "transcripts written")
