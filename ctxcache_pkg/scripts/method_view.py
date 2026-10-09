"""Test the 'tool call = method(params) -> typed return; body behind a reference' idea on the trajectories."""
import glob, json, re, sys
import numpy as np
from collections import Counter, defaultdict
sys.path.insert(0, 'scripts')
from agent_parse import classify

SED = re.compile(r"sed\s+-n\s+'?(\d+),(\d+)p'?\s+(\S+)")
PY_INLINE = re.compile(r"python3?\s+-c\s+([\"'])(.*?)\1\s*$", re.S)
HEREDOC = re.compile(r"<<\s*-?\s*['\"]?(\w+)['\"]?\s*>\s*\S+\n(.*?)\n\1", re.S)
PATHISH = re.compile(r"(?:\./)?[\w.\-]+(?:/[\w.\-]+)+\.\w+")

out_by_type = Counter(); n_by_type = Counter()
run_total = run_tail = 0; run_n = 0
read_total = read_overlap_tokens = 0; reads_same_file = overlap_hits = 0
asst_total = inline_body = 0
search_out = search_list = 0; search_used = search_n = 0
fetch_cost = {5: 0, 10: 0, 20: 0}; n_edits_measured = 0; n_traj = 0
for f in sorted(glob.glob('data/minitraj/trajs/*.json')):
    d = json.load(open(f)); n_traj += 1
    msgs = d['messages']; steps = []
    for m in msgs:
        if m['role'] == 'assistant':
            acts = (m.get('extra') or {}).get('actions', [])
            cmd = acts[0]['command'] if acts else ''
            kind, keys = classify(cmd) if cmd else ('other', [])
            steps.append({'kind': kind, 'keys': keys, 'cmd': cmd, 'a': (m.get('content') or ''), 'o': ''})
        elif m['role'] == 'tool' and steps:
            steps[-1]['o'] = m.get('content') or ''
    # A. output tokens by return type
    for s in steps:
        out_by_type[s['kind']] += len(s['o']) / 4; n_by_type[s['kind']] += 1
    # B. run: rc + last 15 lines as the return value
    for s in steps:
        if s['kind'] == 'run' and s['o']:
            lines = s['o'].splitlines(); run_n += 1
            run_total += len(s['o']); run_tail += len('\n'.join(lines[-15:]))
    # C. read dedup by (file, range): re-reads overlapping an earlier range with no edit in between
    ranges = defaultdict(list)  # file -> list of (lo,hi)
    for s in steps:
        if s['kind'] == 'edit':
            for k in s['keys']: ranges.pop(k, None); ranges.pop('./' + k, None)
        if s['kind'] == 'read':
            m = SED.search(s['cmd'])
            if m:
                lo, hi, fn = int(m[1]), int(m[2]), m[3]
                read_total += len(s['o'])
                if fn in ranges:
                    reads_same_file += 1
                    cov = sum(max(0, min(hi, b) - max(lo, a) + 1) for a, b in ranges[fn])
                    if cov > 0:
                        overlap_hits += 1
                        read_overlap_tokens += len(s['o']) * min(1.0, cov / (hi - lo + 1))
                ranges[fn].append((lo, hi))
    # D. dirty pages: inline script bodies inside the call
    for s in steps:
        asst_total += len(s['a']) + len(s['cmd'])
        for rx in (PY_INLINE, HEREDOC):
            mm = rx.search(s['cmd'])
            if mm: inline_body += len(mm.group(2))
    # F. search: is the return value the hit list (paths) or the matched lines?
    for i, s in enumerate(steps):
        if s['kind'] == 'search' and s['o'] and s['cmd'].lstrip().startswith(('grep', 'find')):
            paths = set(p.lstrip('./') for p in PATHISH.findall(s['o']))
            if not paths: continue
            search_n += 1; search_out += len(s['o']); search_list += len('\n'.join(sorted(paths)))
            nxt = [k.lstrip('./') for t in steps[i + 1:i + 4] for k in t['keys']]
            if any(k in paths or any(p.endswith(k) for p in paths) for k in nxt): search_used += 1
    # E. pointer-out after K steps: how many edits would need a re-fetch
    last_read = {}
    for i, s in enumerate(steps):
        if s['kind'] in ('read', 'search'):
            for k in s['keys']: last_read[k.lstrip('./')] = i
        if s['kind'] == 'edit':
            for k in s['keys']:
                kk = k.lstrip('./')
                if kk.startswith('test_') or 'repro' in kk: continue
                if kk in last_read:
                    n_edits_measured += 1
                    for K in fetch_cost:
                        if i - last_read[kk] > K: fetch_cost[K] += 1
tot = sum(out_by_type.values())
print(f"trajs {n_traj}")
print("[A] tool-output tokens by return type:")
for k, v in out_by_type.most_common():
    print(f"    {k:8s} {v/tot:6.1%}  (calls {n_by_type[k]}, mean {v/max(1,n_by_type[k]):.0f} tok/call)")
print(f"[B] run outputs ({run_n}): keeping rc + last 15 lines keeps {run_tail/run_total:.1%} of run tokens")
print(f"[C] sed-range reads of a file already read (no edit in between): {reads_same_file}; of which overlap an earlier range {overlap_hits/reads_same_file:.1%}; overlapping tokens = {read_overlap_tokens/read_total:.1%} of all sed-read tokens")
print(f"[D] assistant-side (dirty) chars that are inline script bodies inside the call: {inline_body/asst_total:.1%}")
print(f"[F] grep/find outputs ({search_n}): distinct-path list is {search_list/search_out:.1%} of the output; a listed path is read/edited within next 3 steps in {search_used/search_n:.1%}")
print(f"[E] edits of a previously-read file ({n_edits_measured}): extra re-fetch needed if bodies are pointer-ed out after K steps -> " + ", ".join(f"K={K}: {v/n_edits_measured:.1%}" for K, v in fetch_cost.items()))
