"""LoCoMo (long-conversation memory benchmark) structural metrics.

Usage: python -I chat_metrics.py <locomo10.json>
Tokens approximated as chars/4.
"""
import json
import math
import re
import sys
from collections import Counter, defaultdict

import numpy as np

CH = 4.0
STOP = set(
    """a an the and or but if then of to in on at for with by from as is are was were be been being am do does did
    doing have has had having i you he she it we they me him her us them my your his its our their mine yours this
    that these those what which who whom whose when where why how not no yes so than too very can could will would
    shall should may might must about into over under again further once here there all any both each few more most
    other some such only own same just now also up down out off about s t ll re ve d m did does like get got go went
    going one two still ever never always really thing things way""".split()
)


def toks(s):
    return [w for w in re.findall(r"[a-z0-9']+", s.lower()) if w not in STOP and len(w) > 1]


def bm25_rank(turn_tokens, q, k1=1.2, b=0.75):
    N = len(turn_tokens)
    df = Counter()
    for t in turn_tokens:
        for w in set(t):
            df[w] += 1
    avgdl = sum(len(t) for t in turn_tokens) / max(1, N)
    scores = []
    for t in turn_tokens:
        tf = Counter(t)
        s = 0.0
        for w in q:
            if w in tf:
                idf = math.log(1 + (N - df[w] + 0.5) / (df[w] + 0.5))
                s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(t) / avgdl))
        scores.append(s)
    return np.argsort(-np.array(scores), kind="stable")


def main():
    d = json.load(open(sys.argv[1]))
    rel_pos = []
    tail_survive = defaultdict(int)
    n_qa = 0
    spans = []
    recall = defaultdict(list)
    rand_recall = defaultdict(list)
    n_turns_all = []
    n_tok_all = []
    obs_cov_hit = obs_cov_tot = 0
    obs_ratio = []
    cum_pairs = []
    ident_ev = []
    ident_non = []
    cat_counts = Counter()
    for conv in d:
        c = conv["conversation"]
        sa, sb = c["speaker_a"], c["speaker_b"]
        sess_ids = sorted(
            [k for k in c if re.fullmatch(r"session_\d+", k)], key=lambda k: int(k.split("_")[1])
        )
        turns = []  # (dia_id, session_no, text)
        for k in sess_ids:
            sn = int(k.split("_")[1])
            for t in c[k]:
                turns.append((t["dia_id"], sn, t["text"]))
        id2idx = {t[0]: i for i, t in enumerate(turns)}
        tok_len = np.array([len(t[2]) / CH for t in turns])
        cum = np.cumsum(tok_len)
        total = cum[-1]
        n_turns_all.append(len(turns))
        n_tok_all.append(total)
        turn_tokens = [toks(t[2]) for t in turns]
        speaker_words = set(toks(sa) + toks(sb))

        # observations coverage + compression
        ob = conv.get("observation", {})
        obs_ids = set()
        raw_c = 0
        obs_c = 0
        for k in sess_ids:
            sn = k.split("_")[1]
            o = ob.get(f"session_{sn}_observation", {})
            lines = [l for sp in o.values() for l in sp]
            for l in lines:
                for e in (l[1] if isinstance(l[1], list) else [l[1]]):
                    obs_ids.add(e)
            oc = sum(len(l[0]) for l in lines)
            rc = sum(len(t["text"]) for t in c[k])
            raw_c += rc
            obs_c += oc
            if rc:
                obs_ratio.append(oc / rc)
            cum_pairs.append((len(cum_pairs), raw_c / CH, obs_c / CH))

        for q in conv["qa"]:
            cat = q.get("category")
            cat_counts[cat] += 1
            if cat == 5:
                continue
            ev = [e for e in q.get("evidence", []) if e in id2idx]
            if not ev:
                continue
            n_qa += 1
            idxs = [id2idx[e] for e in ev]
            for i in idxs:
                rel_pos.append(cum[i] / total)
                obs_cov_tot += 1
                if turns[i][0] in obs_ids:
                    obs_cov_hit += 1
            for frac in (0.1, 0.25, 0.5):
                if all(cum[i] > total * (1 - frac) for i in idxs):
                    tail_survive[frac] += 1
            if len(idxs) >= 2:
                spans.append(max(turns[i][1] for i in idxs) - min(turns[i][1] for i in idxs))
            # lexical retrieval recall
            qt = toks(q["question"])
            order = bm25_rank(turn_tokens, qt)
            for k in (5, 10, 20, 50):
                top = set(order[:k].tolist())
                recall[k].append(all(i in top for i in idxs))
                rand_recall[k].append((k / len(turns)) ** len(idxs))
            # identifier overlap: capitalised words / numbers in question (excluding speaker names)
            qid = set(
                w.lower()
                for w in re.findall(r"\b(?:[A-Z][a-zA-Z]+|\d[\w]*)\b", q["question"])
            ) - speaker_words
            if qid:
                for i, t in enumerate(turns):
                    tid = set(w.lower() for w in re.findall(r"\b(?:[A-Z][a-zA-Z]+|\d[\w]*)\b", t[2])) - speaker_words
                    hit = bool(qid & tid)
                    (ident_ev if i in idxs else ident_non).append(hit)

    print("conversations:", len(d), "| turns per conv median", int(np.median(n_turns_all)), "| tokens per conv median", int(np.median(n_tok_all)))
    print("QA categories:", dict(cat_counts), "(5 = adversarial, excluded) -> evaluated QA:", n_qa)
    rp = np.array(rel_pos)
    print(f"\n[C1] evidence position (0=conversation start, 1=end): median {np.median(rp):.2f}; share in last 10% of tokens {np.mean(rp>0.9):.1%}, last 25% {np.mean(rp>0.75):.1%}, last 50% {np.mean(rp>0.5):.1%}, first 25% {np.mean(rp<0.25):.1%}")
    print("[C2] recency-only context (keep last X% of tokens): QA whose evidence fully survives:")
    for frac in (0.1, 0.25, 0.5):
        print(f"     last {int(frac*100):2d}% -> {tail_survive[frac]/n_qa:.1%}")
    sp = np.array(spans)
    print(f"[C3] multi-evidence QA: {len(sp)}; evidence spread across sessions: median {np.median(sp):.0f} sessions, >=3 sessions apart {np.mean(sp>=3):.1%}, same session {np.mean(sp==0):.1%}")
    print("[C4] lexical (BM25 over turns) retrieval recall of ALL evidence turns, vs random-pick baseline:")
    for k in (5, 10, 20, 50):
        print(f"     top-{k:2d}: {np.mean(recall[k]):.1%}   (random {np.mean(rand_recall[k]):.1%})")
    print(f"[C5] question shares a capitalised word/number (non-speaker) with: evidence turns {np.mean(ident_ev):.1%} vs non-evidence turns {np.mean(ident_non):.1%}   (n={len(ident_ev)} evidence turns, {len(ident_non)} others)")
    print(f"[C6] LLM-extracted fact lines (LoCoMo 'observation'): size vs raw dialogue median {np.median(obs_ratio):.0%}; evidence turns covered by at least one fact line {obs_cov_hit/obs_cov_tot:.1%}")
    # growth: cumulative obs vs raw within each conversation -> ratio constancy
    print("[C7] cumulative fact-store size as share of cumulative raw, by session index (all convs):")
    by_sess = defaultdict(list)
    for conv in d:
        c = conv["conversation"]
        sess_ids = sorted([k for k in c if re.fullmatch(r"session_\d+", k)], key=lambda k: int(k.split("_")[1]))
        raw = obs = 0
        for j, k in enumerate(sess_ids):
            sn = k.split("_")[1]
            o = conv.get("observation", {}).get(f"session_{sn}_observation", {})
            obs += sum(len(l[0]) for sp in o.values() for l in sp)
            raw += sum(len(t["text"]) for t in c[k])
            by_sess[j].append(obs / raw if raw else 0)
    print("     session#:  " + "  ".join(f"{j+1:2d}" for j in range(0, 19, 3)))
    print("     obs/raw :  " + "  ".join(f"{np.median(by_sess[j]):.0%}" for j in range(0, 19, 3)))


if __name__ == "__main__":
    main()
