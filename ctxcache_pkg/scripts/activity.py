"""Which references are 'active'? Predict whether a file key is touched again within the next H steps."""
import glob, json, re, sys, numpy as np
from collections import defaultdict, Counter
sys.path.insert(0,'scripts')
from agent_parse import classify
H=20
def norm(k): return k.lstrip('./')
def isfile(k): return '.' in k.split('/')[-1] and not k.startswith('<')
rows=[]; nkeys=[]; nruns=[]; raw_tok=[]; dead_at_end=[]
for f in sorted(glob.glob('data/minitraj/trajs/*.json')):
    d=json.load(open(f)); msgs=d['messages']
    task=next((m['content'] for m in msgs if m['role']=='user'),'')
    steps=[]
    for m in msgs:
        if m['role']=='assistant':
            acts=(m.get('extra') or {}).get('actions',[]); cmd=acts[0]['command'] if acts else ''
            kind,keys=classify(cmd) if cmd else ('other',[])
            steps.append({'kind':kind,'keys':[norm(k) for k in keys if isfile(k)],'cmd':cmd,'text':(m.get('content') or ''),'o':0})
        elif m['role']=='tool' and steps: steps[-1]['o']=len(m.get('content') or '')/4
    n=len(steps)
    touched=defaultdict(list)   # key -> list of step idx
    for i,s in enumerate(steps):
        for k in set(s['keys']): touched[k].append(i)
    allkeys=set(touched)
    nkeys.append(len(allkeys)); nruns.append(len({s['cmd'] for s in steps if s['kind']=='run'}))
    raw_tok.append(sum(s['o'] for s in steps))
    edited_by=defaultdict(lambda:10**9)
    for i,s in enumerate(steps):
        if s['kind']=='edit':
            for k in s['keys']: edited_by[k]=min(edited_by[k],i)
    # sample every 5th step
    for t in range(5,n-1,5):
        recent_text=' '.join(s['text']+' '+s['cmd'] for s in steps[max(0,t-2):t+1])
        for k,idx in touched.items():
            past=[i for i in idx if i<=t]
            if not past: continue
            fut=any(t<i<=t+H for i in idx)
            base=k.split('/')[-1]
            rows.append(dict(y=fut, rec=t-past[-1], freq=len(past), edited=edited_by[k]<=t,
                             named=(base in recent_text), intask=(base in task), kind_last=steps[past[-1]]['kind']))
    for k,idx in touched.items(): dead_at_end.append(n-1-idx[-1])
R=rows; N=len(R)
def rate(mask): 
    m=[r for r in R if mask(r)]; return (np.mean([r['y'] for r in m]) if m else float('nan')), len(m)
print(f"trajs 441 | distinct file keys per traj median {np.median(nkeys):.0f} p90 {np.percentile(nkeys,90):.0f} | distinct run cmds median {np.median(nruns):.0f}")
print(f"ref table cost if 25 tok/key: median {25*np.median(nkeys):.0f} tok vs raw obs median {np.median(raw_tok):.0f} tok  ({25*np.median(nkeys)/np.median(raw_tok):.1%})")
print(f"\nP(key touched again within {H} steps) — base rate {np.mean([r['y'] for r in R]):.1%}  (n={N} key-snapshots)")
print("by steps since last touch:")
for lo,hi in ((0,2),(3,5),(6,10),(11,20),(21,40),(41,999)):
    p,c=rate(lambda r:lo<=r['rec']<=hi); print(f"   {lo:3d}-{hi:<3d}: {p:6.1%}  (n={c})")
print("by number of past touches:")
for lo,hi in ((1,1),(2,2),(3,4),(5,999)):
    p,c=rate(lambda r:lo<=r['freq']<=hi); print(f"   {lo}-{hi:<3}: {p:6.1%}  (n={c})")
for name,fn in (("already edited",lambda r:r['edited']),("not edited",lambda r:not r['edited']),
                ("named in last 3 agent msgs/cmds",lambda r:r['named']),("not named",lambda r:not r['named']),
                ("named in task description",lambda r:r['intask']),("not in task",lambda r:not r['intask'])):
    p,c=rate(fn); print(f"{name:34s}: {p:6.1%} (n={c})")
print("combos:")
for name,fn in (("rec<=5 & named",lambda r:r['rec']<=5 and r['named']),("rec<=5 & not named",lambda r:r['rec']<=5 and not r['named']),
                ("rec>20 & edited",lambda r:r['rec']>20 and r['edited']),("rec>20 & not edited & freq==1",lambda r:r['rec']>20 and not r['edited'] and r['freq']==1),
                ("rec>20 & named",lambda r:r['rec']>20 and r['named'])):
    p,c=rate(fn); print(f"   {name:30s}: {p:6.1%} (n={c})")
# simple score: coverage vs kept
print("\nkeep bodies whose key satisfies rule -> share of future touches covered / share of key-snapshots kept:")
tot_y=sum(r['y'] for r in R)
for name,fn in (("rec<=10",lambda r:r['rec']<=10),("rec<=10 or edited",lambda r:r['rec']<=10 or r['edited']),
                ("rec<=10 or edited or named",lambda r:r['rec']<=10 or r['edited'] or r['named']),
                ("rec<=10 or edited or named or freq>=3",lambda r:r['rec']<=10 or r['edited'] or r['named'] or r['freq']>=3),("rec<=20",lambda r:r['rec']<=20)):
    m=[r for r in R if fn(r)]; print(f"   {name:40s}: covers {sum(r['y'] for r in m)/tot_y:6.1%}, keeps {len(m)/N:6.1%}")
d=np.array(dead_at_end); print(f"\nat trajectory end, steps since a key's last touch: median {np.median(d):.0f}; keys never touched again after their first 10 steps: {np.mean(d>10):.0%}")
