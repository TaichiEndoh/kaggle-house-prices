import sys, numpy as np, pandas as pd
df = pd.read_pickle(sys.argv[1])
TOPS = ["ANCC","ASTNU","ASTNL","EGFDU","EGFDL","BUDA"]
P = {s: [] for s in TOPS}; P["mean"]=[]; ys=[]
for w, g in df.groupby("well", sort=False):
    known = g.TVT_input.notna().values; ev = ~known
    if ev.sum()==0 or known.sum()<5: continue
    kidx = np.where(known)[0][-100:]
    y = g.TVT.values[ev]; ys.append(y); z = g.Z.values; ti = g.TVT_input.values
    preds=[]
    for s in TOPS:
        top = g[s].values
        K = np.median(ti[kidx] + z[kidx] - top[kidx])
        p = K - (z[ev] - top[ev]); P[s].append(p); preds.append(p)
    P["mean"].append(np.mean(preds,axis=0))
y=np.concatenate(ys)
for k,v in P.items(): print(f"oracle {k:6s} RMSE = {np.sqrt(np.mean((np.concatenate(v)-y)**2)):.3f}")
