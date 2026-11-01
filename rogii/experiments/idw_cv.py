import sys, numpy as np, pandas as pd
from scipy.spatial import cKDTree
df = pd.read_pickle(sys.argv[1]); K = int(sys.argv[2]); POW = float(sys.argv[3]); NCAL = int(sys.argv[4])
TOPS = ["ASTNU","ASTNL","EGFDU","BUDA"]
wells = df.well.unique(); rng = np.random.RandomState(42); rng.shuffle(wells)
fold_of = {w: i % 5 for i, w in enumerate(wells)}
df["fold"] = df.well.map(fold_of)
sub = df[df.row % 10 == 0]
errs = []
for f in range(5):
    tr = sub[sub.fold != f]
    tree = cKDTree(tr[["X","Y"]].values); T = tr[TOPS].values
    va = df[df.fold == f]
    for w, g in va.groupby("well", sort=False):
        known = g.TVT_input.notna().values; ev = ~known
        if ev.sum()==0 or known.sum()<5: continue
        kidx = np.where(known)[0][-NCAL:]
        sel = np.concatenate([kidx, np.where(ev)[0]])
        d, i = tree.query(g[["X","Y"]].values[sel], k=K)
        wgt = 1.0 / np.maximum(d, 1.0)**POW; wgt /= wgt.sum(1, keepdims=True)
        tops_hat = np.einsum("nk,nks->ns", wgt, T[i])
        z = g.Z.values[sel]; ti = g.TVT_input.values
        nk = len(kidx)
        preds = []
        for j in range(len(TOPS)):
            Kc = np.median(ti[kidx] + z[:nk] - tops_hat[:nk, j])
            preds.append(Kc - (z[nk:] - tops_hat[nk:, j]))
        p = np.mean(preds, 0); y = g.TVT.values[ev]
        errs.append((p - y))
e = np.concatenate(errs)
print(f"K={K} pow={POW} ncal={NCAL}  CV RMSE = {np.sqrt(np.mean(e**2)):.3f}")
