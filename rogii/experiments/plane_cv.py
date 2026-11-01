import sys, numpy as np, pandas as pd
from scipy.spatial import cKDTree
df = pd.read_pickle(sys.argv[1]); R = float(sys.argv[2]); QUAD = int(sys.argv[3])
TOPS = ["ASTNU","ASTNL","EGFDU","BUDA"]
wells = np.array(sorted(df.well.unique())); rng = np.random.RandomState(42); rng.shuffle(wells)
fold_of = {w: i % 5 for i, w in enumerate(wells)}; df["fold"] = df.well.map(fold_of)
sub = df[df.row % 20 == 0]
E = {"const": [], "plane": [], "a0.5": []}; Ews=[]
for f in range(5):
    tr = sub[sub.fold != f]; XY = tr[["X","Y"]].values; tree = cKDTree(XY); T = tr[TOPS].values.mean(1)
    for w, g in df[df.fold == f].groupby("well", sort=False):
        known = g.TVT_input.notna().values; ev = ~known
        if ev.sum()==0 or known.sum()<5: continue
        x, y, z, ti = g.X.values, g.Y.values, g.Z.values, g.TVT_input.values
        ps = np.where(known)[0].max()
        pts = np.c_[x[ps:], y[ps:]][::200]
        nb = sorted(set(j for L in tree.query_ball_point(pts, R) for j in L))
        yt = g.TVT.values[ev]; cst = ti[ps]
        if len(nb) < 30:
            p = np.full(ev.sum(), cst)
        else:
            nb = np.array(nb); cx, cy = x[ps], y[ps]
            dx, dy = (XY[nb,0]-cx)/1000, (XY[nb,1]-cy)/1000
            cols = [np.ones_like(dx), dx, dy] + ([dx*dx, dy*dy, dx*dy] if QUAD else [])
            A = np.vstack(cols).T
            coef = np.linalg.lstsq(A, T[nb], rcond=None)[0]
            ex, ey = (x[ev]-cx)/1000, (y[ev]-cy)/1000
            Ae = np.vstack([np.ones_like(ex), ex, ey] + ([ex*ex, ey*ey, ex*ey] if QUAD else [])).T
            top_ps = coef[0]; top_ev = Ae @ coef
            p = cst - (z[ev] - z[ps]) + (top_ev - top_ps)
        E["const"].append(cst - yt); E["plane"].append(p - yt); E["a0.5"].append(0.5*(p+cst) - yt)
for k, v in E.items():
    e = np.concatenate(v); print(f"R={R:.0f} quad={QUAD}  {k:6s} RMSE={np.sqrt(np.mean(e**2)):.3f}")
