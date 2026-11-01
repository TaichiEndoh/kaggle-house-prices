import sys, numpy as np, pandas as pd
from scipy.spatial import cKDTree
sys.path.insert(0, sys.argv[3])
from feats import well_features, TOPS
df = pd.read_pickle(sys.argv[1]); OUT = sys.argv[2]
wells = np.array(sorted(df.well.unique())); rng = np.random.RandomState(42); rng.shuffle(wells)
fold_of = {w: i % 5 for i, w in enumerate(wells)}; df["fold"] = df.well.map(fold_of)
sub = df[df.row % 20 == 0]
parts = []
for f in range(5):
    tr = sub[sub.fold != f]; XY = tr[["X","Y"]].values; tree = cKDTree(XY); T = tr[TOPS].values.mean(1)
    for w, g in df[df.fold == f].groupby("well", sort=False):
        k = g.TVT_input.notna().values
        if (~k).sum()==0 or k.sum()<5: continue
        g = g.reset_index(drop=True)
        F = well_features(g, tree, XY, T)
        F["y"] = g.TVT.values[F.row_ev.values] - F.cst.values
        F["well"] = w; F["fold"] = f
        parts.append(F)
    print("fold", f, "done", flush=True)
feat = pd.concat(parts, ignore_index=True)
for c in feat.columns:
    if feat[c].dtype == "float64": feat[c] = feat[c].astype("float32")
feat.to_pickle(OUT); print(feat.shape)
