import sys, numpy as np, pandas as pd
df = pd.read_pickle(sys.argv[1])
res = {k: [] for k in ["const", "flat_z", "trend50", "trend200", "zdip200"]}
ys = []
for w, g in df.groupby("well", sort=False):
    known = g.TVT_input.notna().values
    ev = ~known
    if ev.sum() == 0 or known.sum() < 5: continue
    last = np.where(known)[0].max()
    md, z, tvt_in = g.MD.values, g.Z.values, g.TVT_input.values
    y = g.TVT.values[ev]; ys.append(y)
    res["const"].append(np.full(ev.sum(), tvt_in[last]))
    res["flat_z"].append(tvt_in[last] + (z[last] - z[ev]))
    for n, key in [(50, "trend50"), (200, "trend200")]:
        idx = np.where(known)[0][-n:]
        b = np.polyfit(md[idx], tvt_in[idx], 1)
        res[key].append(np.polyval(b, md[ev]))
    # TVT + Z (structure) linear in MD over last 200 known pts -> dip-aware
    idx = np.where(known)[0][-200:]
    b = np.polyfit(md[idx], tvt_in[idx] + z[idx], 1)
    res["zdip200"].append(np.polyval(b, md[ev]) - z[ev])
y = np.concatenate(ys)
for k, v in res.items():
    p = np.concatenate(v); print(f"{k:9s} RMSE = {np.sqrt(np.mean((p-y)**2)):.3f}")
