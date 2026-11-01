import sys, numpy as np, pandas as pd

def build_ref(grid, tw, hk_tvt, hk_gr):
    tw = tw.dropna(subset=["TVT","GR"]).sort_values("TVT")
    ref_tw = np.interp(grid, tw.TVT.values, tw.GR.values)
    # PS前の横坑井GRをTVTビンで平均(解像度が高い)
    m = ~np.isnan(hk_gr)
    href = np.full(len(grid), np.nan)
    if m.sum() > 20:
        b = np.clip(np.round((hk_tvt[m]-grid[0])/(grid[1]-grid[0])).astype(int), 0, len(grid)-1)
        s = np.bincount(b, hk_gr[m], len(grid)); c = np.bincount(b, None, len(grid))
        href = np.where(c > 0, s/np.maximum(c,1), np.nan)
        ok = ~np.isnan(href)
        if ok.sum() > 10:  # typewellを横坑井GRのスケールに合わせる
            A = np.vstack([ref_tw[ok], np.ones(ok.sum())]).T
            a, b0 = np.linalg.lstsq(A, href[ok], rcond=None)[0]
            ref_tw = a*ref_tw + b0
    return np.where(np.isnan(href), ref_tw, href)

def viterbi_well(g, tw, step=2, R=100.0, dz=0.5, M=2, lam=1.0, sig=15.0, win=5):
    known = g.TVT_input.notna().values; ev = np.where(~known)[0]
    ps = np.where(known)[0].max(); t0 = g.TVT_input.values[ps]
    grid = np.arange(t0 - R, t0 + R + dz/2, dz); S_ = len(grid)
    ref = build_ref(grid, tw, g.TVT_input.values[known], g.GR.values[known])
    gr = pd.Series(g.GR.values).rolling(win, center=True, min_periods=1).mean().values
    obs = ev[::step]
    cost = np.full(S_, 1e9); cost[np.argmin(np.abs(grid - t0))] = 0.0
    bp = np.zeros((len(obs), S_), dtype=np.int8)
    shifts = np.arange(-M, M+1)
    for ti, r in enumerate(obs):
        cand = np.full((len(shifts), S_), 1e18)
        for k, d in enumerate(shifts):
            if d >= 0: cand[k, d:] = cost[:S_-d] if d>0 else cost
            else: cand[k, :d] = cost[-d:]
            cand[k] += lam * d*d
        a = np.argmin(cand, 0); cost = cand[a, np.arange(S_)]; bp[ti] = shifts[a]
        if not np.isnan(gr[r]):
            cost = cost + ((gr[r] - ref) / sig) ** 2
        cost -= cost.min()
    s = int(np.argmin(cost)); path = np.empty(len(obs), dtype=int)
    for ti in range(len(obs)-1, -1, -1):
        path[ti] = s; s = s - int(bp[ti, s])
    md = g.MD.values
    return np.interp(md[ev], md[obs], grid[path])

if __name__ == "__main__":
    df = pd.read_pickle(sys.argv[1]); D = sys.argv[2]; N = int(sys.argv[3])
    kw = dict(eval(sys.argv[4])) if len(sys.argv) > 4 else {}
    wells = sorted(df.well.unique())[:N]
    e_v, e_c = [], []
    for w in wells:
        g = df[df.well == w].reset_index(drop=True)
        k = g.TVT_input.notna().values
        if (~k).sum()==0 or k.sum()<5: continue
        tw = pd.read_csv(f"{D}/train/{w}__typewell.csv")
        y = g.TVT.values[~k]
        p = viterbi_well(g, tw, **kw)
        e_v.append(p - y); e_c.append(g.TVT_input.values[np.where(k)[0].max()] - y)
    ev = np.concatenate(e_v); ec = np.concatenate(e_c)
    print(f"{kw}  wells={len(e_v)}  viterbi RMSE={np.sqrt(np.mean(ev**2)):.3f}  const RMSE={np.sqrt(np.mean(ec**2)):.3f}")
