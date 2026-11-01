"""坑井ごとの予測区間(評価ゾーン)の行に、物理ベースの特徴量を作る。
   近傍特徴は渡された木(=自分の坑井を含まない学習坑井)だけを使う。"""
import numpy as np, pandas as pd
TOPS = ["ASTNU","ASTNL","EGFDU","BUDA"]

def plane_drift(XY, T, nb, cx, cy, x, y):
    dx, dy = (XY[nb,0]-cx)/1000, (XY[nb,1]-cy)/1000
    A = np.vstack([np.ones_like(dx), dx, dy]).T
    c = np.linalg.lstsq(A, T[nb], rcond=None)[0]
    return c[1]*(x-cx)/1000 + c[2]*(y-cy)/1000, c[1], c[2]

def well_features(g, tree, XY, T):
    known = g.TVT_input.notna().values; ev = np.where(~known)[0]
    kidx = np.where(known)[0]; ps = kidx.max()
    md, x, y, z, ti = (g[c].values for c in ["MD","X","Y","Z","TVT_input"])
    cst = ti[ps]
    F = {"dmd": md[ev]-md[ps], "dz": z[ev]-z[ps],
         "dxy": np.hypot(x[ev]-x[ps], y[ev]-y[ps])}
    # 横方向の向き(PS->各点)
    F["head_cos"] = np.where(F["dxy"]>0, (x[ev]-x[ps])/np.maximum(F["dxy"],1), 0)
    F["head_sin"] = np.where(F["dxy"]>0, (y[ev]-y[ps])/np.maximum(F["dxy"],1), 0)
    # 近傍坑井の平面(複数半径)
    pts = np.c_[x[ps:], y[ps:]][::200]
    for R in (3000, 5000, 8000):
        nb = sorted(set(j for L in tree.query_ball_point(pts, R) for j in L))
        if len(nb) >= 30:
            dtop, gx, gy = plane_drift(XY, T, np.array(nb), x[ps], y[ps], x[ev], y[ev])
            F[f"plane{R}"] = dtop - F["dz"]; F[f"gx{R}"] = np.full(len(ev), gx); F[f"gy{R}"] = np.full(len(ev), gy)
        else:
            F[f"plane{R}"] = np.full(len(ev), np.nan); F[f"gx{R}"] = F[f"gy{R}"] = np.full(len(ev), np.nan)
        F[f"nnb{R}"] = np.full(len(ev), len(nb))
    # IDW(点ごと)の地層面変化
    q = np.r_[ps, ev]
    d, i = tree.query(np.c_[x[q], y[q]], k=16)
    wgt = 1.0/np.maximum(d,1.0)**2; wgt /= wgt.sum(1, keepdims=True)
    th = (wgt*T[i]).sum(1)
    F["idw"] = (th[1:] - th[0]) - F["dz"]
    F["nn_dist"] = d[1:,0]
    # 自坑井の既知区間の直近トレンド
    for n in (100, 300):
        idx = kidx[-n:]
        b = np.polyfit(md[idx], ti[idx], 1)[0] if len(idx) > 5 else 0.0
        F[f"own_slope{n}"] = np.full(len(ev), b); F[f"own_ext{n}"] = b*F["dmd"]
        bs = np.polyfit(md[idx], ti[idx]+z[idx], 1)[0] if len(idx) > 5 else 0.0
        F[f"own_struct{n}"] = bs*F["dmd"] - F["dz"]
    F["n_known"] = np.full(len(ev), len(kidx)); F["frac"] = F["dmd"]/max(md[-1]-md[ps],1)
    out = pd.DataFrame(F); out["cst"] = cst; out["row_ev"] = ev
    return out
