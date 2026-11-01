"""ROGII Wellbore Geology Prediction — Late Submission 用カーネル。

方針（物理特徴 + LightGBM のハイブリッド）:
  - TVT は「地層面からの高さ」で決まる（学習データの地層面を使うと RMSE≈0.006）。
    test には地層面が無いので、周辺の学習坑井の地層面から “傾き” を推定する。
  - 推定は大げさになりがちなので、LightGBM に「どれだけ信じるか」を学習させる。
  - 目的変数 = PS(予測開始点)からの TVT のずれ。基準は PS の TVT（const）。
  - 学習用特徴量は、自分の坑井を含まない近傍（坑井単位 5 分割）だけで作る。
    test 側も同じ 5 本の木で特徴量を作り、予測を平均して分布をそろえる。

ローカル CV（坑井単位 5 分割）: const 15.91 → このモデル 11.53。
"""

import glob
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

TOPS = ["ASTNU", "ASTNL", "EGFDU", "BUDA"]
N_FOLDS = 5
SEED = 42


def find_root() -> str:
    """コンペデータのルート(train/ と sample_submission.csv がある場所)を探す。"""
    if len(sys.argv) > 1:
        return sys.argv[1]
    for p in glob.glob("/kaggle/input/**/sample_submission.csv", recursive=True):
        root = os.path.dirname(p)
        if os.path.isdir(os.path.join(root, "train")):
            return root
    raise SystemExit("competition data not found")


def load_split(root: str, split: str, with_truth: bool) -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(f"{root}/{split}/*__horizontal_well.csv")):
        w = os.path.basename(f).split("__")[0]
        h = pd.read_csv(f)
        h.insert(0, "row", np.arange(len(h)))
        h.insert(0, "well", w)
        keep = ["well", "row", "MD", "X", "Y", "Z", "TVT_input"]
        if with_truth:
            keep += TOPS + ["TVT"]
        frames.append(h[keep])
    return pd.concat(frames, ignore_index=True)


def plane_drift(XY, T, nb, cx, cy, x, y):
    dx, dy = (XY[nb, 0] - cx) / 1000, (XY[nb, 1] - cy) / 1000
    A = np.vstack([np.ones_like(dx), dx, dy]).T
    c = np.linalg.lstsq(A, T[nb], rcond=None)[0]
    return c[1] * (x - cx) / 1000 + c[2] * (y - cy) / 1000, c[1], c[2]


def well_features(g: pd.DataFrame, tree, XY, T) -> pd.DataFrame:
    """1 坑井の予測区間の各行に特徴量を作る（近傍は渡された木だけを使う）。"""
    known = g.TVT_input.notna().values
    ev = np.where(~known)[0]
    kidx = np.where(known)[0]
    ps = kidx.max()
    md, x, y, z, ti = (g[c].values for c in ["MD", "X", "Y", "Z", "TVT_input"])
    F = {"dmd": md[ev] - md[ps], "dz": z[ev] - z[ps],
         "dxy": np.hypot(x[ev] - x[ps], y[ev] - y[ps])}
    F["head_cos"] = np.where(F["dxy"] > 0, (x[ev] - x[ps]) / np.maximum(F["dxy"], 1), 0)
    F["head_sin"] = np.where(F["dxy"] > 0, (y[ev] - y[ps]) / np.maximum(F["dxy"], 1), 0)
    pts = np.c_[x[ps:], y[ps:]][::200]
    for R in (3000, 5000, 8000):
        nb = sorted(set(j for L in tree.query_ball_point(pts, R) for j in L))
        if len(nb) >= 30:
            dtop, gx, gy = plane_drift(XY, T, np.array(nb), x[ps], y[ps], x[ev], y[ev])
            F[f"plane{R}"] = dtop - F["dz"]
            F[f"gx{R}"] = np.full(len(ev), gx)
            F[f"gy{R}"] = np.full(len(ev), gy)
        else:
            F[f"plane{R}"] = np.full(len(ev), np.nan)
            F[f"gx{R}"] = np.full(len(ev), np.nan)
            F[f"gy{R}"] = np.full(len(ev), np.nan)
        F[f"nnb{R}"] = np.full(len(ev), len(nb))
    q = np.r_[ps, ev]
    d, i = tree.query(np.c_[x[q], y[q]], k=16)
    wgt = 1.0 / np.maximum(d, 1.0) ** 2
    wgt /= wgt.sum(1, keepdims=True)
    th = (wgt * T[i]).sum(1)
    F["idw"] = (th[1:] - th[0]) - F["dz"]
    F["nn_dist"] = d[1:, 0]
    for n in (100, 300):
        idx = kidx[-n:]
        b = np.polyfit(md[idx], ti[idx], 1)[0] if len(idx) > 5 else 0.0
        F[f"own_slope{n}"] = np.full(len(ev), b)
        F[f"own_ext{n}"] = b * F["dmd"]
        bs = np.polyfit(md[idx], ti[idx] + z[idx], 1)[0] if len(idx) > 5 else 0.0
        F[f"own_struct{n}"] = bs * F["dmd"] - F["dz"]
    F["n_known"] = np.full(len(ev), len(kidx))
    F["frac"] = F["dmd"] / max(md[-1] - md[ps], 1)
    out = pd.DataFrame(F)
    out["cst"] = ti[ps]
    out["row_ev"] = ev
    return out


def main() -> None:
    root = find_root()
    print("data root:", root, flush=True)
    train = load_split(root, "train", with_truth=True)
    test = load_split(root, "test", with_truth=False)
    print("train", train.shape, "test", test.shape, flush=True)

    # 坑井単位の 5 分割。各分割で「その分割を除いた坑井」の地層面点から木を作る。
    wells = np.array(sorted(train.well.unique()))
    np.random.RandomState(SEED).shuffle(wells)
    fold_of = {w: i % N_FOLDS for i, w in enumerate(wells)}
    train["fold"] = train.well.map(fold_of)
    sub = train[train.row % 20 == 0]
    trees = []
    for f in range(N_FOLDS):
        tr = sub[sub.fold != f]
        XY = tr[["X", "Y"]].values
        trees.append((cKDTree(XY), XY, tr[TOPS].values.mean(1)))

    # 学習用特徴量（自分の坑井を含まない近傍だけで作る）
    parts = []
    for w, g in train.groupby("well", sort=False):
        k = g.TVT_input.notna().values
        if (~k).sum() == 0 or k.sum() < 5:
            continue
        g = g.reset_index(drop=True)
        tree, XY, T = trees[fold_of[w]]
        F = well_features(g, tree, XY, T)
        F["y"] = g.TVT.values[F.row_ev.values] - F.cst.values
        parts.append(F[F.row_ev % 5 == 0])
    feat = pd.concat(parts, ignore_index=True)
    cols = [c for c in feat.columns if c not in ("row_ev", "y", "cst")]
    print("train features", feat.shape, flush=True)

    params = dict(objective="regression", learning_rate=0.05, num_leaves=63,
                  min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=SEED)
    model = lgb.train(params, lgb.Dataset(feat[cols], feat.y), num_boost_round=600)

    # test：5 本の木それぞれで特徴量を作って予測し、平均する
    preds = {}
    for w, g in test.groupby("well", sort=False):
        g = g.reset_index(drop=True)
        k = g.TVT_input.notna().values
        if (~k).sum() == 0:
            continue
        if k.sum() < 5:  # 既知区間が極端に短い坑井は const で埋める
            last = g.TVT_input.ffill().fillna(0).values
            for r in np.where(~k)[0]:
                preds[f"{w}_{r}"] = last[r]
            continue
        p = 0.0
        for tree, XY, T in trees:
            F = well_features(g, tree, XY, T)
            p = p + model.predict(F[cols]) / len(trees)
        for r, v in zip(F.row_ev.values, F.cst.values + p):
            preds[f"{w}_{r}"] = v

    sample = pd.read_csv(f"{root}/sample_submission.csv")
    sub_df = pd.DataFrame({"id": sample["id"]})
    sub_df["tvt"] = sub_df["id"].map(preds)
    miss = sub_df.tvt.isna().sum()
    if miss:
        print("WARNING: missing predictions", miss, flush=True)
        sub_df["tvt"] = sub_df["tvt"].fillna(0.0)
    out = "/kaggle/working/submission.csv" if os.path.isdir("/kaggle/working") else "submission_rogii.csv"
    sub_df.to_csv(out, index=False)
    print("wrote", out, sub_df.shape, flush=True)


if __name__ == "__main__":
    main()
