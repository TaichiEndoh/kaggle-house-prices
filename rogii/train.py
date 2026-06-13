"""ROGII 用 学習ドライバ(叩き台) — 回帰評価への対応版。

House Prices テンプレとの違いは2点:
  1. CV を **GroupKFold(groups=坑井ID)** にする(同一坑井のリーク防止)。
  2. 評価指標を **MAE / RMSE(ft、小さいほど良い)** に切り替える(log 変換しない)。

データ取得前は動かない(雛形)。NOTES.md の手順でデータを用意してから回す。

  python rogii/train.py
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.optimize import nnls
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor

import config as cfg
from data_loader import load_train_test


def score(y_true, y_pred) -> float:
    """config.METRIC に従った誤差(小さいほど良い)。"""
    err = np.asarray(y_true) - np.asarray(y_pred)
    if cfg.METRIC == "rmse":
        return float(np.sqrt(np.mean(err ** 2)))
    return float(np.mean(np.abs(err)))  # 既定: MAE


def _prep(numeric_cols, cat_cols) -> ColumnTransformer:
    """最小限の前処理: 数値は中央値補完、カテゴリは順序エンコード。

    ★ fold 内 fit されるよう Pipeline に組み込んで使う(リーク防止)。
    """
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric_cols),
        ("cat", make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        ), cat_cols),
    ])


def build_models(numeric_cols, cat_cols) -> dict:
    """まずは欠損に強い木モデル中心(回帰)。効果を見て増やす。"""
    prep = _prep(numeric_cols, cat_cols)
    return {
        "lgb": Pipeline([("prep", prep), ("m", LGBMRegressor(
            n_estimators=2000, learning_rate=0.03, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, random_state=9,
            n_jobs=1, verbose=-1))]),   # ★小データでは n_jobs=1
        "xgb": Pipeline([("prep", prep), ("m", XGBRegressor(
            n_estimators=2000, learning_rate=0.03, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=7, n_jobs=1))]),
        "hgb": Pipeline([("prep", prep), ("m", HistGradientBoostingRegressor(
            max_iter=1500, learning_rate=0.03, max_depth=6, random_state=5))]),
    }


def main() -> None:
    X, y, groups, X_test, test_keys = load_train_test()
    y = np.asarray(y, dtype=float)
    print(f"X_train {X.shape} / 坑井 {len(set(groups))} / X_test {X_test.shape}")

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # ★ GroupKFold: 同一坑井を train/val に跨がせない(最重要)
    gkf = GroupKFold(n_splits=cfg.N_SPLITS)
    models = build_models(numeric_cols, cat_cols)
    names = list(models)

    print(f"\n--- GroupKFold CV ({cfg.METRIC}、小さいほど良い) ---")
    oof = np.zeros((len(y), len(names)))
    test_preds = np.zeros((len(X_test), len(names)))
    for j, name in enumerate(names):
        oof[:, j] = cross_val_predict(models[name], X, y, cv=gkf, groups=groups, n_jobs=-1)
        print(f"  {name:4s}: {score(y, oof[:, j]):.4f}")
        models[name].fit(X, y)
        test_preds[:, j] = models[name].predict(X_test)

    # NNLS→等重み収縮ブレンド(House Prices と同じ発想)
    equal_w = np.full(len(names), 1.0 / len(names))
    nnls_w, _ = nnls(oof, y)
    nnls_w = nnls_w / nnls_w.sum() if nnls_w.sum() > 0 else equal_w
    weights = cfg.BLEND_SHRINK * nnls_w + (1 - cfg.BLEND_SHRINK) * equal_w
    print(f"  blend: {score(y, oof @ weights):.4f}  weights="
          f"{ {n: round(float(w),3) for n,w in zip(names, weights)} }")

    # 提出(★TODO: sample_submission の行・列に合わせて整形する)
    pred = test_preds @ weights
    sub = test_keys.copy()
    sub[cfg.TARGET] = pred
    sub.to_csv(cfg.SUBMISSION_PATH, index=False)
    print(f"\n提出ファイル(暫定): {cfg.SUBMISSION_PATH}")
    print("★TODO: sample_submission.csv の形式に合わせて最終整形すること")


if __name__ == "__main__":
    main()
