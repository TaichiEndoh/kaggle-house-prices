"""モデルを学習し、提出用ファイルを作成するスクリプト(信頼できる CV 版)。

前処理を sklearn 変換器 `HousePreprocessor` にまとめ、各モデルを
「前処理 → (スケーリング) → モデル」の Pipeline にした。
交差検証(CV)では前処理が **fold 内でのみ fit** されるため、CV にリークが入らず
信頼できる値になる。改善の良し悪しを LB に頼らず CV で判断できる。

処理の流れ:
  1. data/ から train.csv / test.csv を読み込む(外れ値除去は CV 前に一度だけ)
  2. 6モデルそれぞれの Pipeline を OOF 予測(cross_val_predict)で評価
  3. OOF をもとに非負最小二乗(NNLS)でブレンド重みを学習
  4. その重みで test を予測し、submission.csv に書き出す

使い方:
  python src/train.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.optimize import nnls
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor

from preprocess import TARGET, HousePreprocessor, remove_outliers

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SUBMISSION_PATH = PROJECT_ROOT / "submission.csv"

KF = KFold(n_splits=5, shuffle=True, random_state=42)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise SystemExit(
            "data/train.csv または data/test.csv が見つかりません。\n"
            "先に 'python src/download_data.py' を実行してデータを取得してください。"
        )
    return pd.read_csv(train_path), pd.read_csv(test_path)


def _linear(model):
    """前処理 → RobustScaler → 線形/カーネルモデル の Pipeline。"""
    return make_pipeline(HousePreprocessor(), RobustScaler(), model)


def _tree(model):
    """前処理 → 木モデル の Pipeline(スケーリング不要)。"""
    return make_pipeline(HousePreprocessor(), model)


def build_models() -> dict:
    """強くて多様な6モデルを、それぞれ前処理込みの Pipeline として作る。"""
    return {
        "lasso": _linear(Lasso(alpha=0.0005, max_iter=10000, random_state=1)),
        "enet": _linear(
            ElasticNet(alpha=0.0005, l1_ratio=0.9, max_iter=10000, random_state=3)
        ),
        "krr": _linear(
            KernelRidge(alpha=0.6, kernel="polynomial", degree=2, coef0=2.5)
        ),
        "gboost": _tree(GradientBoostingRegressor(
            n_estimators=3000, learning_rate=0.05, max_depth=4,
            max_features="sqrt", min_samples_leaf=15, min_samples_split=10,
            loss="huber", random_state=5,
        )),
        "xgb": _tree(XGBRegressor(
            n_estimators=2200, learning_rate=0.05, max_depth=3,
            min_child_weight=1.7, gamma=0.047, subsample=0.52,
            colsample_bytree=0.46, reg_alpha=0.46, reg_lambda=0.86,
            random_state=7, n_jobs=-1,
        )),
        "lgb": _tree(LGBMRegressor(
            objective="regression", n_estimators=720, learning_rate=0.05,
            num_leaves=5, max_bin=55, bagging_fraction=0.8, bagging_freq=5,
            feature_fraction=0.2319, min_data_in_leaf=6,
            min_sum_hessian_in_leaf=11, bagging_seed=9, feature_fraction_seed=9,
            random_state=9, n_jobs=-1, verbose=-1,
        )),
    }


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def main() -> None:
    train, test = load_data()
    print(f"train: {train.shape}, test: {test.shape}")

    # 外れ値除去は CV の外で一度だけ行う
    train = remove_outliers(train)
    y = np.log1p(train[TARGET]).to_numpy()
    X = train.drop(columns=[TARGET])  # 生の特徴量(前処理は Pipeline 内で実行)
    test_id = test["Id"].copy()

    models = build_models()
    names = list(models)

    print("\n--- 信頼できる交差検証 RMSE(log スケール、5分割、OOF) ---")
    oof = np.zeros((len(y), len(names)))
    test_preds = np.zeros((len(test), len(names)))
    for j, name in enumerate(names):
        oof[:, j] = cross_val_predict(models[name], X, y, cv=KF, n_jobs=-1)
        print(f"  {name:7s}: {rmse(y, oof[:, j]):.4f}")
        models[name].fit(X, y)
        test_preds[:, j] = models[name].predict(test)

    # 等重みブレンド(最も頑健・過適合しない参考値)
    equal_w = np.full(len(names), 1.0 / len(names))
    print(f"\n  equal blend : {rmse(y, oof @ equal_w):.4f}")

    # NNLS で OOF 最適な重みを求める(ただし OOF にやや過適合する)
    nnls_w, _ = nnls(oof, y)
    if nnls_w.sum() > 0:
        nnls_w = nnls_w / nnls_w.sum()
    print(f"  nnls blend  : {rmse(y, oof @ nnls_w):.4f}  (OOF に過適合気味)")

    # 過適合を抑えるため NNLS 重みを等重みへ収縮させる(スタッキングの正則化)。
    # OOF だけに最適化しすぎず、木モデルの多様性も残すことで LB での汎化を狙う。
    SHRINK = 0.6  # NNLS をどれだけ信じるか(0=等重み, 1=NNLS そのまま)
    weights = SHRINK * nnls_w + (1 - SHRINK) * equal_w
    print(f"  shrunk blend: {rmse(y, oof @ weights):.4f}  <- 提出に使用")
    print("  重み:", {n: round(float(w), 3) for n, w in zip(names, weights)})

    pred = np.expm1(test_preds @ weights)
    submission = pd.DataFrame({"Id": test_id, "SalePrice": pred})
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"\n提出ファイルを書き出しました: {SUBMISSION_PATH}")
    print(submission.head())


if __name__ == "__main__":
    main()
