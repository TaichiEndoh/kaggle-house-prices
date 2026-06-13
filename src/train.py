"""モデルを学習し、提出用ファイルを作成するスクリプト(本格版)。

処理の流れ:
  1. data/ から train.csv / test.csv を読み込む
  2. preprocess.py で本格的な前処理をする
  3. 強くて多様な6モデルを用意する
     (Lasso / ElasticNet / KernelRidge / GradientBoosting / XGBoost / LightGBM)
  4. 交差検証で各モデルの予測(OOF)を作り、RMSE を確認する
  5. OOF をもとに最適なブレンド重みを求める(非負最小二乗 = 線形スタッキング)
  6. その重みで test を予測し、submission.csv に書き出す

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

from preprocess import preprocess

# パス設定
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SUBMISSION_PATH = PROJECT_ROOT / "submission.csv"

# 交差検証の共通設定(5分割)
KF = KFold(n_splits=5, shuffle=True, random_state=42)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """train.csv と test.csv を読み込む。"""
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"

    if not train_path.exists() or not test_path.exists():
        raise SystemExit(
            "data/train.csv または data/test.csv が見つかりません。\n"
            "先に 'python src/download_data.py' を実行してデータを取得してください。"
        )

    return pd.read_csv(train_path), pd.read_csv(test_path)


def build_models() -> dict:
    """強くて多様なベースモデル群を作る。

    線形・カーネル系(Lasso / ElasticNet / KernelRidge)はスケールの影響を
    受けやすいため、外れ値に強い RobustScaler を前段に挟む。
    ハイパーパラメータは House Prices 上位ノートで定番の値をベースにしている。
    """
    return {
        "lasso": make_pipeline(
            RobustScaler(), Lasso(alpha=0.0005, max_iter=10000, random_state=1)
        ),
        "enet": make_pipeline(
            RobustScaler(),
            ElasticNet(alpha=0.0005, l1_ratio=0.9, max_iter=10000, random_state=3),
        ),
        "krr": make_pipeline(
            RobustScaler(),
            KernelRidge(alpha=0.6, kernel="polynomial", degree=2, coef0=2.5),
        ),
        "gboost": GradientBoostingRegressor(
            n_estimators=3000,
            learning_rate=0.05,
            max_depth=4,
            max_features="sqrt",
            min_samples_leaf=15,
            min_samples_split=10,
            loss="huber",
            random_state=5,
        ),
        "xgb": XGBRegressor(
            n_estimators=2200,
            learning_rate=0.05,
            max_depth=3,
            min_child_weight=1.7,
            gamma=0.047,
            subsample=0.52,
            colsample_bytree=0.46,
            reg_alpha=0.46,
            reg_lambda=0.86,
            random_state=7,
            n_jobs=-1,
        ),
        "lgb": LGBMRegressor(
            objective="regression",
            n_estimators=720,
            learning_rate=0.05,
            num_leaves=5,
            max_bin=55,
            bagging_fraction=0.8,
            bagging_freq=5,
            feature_fraction=0.2319,
            min_data_in_leaf=6,
            min_sum_hessian_in_leaf=11,
            bagging_seed=9,
            feature_fraction_seed=9,
            random_state=9,
            n_jobs=-1,
            verbose=-1,
        ),
    }


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE(log スケール)。"""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def main() -> None:
    # 1. データ読み込み
    train, test = load_data()
    print(f"train: {train.shape}, test: {test.shape}")

    # 2. 前処理
    X_train, y_train, X_test, test_id = preprocess(train, test)
    print(f"前処理後の特徴量数: {X_train.shape[1]}")

    # 住宅価格は分布が右に偏っているため log を取ってから学習する。
    y = np.log1p(y_train).to_numpy()
    Xtr = X_train.to_numpy()
    Xte = X_test.to_numpy()

    models = build_models()
    names = list(models)

    # 3-4. 各モデルの OOF 予測(交差検証で作る未知データ相当の予測)を作る
    print("\n--- 交差検証 RMSE(log スケール、5分割、OOF) ---")
    oof = np.zeros((len(y), len(names)))      # 学習データに対する OOF 予測
    test_preds = np.zeros((len(Xte), len(names)))  # test に対する予測
    for j, name in enumerate(names):
        oof[:, j] = cross_val_predict(models[name], Xtr, y, cv=KF, n_jobs=-1)
        print(f"  {name:7s}: {rmse(y, oof[:, j]):.4f}")
        # 全データで学習し直して test を予測
        models[name].fit(Xtr, y)
        test_preds[:, j] = models[name].predict(Xte)

    # 5. ブレンド重みを決める
    # (a) 単純平均(等重み): 偏りのない素直な参考値
    equal_oof = oof.mean(axis=1)
    print(f"\n  equal blend : {rmse(y, equal_oof):.4f}")

    # (b) 非負最小二乗で最適な重みを求める(= 線形スタッキング)。
    #     OOF 予測を組み合わせて正解に最も近づく重みを学習する。
    #     ※同じ OOF で重みを決めるため、やや楽観的な値が出る点に注意。
    weights, _ = nnls(oof, y)
    if weights.sum() > 0:
        weights = weights / weights.sum()  # 合計1に正規化(解釈しやすく)
    blend_oof = oof @ weights
    print(f"  nnls blend  : {rmse(y, blend_oof):.4f}  (やや楽観的)")
    print("  重み:", {n: round(float(w), 3) for n, w in zip(names, weights)})

    # 6. 最適重みで test を予測して提出ファイルを作成
    pred_log = test_preds @ weights
    pred = np.expm1(pred_log)
    submission = pd.DataFrame({"Id": test_id, "SalePrice": pred})
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"\n提出ファイルを書き出しました: {SUBMISSION_PATH}")
    print(submission.head())


if __name__ == "__main__":
    main()
