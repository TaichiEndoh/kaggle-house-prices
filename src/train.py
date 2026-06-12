"""モデルを学習し、提出用ファイルを作成するスクリプト。

処理の流れ:
  1. data/ から train.csv / test.csv を読み込む
  2. preprocess.py で前処理する(外れ値除去・特徴量づくり・品質順序エンコード・歪度補正)
  3. 複数モデル(XGBoost / LightGBM / GradientBoosting / Lasso / Ridge)を学習する
  4. 交差検証で各モデル・ブレンド・スタッキングの RMSE を比較する
  5. CV が最も良い方式で test を予測し、submission.csv に書き出す

使い方:
  python src/train.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import KFold, cross_val_score
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

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def build_base_models() -> dict:
    """ブレンド / スタッキングに使うベースモデル群を作る。

    線形モデル(Lasso / Ridge)はスケールの影響を受けやすいため、
    外れ値に強い RobustScaler を前段に挟んだパイプラインにする。
    """
    return {
        "xgb": XGBRegressor(
            n_estimators=2000,
            learning_rate=0.02,
            max_depth=3,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
        ),
        "lgb": LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.02,
            num_leaves=15,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
        "gbr": GradientBoostingRegressor(
            n_estimators=1500,
            learning_rate=0.02,
            max_depth=3,
            subsample=0.8,
            random_state=42,
        ),
        "lasso": make_pipeline(RobustScaler(), Lasso(alpha=0.0005, max_iter=10000)),
        "ridge": make_pipeline(RobustScaler(), Ridge(alpha=10.0)),
    }


def build_stack() -> StackingRegressor:
    """ベースモデルを Lasso メタモデルでまとめるスタッキングモデルを作る。"""
    base = list(build_base_models().items())
    return StackingRegressor(
        estimators=base,
        final_estimator=make_pipeline(
            RobustScaler(), Lasso(alpha=0.0005, max_iter=10000)
        ),
        cv=KF,
        n_jobs=-1,
    )


def cv_rmse(model, X, y) -> np.ndarray:
    """交差検証で RMSE(log スケール)を計算する。"""
    neg_mse = cross_val_score(model, X, y, cv=KF, scoring="neg_mean_squared_error")
    return np.sqrt(-neg_mse)


def blend_cv(X, y) -> np.ndarray:
    """ベースモデルの単純平均(ブレンド)の交差検証 RMSE を計算する。"""
    scores = []
    for tr_idx, va_idx in KF.split(X):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        preds = []
        for model in build_base_models().values():
            model.fit(X_tr, y_tr)
            preds.append(model.predict(X_va))
        blend_pred = np.mean(preds, axis=0)
        scores.append(np.sqrt(np.mean((blend_pred - y_va) ** 2)))
    return np.array(scores)


def blend_predict(X_train, y_train, X_test) -> np.ndarray:
    """全データで各ベースモデルを学習し、平均した予測(log スケール)を返す。"""
    preds = []
    for model in build_base_models().values():
        model.fit(X_train, y_train)
        preds.append(model.predict(X_test))
    return np.mean(preds, axis=0)


def main() -> None:
    # 1. データ読み込み
    train, test = load_data()
    print(f"train: {train.shape}, test: {test.shape}")

    # 2. 前処理
    X_train, y_train, X_test, test_id = preprocess(train, test)
    print(f"前処理後の特徴量数: {X_train.shape[1]}")

    # 住宅価格は分布が右に偏っているため log を取ってから学習する。
    y_train_log = np.log1p(y_train)

    # 3-4. 各方式を交差検証して RMSE を比較する
    print("\n--- 交差検証 RMSE(log スケール、5分割) ---")
    for name, model in build_base_models().items():
        s = cv_rmse(model, X_train, y_train_log)
        print(f"  {name:6s}: {s.mean():.4f} (+/- {s.std():.4f})")

    blend_s = blend_cv(X_train, y_train_log)
    print(f"  blend : {blend_s.mean():.4f} (+/- {blend_s.std():.4f})")

    stack_s = cv_rmse(build_stack(), X_train, y_train_log)
    print(f"  stack : {stack_s.mean():.4f} (+/- {stack_s.std():.4f})")

    # 5. CV が良い方を採用して test を予測する
    if stack_s.mean() <= blend_s.mean():
        print(f"\n=> stack を採用(CV {stack_s.mean():.4f})")
        stack = build_stack()
        stack.fit(X_train, y_train_log)
        pred_log = stack.predict(X_test)
    else:
        print(f"\n=> blend を採用(CV {blend_s.mean():.4f})")
        pred_log = blend_predict(X_train, y_train_log, X_test)

    pred = np.expm1(pred_log)  # log を元のスケール(価格)に戻す

    # 6. 提出ファイルを作成する
    submission = pd.DataFrame({"Id": test_id, "SalePrice": pred})
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"\n提出ファイルを書き出しました: {SUBMISSION_PATH}")
    print(submission.head())


if __name__ == "__main__":
    main()
