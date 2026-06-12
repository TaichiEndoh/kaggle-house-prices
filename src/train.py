"""モデルを学習し、提出用ファイルを作成するスクリプト。

処理の流れ:
  1. data/ から train.csv / test.csv を読み込む
  2. preprocess.py で前処理する(外れ値除去・特徴量づくり・歪度補正)
  3. 複数モデル(XGBoost / GradientBoosting / Lasso / Ridge)を学習する
  4. 交差検証(クロスバリデーション)で各モデルとブレンドの RMSE を確認する
  5. 各モデルの予測を平均(ブレンド)して submission.csv に書き出す

使い方:
  python src/train.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
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


def build_models() -> dict:
    """ブレンドに使うモデル群を作る。

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


def cv_rmse(model, X, y) -> np.ndarray:
    """交差検証で RMSE(log スケール)を計算する。"""
    neg_mse = cross_val_score(model, X, y, cv=KF, scoring="neg_mean_squared_error")
    return np.sqrt(-neg_mse)


def main() -> None:
    # 1. データ読み込み
    train, test = load_data()
    print(f"train: {train.shape}, test: {test.shape}")

    # 2. 前処理
    X_train, y_train, X_test, test_id = preprocess(train, test)
    print(f"前処理後の特徴量数: {X_train.shape[1]}")

    # 住宅価格は分布が右に偏っているため log を取ってから学習する。
    # (Kaggle の評価指標も「価格の log の RMSE」なので相性が良い)
    y_train_log = np.log1p(y_train)

    # 3. モデル定義
    models = build_models()

    # 4. 各モデルを交差検証して RMSE を確認する
    print("\n--- 交差検証 RMSE(log スケール、5分割) ---")
    for name, model in models.items():
        scores = cv_rmse(model, X_train, y_train_log)
        print(f"  {name:6s}: {scores.mean():.4f} (+/- {scores.std():.4f})")

    # ブレンド(各モデルの予測を単純平均)の交差検証スコアを手動で算出する
    blend_scores = []
    for tr_idx, va_idx in KF.split(X_train):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train_log.iloc[tr_idx], y_train_log.iloc[va_idx]
        preds = []
        for model in build_models().values():
            model.fit(X_tr, y_tr)
            preds.append(model.predict(X_va))
        blend_pred = np.mean(preds, axis=0)
        blend_scores.append(np.sqrt(np.mean((blend_pred - y_va) ** 2)))
    blend_scores = np.array(blend_scores)
    print(f"  blend : {blend_scores.mean():.4f} (+/- {blend_scores.std():.4f})  <- 提出に使用")

    # 5. 全データで学習し、test を予測する(各モデルの平均をとる)
    test_preds = []
    for model in models.values():
        model.fit(X_train, y_train_log)
        test_preds.append(model.predict(X_test))
    pred_log = np.mean(test_preds, axis=0)
    pred = np.expm1(pred_log)  # log を元のスケール(価格)に戻す

    # 6. 提出ファイルを作成する
    submission = pd.DataFrame({"Id": test_id, "SalePrice": pred})
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"\n提出ファイルを書き出しました: {SUBMISSION_PATH}")
    print(submission.head())


if __name__ == "__main__":
    main()
