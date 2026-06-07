"""ベースラインモデルを学習し、提出用ファイルを作成するスクリプト。

処理の流れ:
  1. data/ から train.csv / test.csv を読み込む
  2. preprocess.py で前処理する
  3. XGBoost で住宅価格を予測するモデルを学習する
  4. 交差検証(クロスバリデーション)でスコア(RMSE)を確認する
  5. test.csv の予測結果を submission.csv に書き出す

使い方:
  python src/train.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor

from preprocess import preprocess

# パス設定
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SUBMISSION_PATH = PROJECT_ROOT / "submission.csv"


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


def main() -> None:
    # 1. データ読み込み
    train, test = load_data()
    print(f"train: {train.shape}, test: {test.shape}")

    # 2. 前処理
    X_train, y_train, X_test, test_id = preprocess(train, test)

    # 住宅価格は分布が右に偏っているため、log を取ってから学習すると安定する。
    # (Kaggle の評価指標も「価格の log の RMSE」なので相性が良い)
    y_train_log = np.log1p(y_train)

    # 3. モデル定義(XGBoost の回帰モデル)
    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    # 4. 交差検証でスコアを確認(5分割)
    #    評価指標は RMSE(値が小さいほど良い)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    neg_mse = cross_val_score(
        model, X_train, y_train_log, cv=kf, scoring="neg_mean_squared_error"
    )
    rmse_scores = np.sqrt(-neg_mse)
    print(f"交差検証 RMSE(log スケール): {rmse_scores.mean():.4f} (+/- {rmse_scores.std():.4f})")

    # 5. 全データで学習し、test を予測する
    model.fit(X_train, y_train_log)
    pred_log = model.predict(X_test)
    pred = np.expm1(pred_log)  # log を元のスケール(価格)に戻す

    # 6. 提出ファイルを作成する
    submission = pd.DataFrame({"Id": test_id, "SalePrice": pred})
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"提出ファイルを書き出しました: {SUBMISSION_PATH}")
    print(submission.head())


if __name__ == "__main__":
    main()
