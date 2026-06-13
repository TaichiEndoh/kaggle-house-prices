"""学習ドライバ(コンペ非依存)。

  1. data/ から train/test を読み込み、外れ値除去(config のルール)
  2. 6モデルを Pipeline で OOF 評価(cross_val_predict、fold 内 fit で信頼できる CV)
  3. OOF と test 予測を data/oof_cache.npz に保存(blend.py で重み調整を瞬時にやるため)
  4. NNLS→等重み収縮のブレンドで submission.csv を書き出す

転用時に書き換えるのは config.py / features.py のみ。

使い方:
  python src/train.py
  python src/blend.py    # ← 保存済み OOF から重みだけ再調整(再学習なしで一瞬)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict

import config as cfg
from blend import blend_weights, write_submission
from models import build_models
from preprocess import remove_outliers

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_PATH = DATA_DIR / "oof_cache.npz"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise SystemExit(
            "data/train.csv または data/test.csv が見つかりません。\n"
            "先に 'python src/download_data.py' を実行してデータを取得してください。"
        )
    return pd.read_csv(train_path), pd.read_csv(test_path)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def main() -> None:
    train, test = load_data()
    print(f"train: {train.shape}, test: {test.shape}")

    train = remove_outliers(train)
    y_raw = train[cfg.TARGET].to_numpy()
    y = np.log1p(y_raw) if cfg.LOG_TARGET else y_raw
    X = train.drop(columns=[cfg.TARGET])
    test_id = test[cfg.ID].to_numpy()

    kf = KFold(n_splits=cfg.N_SPLITS, shuffle=True, random_state=cfg.RANDOM_STATE)
    models = build_models()
    names = list(models)

    print("\n--- 信頼できる交差検証 RMSE(5分割、OOF) ---")
    oof = np.zeros((len(y), len(names)))
    test_preds = np.zeros((len(test), len(names)))
    for j, name in enumerate(names):
        oof[:, j] = cross_val_predict(models[name], X, y, cv=kf, n_jobs=-1)
        print(f"  {name:7s}: {rmse(y, oof[:, j]):.4f}")
        models[name].fit(X, y)
        test_preds[:, j] = models[name].predict(test)

    # 後で blend.py から重みだけ再調整できるよう保存
    np.savez(CACHE_PATH, oof=oof, test_preds=test_preds, y=y,
             test_id=test_id, names=np.array(names))
    print(f"\nOOF を保存しました: {CACHE_PATH}（blend.py で重み再調整に使えます）")

    # ブレンドして submission を作成
    weights = blend_weights(oof, y, names, verbose=True)
    write_submission(test_preds, weights, test_id)


if __name__ == "__main__":
    main()
