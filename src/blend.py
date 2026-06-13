"""ブレンド重みの計算と submission 書き出し(コンペ非依存)。

train.py から呼ばれるほか、単体で実行すると **保存済みの OOF(data/oof_cache.npz)** を
読み込んで重みだけを再計算する。モデルの再学習が要らないので一瞬で終わり、
ブレンド方式や収縮率(config.BLEND_SHRINK)の試行錯誤を効率的に回せる。

使い方:
  python src/blend.py            # キャッシュから重みを再計算し submission を作り直す
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

import config as cfg

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_PATH = DATA_DIR / "oof_cache.npz"
SUBMISSION_PATH = PROJECT_ROOT / "submission.csv"


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def blend_weights(oof: np.ndarray, y: np.ndarray, names, verbose: bool = False) -> np.ndarray:
    """OOF から NNLS 重みを求め、過適合を抑えるため等重みへ収縮させて返す。"""
    n = oof.shape[1]
    equal_w = np.full(n, 1.0 / n)
    nnls_w, _ = nnls(oof, y)
    if nnls_w.sum() > 0:
        nnls_w = nnls_w / nnls_w.sum()
    weights = cfg.BLEND_SHRINK * nnls_w + (1 - cfg.BLEND_SHRINK) * equal_w

    if verbose:
        print(f"  equal blend : {_rmse(y, oof @ equal_w):.4f}")
        print(f"  nnls  blend : {_rmse(y, oof @ nnls_w):.4f}  (OOF に過適合気味)")
        print(f"  shrunk blend: {_rmse(y, oof @ weights):.4f}  "
              f"<- 提出に使用 (shrink={cfg.BLEND_SHRINK})")
        print("  重み:", {n_: round(float(w), 3) for n_, w in zip(names, weights)})
    return weights


def write_submission(test_preds: np.ndarray, weights: np.ndarray, test_id: np.ndarray) -> None:
    """ブレンド予測を submission.csv に書き出す。"""
    pred_log = test_preds @ weights
    pred = np.expm1(pred_log) if cfg.LOG_TARGET else pred_log
    submission = pd.DataFrame({cfg.ID: test_id, cfg.TARGET: pred})
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"提出ファイルを書き出しました: {SUBMISSION_PATH}")
    print(submission.head())


def main() -> None:
    if not CACHE_PATH.exists():
        raise SystemExit(
            "data/oof_cache.npz が見つかりません。先に 'python src/train.py' を実行してください。"
        )
    cache = np.load(CACHE_PATH, allow_pickle=True)
    oof, test_preds = cache["oof"], cache["test_preds"]
    y, test_id, names = cache["y"], cache["test_id"], [str(n) for n in cache["names"]]

    print("--- 保存済み OOF から重みを再計算 ---")
    for j, name in enumerate(names):
        print(f"  {name:7s}: {_rmse(y, oof[:, j]):.4f}")
    weights = blend_weights(oof, y, names, verbose=True)
    write_submission(test_preds, weights, test_id)


if __name__ == "__main__":
    main()
