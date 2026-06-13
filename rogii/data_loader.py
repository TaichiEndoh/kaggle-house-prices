"""ペアの per-well CSV を 1 行=1深度点の表に平坦化するローダ(叩き台)。

このコンペ特有の前処理。horizontal_well.csv を坑井ごとに読み、坑井IDを付けて
縦に結合し、typewell 由来の特徴量(NOTES.md 参照)を features.py で足す想定。

★ TODO: ルール同意 → データ取得 → inspect_data.py の結果で列名・正解の所在を確定し、
   下の TODO を埋める。詳細は rogii/NOTES.md。
"""

from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg
from features import add_well_features


def _well_ids(split_dir: Path) -> list[str]:
    return sorted(p.name.replace(cfg.HORIZONTAL_SUFFIX, "")
                  for p in split_dir.glob(f"*{cfg.HORIZONTAL_SUFFIX}"))


def _load_one(split_dir: Path, well_id: str) -> pd.DataFrame:
    """1坑井分を読み込み、typewell 特徴量を付けて返す(1行=1深度点)。"""
    h = pd.read_csv(split_dir / f"{well_id}{cfg.HORIZONTAL_SUFFIX}")
    t_path = split_dir / f"{well_id}{cfg.TYPEWELL_SUFFIX}"
    typewell = pd.read_csv(t_path) if t_path.exists() else None

    h[cfg.GROUP] = well_id                     # 坑井ID(GroupKFold 用)
    h = add_well_features(h, typewell)         # 坑井内/typewell 由来の特徴量
    return h


def load_split(split: str) -> pd.DataFrame:
    """train / test を全坑井ぶん読み、縦結合した1表を返す。"""
    split_dir = cfg.DATA_DIR / split
    if not split_dir.exists():
        raise SystemExit(
            f"{split_dir} がありません。先にデータを取得してください(NOTES.md 参照)。"
        )
    frames = [_load_one(split_dir, w) for w in _well_ids(split_dir)]
    return pd.concat(frames, ignore_index=True)


def load_train_test():
    """学習用 (X, y, groups) と 予測用 X_test, 提出キーを返す。

    評価ゾーン = TVT_input が NaN の行。
      - train: TVT 正解がある行で学習(★TODO: 正解列の確定)
      - test : 評価ゾーン(TVT_input が NaN)の行を予測

    返り値:
      X_train, y_train, groups_train, X_test, test_keys
    """
    train = load_split("train")
    test = load_split("test")

    # --- 学習データの目的変数 ---
    # ★TODO: train の TVT 正解の所在を確定する。
    #   ケースA) train では TVT_input が全行 full(=正解) → それを y にする
    #   ケースB) 別列 "TVT" に正解がある → それを y にする
    # 暫定実装(ケースA想定): TVT_input を正解とみなす。
    if cfg.TARGET in train.columns:
        y_train = train[cfg.TARGET]
    else:
        y_train = train[cfg.COL_TVT_INPUT]  # ★TODO 要確認

    groups_train = train[cfg.GROUP].to_numpy()

    # 特徴量(目的変数・リーク源・ID系は除外)。★TODO: 実データで除外列を調整
    drop = [c for c in [cfg.TARGET, cfg.COL_TVT_INPUT, *cfg.DROP_COLS]
            if c in train.columns]
    X_train = train.drop(columns=drop, errors="ignore")

    # --- テスト(評価ゾーン)の抽出 ---
    eval_mask = test[cfg.COL_TVT_INPUT].isna()
    X_test = test.loc[eval_mask].drop(columns=drop, errors="ignore")
    # ★TODO: sample_submission の行と対応づくキー(例: well_id + MD)を確認して合わせる
    test_keys = test.loc[eval_mask, [cfg.GROUP, cfg.COL_MD]].reset_index(drop=True)

    return X_train, y_train, groups_train, X_test, test_keys


if __name__ == "__main__":
    X, y, g, Xt, keys = load_train_test()
    print("X_train:", X.shape, "| y:", np.asarray(y).shape,
          "| 坑井数:", len(set(g)))
    print("X_test :", Xt.shape)
    print("特徴量列:", list(X.columns)[:20], "...")
