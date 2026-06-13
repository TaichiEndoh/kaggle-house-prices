"""データ取得後の最初の点検スクリプト。

ルール同意 → ダウンロード後に実行し、実際の列名・行数・評価ゾーンの形を確認する。
ここで分かったことを config.py / data_loader.py / features.py の TODO に反映する。

  kaggle competitions download -c rogii-wellbore-geology-prediction -p data_rogii
  unzip -o data_rogii/*.zip -d data_rogii
  python rogii/inspect_data.py
"""

from pathlib import Path

import pandas as pd

import config as cfg


def _first_pair(split_dir: Path):
    """split_dir 内の最初の坑井ペア(horizontal, typewell)のパスを返す。"""
    hs = sorted(split_dir.glob(f"*{cfg.HORIZONTAL_SUFFIX}"))
    if not hs:
        return None, None
    h = hs[0]
    well_id = h.name.replace(cfg.HORIZONTAL_SUFFIX, "")
    t = split_dir / f"{well_id}{cfg.TYPEWELL_SUFFIX}"
    return h, (t if t.exists() else None)


def main() -> None:
    for split in ["train", "test"]:
        d = cfg.DATA_DIR / split
        if not d.exists():
            print(f"[{split}] フォルダなし: {d}")
            continue
        wells = sorted(p.name.replace(cfg.HORIZONTAL_SUFFIX, "")
                       for p in d.glob(f"*{cfg.HORIZONTAL_SUFFIX}"))
        print(f"\n===== {split}: 坑井数 {len(wells)} =====")
        h, t = _first_pair(d)
        if h is not None:
            hdf = pd.read_csv(h)
            print(f"[horizontal] {h.name}  shape={hdf.shape}")
            print("  columns:", list(hdf.columns))
            print(hdf.head(3).to_string())
            if cfg.COL_TVT_INPUT in hdf.columns:
                n_nan = hdf[cfg.COL_TVT_INPUT].isna().sum()
                print(f"  {cfg.COL_TVT_INPUT} の NaN(評価ゾーン)行数: {n_nan} / {len(hdf)}")
        if t is not None:
            tdf = pd.read_csv(t)
            print(f"[typewell]  {t.name}  shape={tdf.shape}")
            print("  columns:", list(tdf.columns))

    sub = cfg.DATA_DIR / "sample_submission.csv"
    if sub.exists():
        s = pd.read_csv(sub)
        print(f"\n===== sample_submission shape={s.shape} =====")
        print("  columns:", list(s.columns))
        print(s.head(3).to_string())


if __name__ == "__main__":
    main()
