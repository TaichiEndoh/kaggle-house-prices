"""坑井ログの特徴量づくり(叩き台)。

横坑井は「深度に沿った系列」なので、単純な行ごとの特徴に加えて
**深度方向の文脈**(移動平均・勾配・評価ゾーンまでの距離)や、
**typewell(垂直参照)との対比**が効きやすいと考えられる。

★ まずは効きそうなものを少しだけ入れ、必ず GroupKFold の CV で効果を確認して
   採用/不採用を決める(House Prices で確立した原則)。詳細は rogii/NOTES.md。
"""

import numpy as np
import pandas as pd

import config as cfg


def add_well_features(h: pd.DataFrame, typewell: pd.DataFrame | None) -> pd.DataFrame:
    """1坑井分(h=horizontal_well)に特徴量を追加して返す。

    h は MD で昇順に並んでいる前提(★TODO: 実データで並びを確認)。
    """
    df = h.copy()

    # 念のため MD で昇順に
    if cfg.COL_MD in df.columns:
        df = df.sort_values(cfg.COL_MD).reset_index(drop=True)

    # --- 深度方向の文脈(GR の移動平均・勾配) ---
    if cfg.COL_GR in df.columns:
        for w in (5, 15, 31):  # 窓幅は要調整
            df[f"GR_roll_mean_{w}"] = (
                df[cfg.COL_GR].rolling(w, center=True, min_periods=1).mean()
            )
            df[f"GR_roll_std_{w}"] = (
                df[cfg.COL_GR].rolling(w, center=True, min_periods=1).std().fillna(0)
            )
        df["GR_grad"] = df[cfg.COL_GR].diff().fillna(0)  # GR の勾配

    # --- 評価ゾーン(TVT_input が NaN)までの相対位置 ---
    # 既知ゾーンと未知ゾーンの境界からの距離は、補間的に効く可能性。
    if {cfg.COL_TVT_INPUT, cfg.COL_MD}.issubset(df.columns):
        known = df[cfg.COL_TVT_INPUT].notna()
        if known.any():
            last_known_md = df.loc[known, cfg.COL_MD].max()
            df["dist_from_known_md"] = df[cfg.COL_MD] - last_known_md
            # 直近の既知 TVT(前方埋め)を補助特徴に
            df["TVT_ffill"] = df[cfg.COL_TVT_INPUT].ffill()

    # --- typewell(垂直参照)由来の特徴量 ---
    # ★TODO: typewell の列を確認し、GR プロファイルの統計量などを足す。
    #   例) typewell の GR 平均/分散、対比した相関など。今は雛形のみ。
    if typewell is not None and cfg.COL_GR in typewell.columns:
        df["typewell_GR_mean"] = typewell[cfg.COL_GR].mean()
        df["typewell_GR_std"] = typewell[cfg.COL_GR].std()

    return df
