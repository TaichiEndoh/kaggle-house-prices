"""コンペ固有の特徴量づくり。

★新しいコンペに転用するときは、ここの add_features を書き換える
  （または中身を空にして `return df` だけにする）。

前処理エンジン(preprocess.py)が、欠損補完の直後にこの関数を呼ぶ。
状態を持たない決定的な処理だけを書くこと(fit/transform で同じ結果になるように)。
"""

import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """既存列を組み合わせた特徴量を追加する(House Prices 用)。"""
    df = df.copy()

    # 家全体の床面積(地下 + 1階 + 2階)。最重要特徴の一つ。
    if {"TotalBsmtSF", "1stFlrSF", "2ndFlrSF"}.issubset(df.columns):
        df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]

    # バスルームの合計数(半分のバスは 0.5)
    bath_cols = {"FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"}
    if bath_cols.issubset(df.columns):
        df["TotalBath"] = (
            df["FullBath"] + 0.5 * df["HalfBath"]
            + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"]
        )

    # ポーチ・デッキ面積の合計
    porch_cols = {
        "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "WoodDeckSF",
    }
    if porch_cols.issubset(df.columns):
        df["TotalPorchSF"] = sum(df[c] for c in porch_cols)

    # 各種設備の「あり / なし」フラグ
    for col, flag in [
        ("PoolArea", "HasPool"), ("GarageArea", "HasGarage"),
        ("TotalBsmtSF", "HasBsmt"), ("Fireplaces", "HasFireplace"),
        ("2ndFlrSF", "Has2ndFloor"),
    ]:
        if col in df.columns:
            df[flag] = (df[col] > 0).astype(int)

    return df
