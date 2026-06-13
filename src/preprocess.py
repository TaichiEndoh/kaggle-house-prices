"""データの前処理を行うモジュール(本格版)。

House Prices コンペで上位ノートが使う定番の前処理をまとめている。
  - 外れ値の除去(学習データのみ)
  - 列ごとの意味を踏まえた欠損値補完(「設備なし」は None、数量系は 0、ほかは最頻値)
  - 近隣(Neighborhood)別の LotFrontage 補完
  - 数値だが実体はカテゴリの列を文字列化
  - 順序のある品質・状態カテゴリのラベルエンコード
  - 合計床面積などの特徴量づくり
  - 歪んだ数値列の Box-Cox 変換
  - 残りのカテゴリ変数のワンホットエンコード

train / test を結合してから同じルールで処理し、最後に分け直す。
(列のずれを防ぎ、エンコードを揃えるため。上位ノートでも一般的なやり方。)
"""

import numpy as np
import pandas as pd
from scipy.special import boxcox1p
from sklearn.preprocessing import LabelEncoder

# 予測したい目的変数(住宅価格)
TARGET = "SalePrice"

# この値より歪度(skew)が大きい数値列は Box-Cox 変換する
SKEW_THRESHOLD = 0.75
BOXCOX_LAMBDA = 0.15

# 欠損が「その設備が無い」を意味する列 → "None" で補完
NONE_COLS = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "MasVnrType",
]
# 欠損が「数量ゼロ」を意味する列 → 0 で補完
ZERO_COLS = [
    "GarageYrBlt", "GarageArea", "GarageCars",
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath", "MasVnrArea",
]
# 欠損を最頻値で補完する列
MODE_COLS = [
    "MSZoning", "Electrical", "KitchenQual",
    "Exterior1st", "Exterior2nd", "SaleType",
]
# 数値で入っているが実体はカテゴリ(順序に意味が無い)列 → 文字列化
TO_STRING_COLS = ["MSSubClass", "OverallCond", "YrSold", "MoSold"]
# 順序のある品質・状態カテゴリ → ラベルエンコード(整数化)
LABEL_COLS = [
    "FireplaceQu", "BsmtQual", "BsmtCond", "GarageQual", "GarageCond",
    "ExterQual", "ExterCond", "HeatingQC", "PoolQC", "KitchenQual",
    "BsmtFinType1", "BsmtFinType2", "Functional", "Fence", "BsmtExposure",
    "GarageFinish", "LandSlope", "LotShape", "PavedDrive", "Street", "Alley",
    "CentralAir", "MSSubClass", "OverallCond", "YrSold", "MoSold",
]


def remove_outliers(train: pd.DataFrame) -> pd.DataFrame:
    """学習データから明らかな外れ値(広いのに激安な物件)を取り除く。"""
    if {"GrLivArea", TARGET}.issubset(train.columns):
        mask = ~((train["GrLivArea"] > 4000) & (train[TARGET] < 300000))
        removed = int((~mask).sum())
        if removed:
            print(f"外れ値を {removed} 件除去しました。")
        return train[mask].reset_index(drop=True)
    return train


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """列の意味に応じて欠損値を補完する。"""
    df = df.copy()

    for col in NONE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    for col in ZERO_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # LotFrontage(間口)は近隣ごとの中央値で補完
    if {"LotFrontage", "Neighborhood"}.issubset(df.columns):
        df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
            lambda s: s.fillna(s.median())
        )

    # Functional は欠損なら "Typ"(標準的)とみなす
    if "Functional" in df.columns:
        df["Functional"] = df["Functional"].fillna("Typ")

    for col in MODE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # 情報量の無い列は削除(ほぼ全行が同じ値)
    if "Utilities" in df.columns:
        df = df.drop(columns=["Utilities"])

    # 取りこぼした欠損の保険(数値=中央値 / カテゴリ="None")
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype.kind in "biufc":
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna("None")

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """予測に役立つ特徴量を追加する。"""
    df = df.copy()

    # 家全体の床面積(地下 + 1階 + 2階)。最も効く特徴量の一つ。
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
    if "PoolArea" in df.columns:
        df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    if "GarageArea" in df.columns:
        df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    if "TotalBsmtSF" in df.columns:
        df["HasBsmt"] = (df["TotalBsmtSF"] > 0).astype(int)
    if "Fireplaces" in df.columns:
        df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
    if "2ndFlrSF" in df.columns:
        df["Has2ndFloor"] = (df["2ndFlrSF"] > 0).astype(int)

    # (主要ドライバの多項式・交互作用特徴も試したが、Box-Cox 済みの
    #  元特徴と冗長で CV が悪化したため不採用。)

    return df


def to_string(df: pd.DataFrame) -> pd.DataFrame:
    """数値だが実体はカテゴリの列を文字列に変換する。"""
    df = df.copy()
    for col in TO_STRING_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


def label_encode(df: pd.DataFrame) -> pd.DataFrame:
    """順序のある品質・状態カテゴリを整数に変換する。"""
    df = df.copy()
    for col in LABEL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    return df


def fix_skew(df: pd.DataFrame) -> pd.DataFrame:
    """歪んだ数値列を Box-Cox 変換して分布を正規分布に近づける。"""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    skewness = df[numeric_cols].apply(lambda s: s.skew()).abs()
    skewed_cols = skewness[skewness > SKEW_THRESHOLD].index
    for col in skewed_cols:
        df[col] = boxcox1p(df[col].clip(lower=0), BOXCOX_LAMBDA)
    return df


def preprocess(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """train / test をまとめて前処理する。

    返り値:
        X_train: 学習用の特徴量
        y_train: 学習用の目的変数(SalePrice)
        X_test : 予測用の特徴量(train と列をそろえたもの)
        test_id: 提出用に使う test 側の Id 列
    """
    # 外れ値を除去(学習データのみ)
    train = remove_outliers(train)

    y_train = train[TARGET].copy()
    test_id = test["Id"].copy()
    n_train = len(train)

    # train / test を結合して同じルールで処理する
    all_df = pd.concat(
        [train.drop(columns=[TARGET]), test], axis=0, ignore_index=True
    )
    all_df = all_df.drop(columns=["Id"])

    # 欠損補完 → 特徴量づくり → 文字列化 → ラベルエンコード
    #   → Box-Cox → ワンホット の順
    all_df = fill_missing(all_df)
    all_df = add_features(all_df)
    all_df = to_string(all_df)
    all_df = label_encode(all_df)
    all_df = fix_skew(all_df)
    all_df = pd.get_dummies(all_df)

    # train / test に分け直す
    X_train = all_df.iloc[:n_train].reset_index(drop=True)
    X_test = all_df.iloc[n_train:].reset_index(drop=True)

    return X_train, y_train, X_test, test_id
