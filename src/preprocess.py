"""データの前処理を行うモジュール。

House Prices コンペ向けに、以下の処理をまとめています。
  - 外れ値の除去(学習データのみ)
  - 欠損値の処理(数値列・カテゴリ列)
  - 特徴量づくり(家の築年数や合計面積など)
  - 歪んだ数値列の対数変換(skew の大きい列を log1p)
  - カテゴリ変数のエンコード(ワンホットエンコーディング)

train と test を同じルールで処理できるように、まとめて関数化しています。
"""

import numpy as np
import pandas as pd

# 予測したい目的変数(住宅価格)
TARGET = "SalePrice"

# この値より歪度(skew)が大きい数値列は log1p で変換する
SKEW_THRESHOLD = 0.75


def remove_outliers(train: pd.DataFrame) -> pd.DataFrame:
    """学習データから明らかな外れ値を取り除く。

    House Prices コンペで有名な外れ値として、
    「居住面積(GrLivArea)が非常に大きいのに価格が安い」物件があります。
    これらは回帰モデルの学習を歪めるため除外します。
    """
    if {"GrLivArea", TARGET}.issubset(train.columns):
        mask = ~((train["GrLivArea"] > 4000) & (train[TARGET] < 300000))
        removed = (~mask).sum()
        if removed:
            print(f"外れ値を {removed} 件除去しました。")
        return train[mask].reset_index(drop=True)
    return train


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """既存の列を組み合わせて、予測に役立つ新しい列を作る。"""
    df = df.copy()

    # 家全体の床面積(地下 + 1階 + 2階)
    if {"TotalBsmtSF", "1stFlrSF", "2ndFlrSF"}.issubset(df.columns):
        df["TotalSF"] = df["TotalBsmtSF"].fillna(0) + df["1stFlrSF"] + df["2ndFlrSF"]

    # 築年数(販売年 - 建築年)とリフォームからの経過年数
    if {"YrSold", "YearBuilt"}.issubset(df.columns):
        df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    if {"YrSold", "YearRemodAdd"}.issubset(df.columns):
        df["SinceRemodel"] = df["YrSold"] - df["YearRemodAdd"]

    # バスルームの合計数(地上 + 地下、半分のバスは0.5として数える)
    bath_cols = {"FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"}
    if bath_cols.issubset(df.columns):
        df["TotalBath"] = (
            df["FullBath"].fillna(0)
            + 0.5 * df["HalfBath"].fillna(0)
            + df["BsmtFullBath"].fillna(0)
            + 0.5 * df["BsmtHalfBath"].fillna(0)
        )

    # ポーチ・デッキ面積の合計
    porch_cols = {
        "OpenPorchSF",
        "EnclosedPorch",
        "3SsnPorch",
        "ScreenPorch",
        "WoodDeckSF",
    }
    if porch_cols.issubset(df.columns):
        df["TotalPorchSF"] = sum(df[c].fillna(0) for c in porch_cols)

    # 各種設備の「あり / なし」を表すフラグ
    if "PoolArea" in df.columns:
        df["HasPool"] = (df["PoolArea"].fillna(0) > 0).astype(int)
    if "GarageArea" in df.columns:
        df["HasGarage"] = (df["GarageArea"].fillna(0) > 0).astype(int)
    if "TotalBsmtSF" in df.columns:
        df["HasBsmt"] = (df["TotalBsmtSF"].fillna(0) > 0).astype(int)
    if "Fireplaces" in df.columns:
        df["HasFireplace"] = (df["Fireplaces"].fillna(0) > 0).astype(int)

    return df


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """欠損値を埋める。

    - 数値列: 中央値(median)で補完
    - カテゴリ列: 文字列 "None" で補完(「該当なし」を意味することが多いため)
    """
    df = df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna("None")

    return df


def fix_skew(df: pd.DataFrame, skewed_cols: list[str]) -> pd.DataFrame:
    """歪んだ数値列を log1p で変換し、分布を正規分布に近づける。"""
    df = df.copy()
    for col in skewed_cols:
        if col in df.columns:
            # log1p は負値に使えないため、念のため下限を0にクリップ
            df[col] = np.log1p(df[col].clip(lower=0))
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

    # 目的変数を取り出す
    y_train = train[TARGET].copy()

    # 提出時に必要な test の Id を保持しておく
    test_id = test["Id"].copy()

    # Id と目的変数は特徴量から除外する
    X_train = train.drop(columns=[TARGET, "Id"])
    X_test = test.drop(columns=["Id"])

    # 特徴量づくり → 欠損値補完 の順で処理
    X_train = fill_missing(add_features(X_train))
    X_test = fill_missing(add_features(X_test))

    # 歪みの大きい数値列を log1p で変換する。
    # 補正対象は学習データの歪度を基準に決め、test にも同じ列を適用する。
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    skewness = X_train[numeric_cols].skew()
    skewed_cols = skewness[skewness.abs() > SKEW_THRESHOLD].index.tolist()
    X_train = fix_skew(X_train, skewed_cols)
    X_test = fix_skew(X_test, skewed_cols)

    # カテゴリ変数をワンホットエンコーディング(0/1 の列に変換)
    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)

    # train と test で列がずれることがあるため、列をそろえる。
    # test に無い列は 0 で埋め、test だけにある列は捨てる(train に合わせる)。
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    return X_train, y_train, X_test, test_id
