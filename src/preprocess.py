"""データの前処理を行うモジュール。

House Prices コンペ向けに、以下の基本的な処理をまとめています。
  - 欠損値の処理(数値列・カテゴリ列)
  - 簡単な特徴量づくり(家の築年数や合計面積など)
  - カテゴリ変数のエンコード(ワンホットエンコーディング)

train と test を同じルールで処理できるように、まとめて関数化しています。
"""

import numpy as np
import pandas as pd

# 予測したい目的変数(住宅価格)
TARGET = "SalePrice"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """簡単な特徴量を追加する。

    既存の列を組み合わせて、予測に役立ちそうな新しい列を作ります。
    初学者向けに分かりやすい3つだけを用意しています。
    """
    df = df.copy()

    # 家全体の床面積(地下 + 1階 + 2階)
    if {"TotalBsmtSF", "1stFlrSF", "2ndFlrSF"}.issubset(df.columns):
        df["TotalSF"] = df["TotalBsmtSF"].fillna(0) + df["1stFlrSF"] + df["2ndFlrSF"]

    # 築年数(販売年 - 建築年)
    if {"YrSold", "YearBuilt"}.issubset(df.columns):
        df["HouseAge"] = df["YrSold"] - df["YearBuilt"]

    # バスルームの合計数(地上 + 地下、半分のバスは0.5として数える)
    bath_cols = {"FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"}
    if bath_cols.issubset(df.columns):
        df["TotalBath"] = (
            df["FullBath"].fillna(0)
            + 0.5 * df["HalfBath"].fillna(0)
            + df["BsmtFullBath"].fillna(0)
            + 0.5 * df["BsmtHalfBath"].fillna(0)
        )

    return df


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """欠損値を埋める。

    - 数値列: 中央値(median)で補完
    - カテゴリ列: 文字列 "None" で補完(「該当なし」を意味することが多いため)
    """
    df = df.copy()

    # 数値列とカテゴリ列を分ける
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna("None")

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

    # カテゴリ変数をワンホットエンコーディング(0/1 の列に変換)
    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)

    # train と test で列がずれることがあるため、列をそろえる。
    # test に無い列は 0 で埋め、test だけにある列は捨てる(train に合わせる)。
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    return X_train, y_train, X_test, test_id
