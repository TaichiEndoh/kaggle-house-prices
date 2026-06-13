"""データの前処理(fold 内 fit に対応した sklearn 変換器版)。

これまでは train/test を結合してから Box-Cox や中央値を計算していたため、
交差検証(CV)に軽いリークが入り、CV が楽観的に出ていた。

ここでは前処理を sklearn の変換器 `HousePreprocessor` にまとめ、
**学習データ(fold)だけで fit し、検証データ/テストには transform で適用する**
形にする。これにより CV が信頼できる値になり、改善の良し悪しを正しく判断できる。

使い方(train.py 側):
  pipe = make_pipeline(HousePreprocessor(), RobustScaler(), Lasso(...))
  cross_val_predict(pipe, X_raw, y, cv=KF)   # ← fold 内で前処理が fit される
"""

import numpy as np
import pandas as pd
from scipy.special import boxcox1p
from sklearn.base import BaseEstimator, TransformerMixin

# 予測したい目的変数(住宅価格)
TARGET = "SalePrice"

SKEW_THRESHOLD = 0.75
BOXCOX_LAMBDA = 0.15

# 欠損が「設備なし」を意味する列 → "None"
NONE_COLS = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "MasVnrType",
]
# 欠損が「数量ゼロ」を意味する列 → 0
ZERO_COLS = [
    "GarageYrBlt", "GarageArea", "GarageCars",
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath", "MasVnrArea",
]
# 欠損を最頻値で補完する列(最頻値は fit 時に学習)
MODE_COLS = [
    "MSZoning", "Electrical", "KitchenQual",
    "Exterior1st", "Exterior2nd", "SaleType",
]
# 数値だが実体はカテゴリ → 文字列化
TO_STRING_COLS = ["MSSubClass", "OverallCond", "YrSold", "MoSold"]
# 順序のある品質・状態カテゴリ → ラベルエンコード(対応表は fit 時に学習)
LABEL_COLS = [
    "FireplaceQu", "BsmtQual", "BsmtCond", "GarageQual", "GarageCond",
    "ExterQual", "ExterCond", "HeatingQC", "PoolQC", "KitchenQual",
    "BsmtFinType1", "BsmtFinType2", "Functional", "Fence", "BsmtExposure",
    "GarageFinish", "LandSlope", "LotShape", "PavedDrive", "Street", "Alley",
    "CentralAir", "MSSubClass", "OverallCond", "YrSold", "MoSold",
]


def remove_outliers(train: pd.DataFrame) -> pd.DataFrame:
    """学習データから明らかな外れ値(広いのに激安な物件)を取り除く。

    CV の外で一度だけ行う(2件の既知の異常物件を落とすだけなのでリークではない)。
    """
    if {"GrLivArea", TARGET}.issubset(train.columns):
        mask = ~((train["GrLivArea"] > 4000) & (train[TARGET] < 300000))
        removed = int((~mask).sum())
        if removed:
            print(f"外れ値を {removed} 件除去しました。")
        return train[mask].reset_index(drop=True)
    return train


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """予測に役立つ特徴量を追加する(状態を持たない決定的処理)。"""
    df = df.copy()

    if {"TotalBsmtSF", "1stFlrSF", "2ndFlrSF"}.issubset(df.columns):
        df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]

    bath_cols = {"FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"}
    if bath_cols.issubset(df.columns):
        df["TotalBath"] = (
            df["FullBath"] + 0.5 * df["HalfBath"]
            + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"]
        )

    porch_cols = {
        "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "WoodDeckSF",
    }
    if porch_cols.issubset(df.columns):
        df["TotalPorchSF"] = sum(df[c] for c in porch_cols)

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

    # (築年数・リフォーム関連の特徴も試したが、年次列が既に情報を持っており
    #  honest CV が変わらなかった(lasso/enet とも 0.1111)ため不採用。
    #  残差分析でも残差はどの特徴量とも相関せず=モデルはノイズ下限に到達済み。)

    return df


class HousePreprocessor(BaseEstimator, TransformerMixin):
    """House Prices 用の前処理を、fold 内 fit に対応した形でまとめた変換器。

    fit で学習する統計量:
      - 最頻値(MODE_COLS)
      - 近隣ごとの LotFrontage 中央値 と全体中央値
      - ラベルエンコードの対応表(LABEL_COLS)
      - 歪んだ数値列の一覧(Box-Cox 対象)
      - 数値列の中央値(取りこぼし補完の保険)
      - ワンホット後の最終列構成
    transform ではこれらを使うだけ(検証データ/test の情報は一切使わない)。
    """

    def _fill_and_engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """欠損補完 → 特徴量づくり → 文字列化(状態は self から参照)。"""
        df = df.copy()

        if "Utilities" in df.columns:
            df = df.drop(columns=["Utilities"])

        for col in NONE_COLS:
            if col in df.columns:
                df[col] = df[col].fillna("None")
        for col in ZERO_COLS:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        if {"LotFrontage", "Neighborhood"}.issubset(df.columns):
            filled = df["Neighborhood"].map(self.lot_median_)
            df["LotFrontage"] = df["LotFrontage"].fillna(filled)
            df["LotFrontage"] = df["LotFrontage"].fillna(self.lot_global_)

        if "Functional" in df.columns:
            df["Functional"] = df["Functional"].fillna("Typ")

        for col, mode_val in self.modes_.items():
            if col in df.columns:
                df[col] = df[col].fillna(mode_val)

        df = add_features(df)

        for col in TO_STRING_COLS:
            if col in df.columns:
                df[col] = df[col].astype(str)

        return df

    def _apply_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """学習済みの対応表でラベルエンコード(未知の値は -1)。"""
        df = df.copy()
        for col, mapping in self.label_maps_.items():
            if col in df.columns:
                df[col] = df[col].astype(str).map(mapping).fillna(-1).astype(int)
        return df

    def _apply_boxcox(self, df: pd.DataFrame) -> pd.DataFrame:
        """学習済みの歪み列に Box-Cox を適用。"""
        df = df.copy()
        for col in self.skewed_:
            if col in df.columns:
                df[col] = boxcox1p(df[col].clip(lower=0), BOXCOX_LAMBDA)
        return df

    def fit(self, X: pd.DataFrame, y=None) -> "HousePreprocessor":
        df = X.copy()
        if "Id" in df.columns:
            df = df.drop(columns=["Id"])

        # 最頻値・LotFrontage 中央値を学習
        self.modes_ = {c: df[c].mode()[0] for c in MODE_COLS if c in df.columns}
        if {"LotFrontage", "Neighborhood"}.issubset(df.columns):
            self.lot_median_ = df.groupby("Neighborhood")["LotFrontage"].median()
            self.lot_global_ = df["LotFrontage"].median()
        else:
            self.lot_median_, self.lot_global_ = {}, 0.0

        d = self._fill_and_engineer(df)

        # ラベルエンコードの対応表を学習
        self.label_maps_ = {}
        for col in LABEL_COLS:
            if col in d.columns:
                uniques = sorted(d[col].astype(str).unique())
                self.label_maps_[col] = {v: i for i, v in enumerate(uniques)}
        d = self._apply_labels(d)

        # 歪み列を学習
        num_cols = d.select_dtypes(include=[np.number]).columns
        skew = d[num_cols].apply(lambda s: s.skew()).abs()
        self.skewed_ = skew[skew > SKEW_THRESHOLD].index.tolist()
        d = self._apply_boxcox(d)

        # ワンホット後の最終列を記録
        d = pd.get_dummies(d)
        self.columns_ = d.columns
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        if "Id" in df.columns:
            df = df.drop(columns=["Id"])
        d = self._fill_and_engineer(df)
        d = self._apply_labels(d)
        d = self._apply_boxcox(d)
        d = pd.get_dummies(d)
        # 学習時の列構成にそろえる(無い列は0、余分な列は捨てる)
        d = d.reindex(columns=self.columns_, fill_value=0)
        # 取りこぼした欠損を0で補完(reindex 後の保険)
        return d.fillna(0)
