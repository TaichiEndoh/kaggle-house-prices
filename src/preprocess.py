"""汎用の表形式前処理エンジン(fold 内 fit 対応)。

設定(config.py)と特徴量(features.py)を読み込み、表形式回帰コンペ向けの定番前処理を
sklearn 変換器 `TabularPreprocessor` として提供する。

ポイント：統計量(最頻値・グループ中央値・ラベル対応表・歪み列・最終列構成)を
**fit した fold からのみ学習** するため、交差検証(CV)にリークが入らず信頼できる。

このファイルはコンペ非依存。転用時に書き換えるのは config.py / features.py のみ。
"""

import numpy as np
import pandas as pd
from scipy.special import boxcox1p
from sklearn.base import BaseEstimator, TransformerMixin

import config as cfg
from features import add_features


def remove_outliers(train: pd.DataFrame) -> pd.DataFrame:
    """学習データから外れ値を除去する(CV の外で一度だけ呼ぶ)。"""
    mask = cfg.outlier_mask(train)
    removed = int((~mask).sum())
    if removed:
        print(f"外れ値を {removed} 件除去しました。")
    return train[mask].reset_index(drop=True)


class TabularPreprocessor(BaseEstimator, TransformerMixin):
    """config 駆動の前処理変換器。

    fit で学習する統計量:
      - 最頻値(MODE_COLS)
      - グループ別中央値 と全体中央値(GROUP_MEDIAN)
      - ラベルエンコードの対応表(LABEL_COLS)
      - Box-Cox 対象の歪み列
      - ワンホット後の最終列構成
    transform はこれらを使うだけ(検証データ/test の情報は一切使わない)。
    """

    def __init__(self, feature_fn=add_features):
        self.feature_fn = feature_fn

    def _fill_and_engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """欠損補完 → 特徴量づくり → 文字列化(状態は self から参照)。"""
        df = df.copy()

        drop = [c for c in cfg.DROP_COLS if c in df.columns]
        if drop:
            df = df.drop(columns=drop)

        for col in cfg.NONE_COLS:
            if col in df.columns:
                df[col] = df[col].fillna("None")
        for col in cfg.ZERO_COLS:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        for target_col, group_col in cfg.GROUP_MEDIAN:
            if {target_col, group_col}.issubset(df.columns):
                filled = df[group_col].map(self.group_median_[target_col])
                df[target_col] = df[target_col].fillna(filled)
                df[target_col] = df[target_col].fillna(self.group_global_[target_col])

        for col, val in cfg.CONST_FILL.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)

        for col, mode_val in self.modes_.items():
            if col in df.columns:
                df[col] = df[col].fillna(mode_val)

        df = self.feature_fn(df)

        for col in cfg.TO_STRING_COLS:
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
        df = df.copy()
        for col in self.skewed_:
            if col in df.columns:
                df[col] = boxcox1p(df[col].clip(lower=0), cfg.BOXCOX_LAMBDA)
        return df

    def fit(self, X: pd.DataFrame, y=None) -> "TabularPreprocessor":
        df = X.copy()
        if cfg.ID in df.columns:
            df = df.drop(columns=[cfg.ID])

        # 補完用の統計量を学習
        self.modes_ = {c: df[c].mode()[0] for c in cfg.MODE_COLS if c in df.columns}
        self.group_median_, self.group_global_ = {}, {}
        for target_col, group_col in cfg.GROUP_MEDIAN:
            if {target_col, group_col}.issubset(df.columns):
                self.group_median_[target_col] = df.groupby(group_col)[target_col].median()
                self.group_global_[target_col] = df[target_col].median()

        d = self._fill_and_engineer(df)

        # ラベルエンコードの対応表を学習
        self.label_maps_ = {}
        for col in cfg.LABEL_COLS:
            if col in d.columns:
                uniques = sorted(d[col].astype(str).unique())
                self.label_maps_[col] = {v: i for i, v in enumerate(uniques)}
        d = self._apply_labels(d)

        # 歪み列を学習
        num_cols = d.select_dtypes(include=[np.number]).columns
        skew = d[num_cols].apply(lambda s: s.skew()).abs()
        self.skewed_ = skew[skew > cfg.SKEW_THRESHOLD].index.tolist()
        d = self._apply_boxcox(d)

        # ワンホット後の最終列構成を記録
        self.columns_ = pd.get_dummies(d).columns
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        if cfg.ID in df.columns:
            df = df.drop(columns=[cfg.ID])
        d = self._fill_and_engineer(df)
        d = self._apply_labels(d)
        d = self._apply_boxcox(d)
        d = pd.get_dummies(d)
        # 学習時の列構成にそろえる(無い列は0、余分な列は捨てる)
        return d.reindex(columns=self.columns_, fill_value=0).fillna(0)
