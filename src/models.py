"""モデル群(モデル動物園)の定義。

各モデルは「前処理 → (スケーリング) → 推定器」の sklearn Pipeline。
表形式回帰の定番である正則化線形・カーネル・勾配ブースティングをそろえている。

効率メモ:
  LightGBM は小さなデータ(数千行)に対し n_jobs=-1 だとスレッド競合で激遅になる
  (実測 392s → n_jobs=1 で 2.3s、スコアは不変)。小データでは n_jobs=1 が正解。
"""

from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor

from preprocess import TabularPreprocessor


def _linear(model):
    """前処理 → RobustScaler → 線形/カーネルモデル。"""
    return make_pipeline(TabularPreprocessor(), RobustScaler(), model)


def _tree(model):
    """前処理 → 木モデル(スケーリング不要)。"""
    return make_pipeline(TabularPreprocessor(), model)


def build_models() -> dict:
    """強くて多様な6モデルを Pipeline として返す。"""
    return {
        "lasso": _linear(Lasso(alpha=0.0005, max_iter=10000, random_state=1)),
        "enet": _linear(
            ElasticNet(alpha=0.0005, l1_ratio=0.9, max_iter=10000, random_state=3)
        ),
        "krr": _linear(
            KernelRidge(alpha=0.6, kernel="polynomial", degree=2, coef0=2.5)
        ),
        "gboost": _tree(GradientBoostingRegressor(
            n_estimators=3000, learning_rate=0.05, max_depth=4,
            max_features="sqrt", min_samples_leaf=15, min_samples_split=10,
            loss="huber", random_state=5,
        )),
        "xgb": _tree(XGBRegressor(
            n_estimators=2200, learning_rate=0.05, max_depth=3,
            min_child_weight=1.7, gamma=0.047, subsample=0.52,
            colsample_bytree=0.46, reg_alpha=0.46, reg_lambda=0.86,
            random_state=7, n_jobs=1,
        )),
        "lgb": _tree(LGBMRegressor(
            objective="regression", n_estimators=720, learning_rate=0.05,
            num_leaves=5, max_bin=55, bagging_fraction=0.8, bagging_freq=5,
            feature_fraction=0.2319, min_data_in_leaf=6,
            min_sum_hessian_in_leaf=11, bagging_seed=9, feature_fraction_seed=9,
            random_state=9, n_jobs=1, verbose=-1,  # ← 小データでは n_jobs=1 が高速
        )),
    }
