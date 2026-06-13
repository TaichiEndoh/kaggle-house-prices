"""コンペ固有の設定。

★新しい表形式回帰コンペに転用するときは、基本このファイルと features.py だけを
  書き換えれば動く（テンプレートの心臓部）。

汎用の前処理エンジン(preprocess.py)・モデル(models.py)・学習ドライバ(train.py)は
ここの設定を読むだけなので、原則さわらなくてよい。
"""

# --- コンペ / 入出力 ---
COMPETITION = "house-prices-advanced-regression-techniques"
TARGET = "SalePrice"   # 予測する列
ID = "Id"              # 提出に使う ID 列
LOG_TARGET = True      # 目的変数を log1p して学習するか(右に歪んだ価格なので True)


def outlier_mask(df):
    """学習データから除外する外れ値(True=残す)。不要なら全 True を返す。

    House Prices 定番：居住面積が極端に広いのに激安な物件を除外。
    """
    if {"GrLivArea", TARGET}.issubset(df.columns):
        return ~((df["GrLivArea"] > 4000) & (df[TARGET] < 300000))
    return df.index == df.index  # 全 True


# --- 欠損補完の列ロール ---
# 欠損が「その設備が無い」を意味する → "None"
NONE_COLS = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "MasVnrType",
]
# 欠損が「数量ゼロ」を意味する → 0
ZERO_COLS = [
    "GarageYrBlt", "GarageArea", "GarageCars",
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath", "MasVnrArea",
]
# 欠損を最頻値で補完(最頻値は fit 時に学習)
MODE_COLS = [
    "MSZoning", "Electrical", "KitchenQual",
    "Exterior1st", "Exterior2nd", "SaleType",
]
# 欠損を固定値で補完したい列: {列名: 値}
CONST_FILL = {"Functional": "Typ"}
# 近隣などグループ別の中央値で補完: [(対象列, グループ列), ...]
GROUP_MEDIAN = [("LotFrontage", "Neighborhood")]
# 情報量が無いので削除する列
DROP_COLS = ["Utilities"]

# --- エンコード ---
# 数値だが実体はカテゴリ → 文字列化(後段でラベル/ワンホット対象になる)
TO_STRING_COLS = ["MSSubClass", "OverallCond", "YrSold", "MoSold"]
# 順序のある品質・状態カテゴリ → ラベルエンコード(対応表は fit 時に学習)
LABEL_COLS = [
    "FireplaceQu", "BsmtQual", "BsmtCond", "GarageQual", "GarageCond",
    "ExterQual", "ExterCond", "HeatingQC", "PoolQC", "KitchenQual",
    "BsmtFinType1", "BsmtFinType2", "Functional", "Fence", "BsmtExposure",
    "GarageFinish", "LandSlope", "LotShape", "PavedDrive", "Street", "Alley",
    "CentralAir", "MSSubClass", "OverallCond", "YrSold", "MoSold",
]

# --- 歪度補正(Box-Cox) ---
SKEW_THRESHOLD = 0.75   # |skew| がこれを超える数値列を変換
BOXCOX_LAMBDA = 0.15

# --- 交差検証 / ブレンド ---
N_SPLITS = 5
RANDOM_STATE = 42
# NNLS 重みを等重みへどれだけ収縮させるか(0=等重み, 1=NNLS そのまま)。過適合抑制。
BLEND_SHRINK = 0.6
