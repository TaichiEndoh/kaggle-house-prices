"""ROGII Wellbore Geology Prediction のコンペ固有設定(叩き台)。

★ TODO はデータ取得後(ルール同意 → ダウンロード)に実データで確定させる。
   詳細は rogii/NOTES.md を参照。
"""

from pathlib import Path

# --- コンペ / 入出力 ---
COMPETITION = "rogii-wellbore-geology-prediction"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data_rogii"   # ここに train/ test/ を展開する想定
SUBMISSION_PATH = PROJECT_ROOT / "submission_rogii.csv"

# --- 予測対象 ---
TARGET = "TVT"            # 真垂直層厚 (ft)。評価ゾーン(TVT_input が NaN の行)を予測
GROUP = "well_id"         # CV のグループ単位(坑井)。★同一坑井を跨がせない
LOG_TARGET = False        # 価格ではないので log 変換はしない想定(要検討)

# --- 評価指標(★TODO: Evaluation/Rules で確定する) ---
# 公開ノートのスコアが ft オーダー(~9.25)なので誤差系・小さいほど良いと推定。
# "mae" か "rmse" を入れる。確定するまでは暫定。
METRIC = "mae"            # ★TODO 確定: "mae" or "rmse"

# --- データ構造(判明分。★TODO: 実データで列名を確定) ---
# 各坑井: {id}__horizontal_well.csv と {id}__typewell.csv のペア
HORIZONTAL_SUFFIX = "__horizontal_well.csv"
TYPEWELL_SUFFIX = "__typewell.csv"

# horizontal_well.csv の主な列(判明分)。★TODO: 実データで全列を確認
COL_MD = "MD"             # Measured Depth (ft)
COL_X = "X"               # Easting (ft)
COL_GR = "GR"             # Gamma Ray (API)
COL_GEOLOGY = "Geology"   # 地層ラベル(カテゴリ)
COL_TVT_INPUT = "TVT_input"  # TVT のコピー。評価ゾーンだけ NaN(=予測対象の目印)
# ★TODO: train で TVT 正解がどこにあるか(TVT_input が full か、別列 "TVT" か)を確認

# 特徴量から外す列(ID やリーク源)。★TODO 実データで調整
DROP_COLS = []

# --- 交差検証 / ブレンド ---
N_SPLITS = 5
RANDOM_STATE = 42
BLEND_SHRINK = 0.6
