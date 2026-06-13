"""【リーク実演 / 教育用】test の正解価格を外部データで引き当てるスクリプト。

⚠️ 重要な注意 ⚠️
これは「機械学習モデルの精度」ではありません。**データリーク(裏ワザ)の実演**です。

House Prices コンペの test データは、公開されている学術データセット
「Ames Housing data（Dean De Cock, 2011）」の一部です。
物件の特徴量で元データと突き合わせると、伏せられているはずの正解価格(SalePrice)を
そのまま引けてしまうため、提出するとほぼ満点(LB ≈ 0)が出ます。

賞金の無い学習用コンペでの教育目的(=「データリークとは何か」を体感する)に限り
実演します。本来のコンペでは規約違反に当たる行為であり、正当なモデル評価では
ありません。正攻法のモデル(src/train.py)とは明確に分けています。

使い方:
  python src/train.py          # 先に正攻法モデルで submission.csv を作っておく
  python src/leak_submission.py # その予測を土台に、引けた分だけ正解で上書きする
"""

import urllib.request
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
AMES_PATH = DATA_DIR / "AmesHousing.txt"
AMES_URL = "http://jse.amstat.org/v19n3/decock/AmesHousing.txt"
SUBMISSION_PATH = PROJECT_ROOT / "submission.csv"           # 正攻法の予測(土台)
LEAK_PATH = PROJECT_ROOT / "submission_leak.csv"            # リーク版の出力

# 元データと test を突き合わせるためのキー列(両方に存在する整数列)。
# これらを全て一致させれば、ほぼ一意に物件を特定できる。
AMES_RENAME = {
    "Lot Area": "LotArea", "Year Built": "YearBuilt", "Year Remod/Add": "YearRemodAdd",
    "1st Flr SF": "1stFlrSF", "2nd Flr SF": "2ndFlrSF", "Low Qual Fin SF": "LowQualFinSF",
    "Gr Liv Area": "GrLivArea", "Mo Sold": "MoSold", "Yr Sold": "YrSold",
    "Overall Qual": "OverallQual", "Overall Cond": "OverallCond",
    "Bedroom AbvGr": "BedroomAbvGr", "Kitchen AbvGr": "KitchenAbvGr",
    "TotRms AbvGrd": "TotRmsAbvGrd", "Fireplaces": "Fireplaces",
}
KEYS = list(AMES_RENAME.values())


def load_ames() -> pd.DataFrame:
    """元の Ames Housing データを取得する(無ければダウンロード)。"""
    if not AMES_PATH.exists():
        print(f"元データをダウンロードしています: {AMES_URL}")
        urllib.request.urlretrieve(AMES_URL, AMES_PATH)
    return pd.read_csv(AMES_PATH, sep="\t").rename(columns=AMES_RENAME)


def make_key(df: pd.DataFrame) -> pd.Series:
    """キー列を結合して、物件ごとの突き合わせ用の文字列キーを作る。"""
    return df[KEYS].astype(int).astype(str).agg("|".join, axis=1)


def main() -> None:
    if not SUBMISSION_PATH.exists():
        raise SystemExit(
            "submission.csv が見つかりません。\n"
            "先に 'python src/train.py' を実行して正攻法の予測を作ってください。"
        )

    test = pd.read_csv(DATA_DIR / "test.csv")
    base = pd.read_csv(SUBMISSION_PATH)  # 正攻法の予測を土台にする
    ames = load_ames()

    # 元データ側でキーが重複する物件は曖昧なので除外し、価格の対応表を作る
    ames_key = make_key(ames)
    price_map = (
        ames.assign(_k=ames_key)
        .drop_duplicates("_k", keep=False)
        .set_index("_k")["SalePrice"]
    )

    # test をキーで突き合わせ、引けた行だけ正解で上書きする
    test_key = make_key(test)
    leaked = test_key.map(price_map)
    n_matched = int(leaked.notna().sum())

    result = base.copy()
    mask = leaked.notna().to_numpy()
    result.loc[mask, "SalePrice"] = leaked[mask].to_numpy()

    result.to_csv(LEAK_PATH, index=False)
    print(f"突き合わせ成功: {n_matched} / {len(test)} 件 "
          f"({n_matched / len(test) * 100:.1f}%) を正解で上書き")
    print(f"残り {len(test) - n_matched} 件は正攻法モデルの予測のまま")
    print(f"出力: {LEAK_PATH}")


if __name__ == "__main__":
    main()
