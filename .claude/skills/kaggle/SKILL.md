---
name: kaggle
description: >-
  Run the end-to-end Kaggle workflow for this repository: check the environment
  and network, install dependencies, download competition data, train the model
  with cross-validation, write submission.csv, and (when asked) submit to Kaggle
  and report the leaderboard score. Use whenever the user wants to "run the
  baseline", "出して/提出して", iterate on the model, or check the latest
  submission score for the House Prices competition.
---

# Kaggle ワークフロー (House Prices)

このリポジトリで Kaggle コンペ
`house-prices-advanced-regression-techniques` を回すための手順をまとめたスキル。
ユーザーが「ベースライン出して」「提出して」「スコア確認して」「モデル改善して」
などと言ったときに使う。

## 0. 前提の確認(毎回さっと確認する)

```bash
# Kaggle 認証(環境変数 or ~/.kaggle/kaggle.json のどちらか)
env | grep -i kaggle            # KAGGLE_USERNAME / KAGGLE_KEY
# ネットワーク到達性
curl -sS -m 10 -o /dev/null -w "pypi:%{http_code}\n"  https://pypi.org/simple/
curl -sS -m 10 -o /dev/null -w "kaggle:%{http_code}\n" https://www.kaggle.com/api/v1/
```

- どちらの認証情報も無ければ、設定方法(README 参照)を案内して止める。
- ネットワークが届かない場合は、環境のネットワークポリシーを確認するよう案内する。

## 1. 依存インストール

```bash
pip install -r requirements.txt
```

依存は `pandas / numpy / scikit-learn / xgboost / kaggle` のみ。
新しいモデルを足すとき(例: lightgbm)は requirements.txt も更新する。

## 2. データ取得

```bash
python src/download_data.py
```

`data/` に `train.csv` / `test.csv` / `sample_submission.csv` を展開する。
`data/` 配下と `*.csv` は .gitignore 済みなのでコミットされない。

## 3. 学習 + 交差検証 + submission 作成

```bash
python src/train.py
```

- 5分割の交差検証で **RMSE(log スケール)** を表示する(小さいほど良い)。
- ルート直下に `submission.csv` を書き出す(1459 行 + ヘッダ)。
- 評価指標は「価格の log の RMSE」。目的変数は `np.log1p` で学習し、
  予測は `np.expm1` で戻す。

## 4. Kaggle へ提出(ユーザーが明示的に頼んだときだけ)

```bash
kaggle competitions submit \
  -c house-prices-advanced-regression-techniques \
  -f submission.csv -m "<何をしたか + CV スコア>"
```

提出は外向きの操作。ユーザーが「提出して」と明示したときのみ実行する。
メッセージには「手法 + CV RMSE」を入れておくと履歴で追える。

## 5. スコア確認

```bash
sleep 8   # 採点に少し時間がかかる
kaggle competitions submissions \
  -c house-prices-advanced-regression-techniques | head
```

`publicScore` を読み、CV スコアと比較する。
CV と LB が大きく乖離していたら過学習/リークを疑う。

## コード構成

- `src/preprocess.py` — 外れ値除去・欠損補完・特徴量づくり・歪度補正(log1p)・
  ワンホットエンコード・train/test の列そろえ。
- `src/train.py` — モデル定義、交差検証、ブレンド、submission 書き出し。
- `src/download_data.py` — Kaggle API でデータ取得。

## スコア改善の定石(伸ばしたいと言われたら)

効果が出やすい順:

1. **外れ値除去** — `GrLivArea > 4000 かつ 価格が安い` 物件を学習から除外(実装済み)。
2. **歪度補正** — 歪んだ数値列を log1p(実装済み)。目的変数も log で学習。
3. **特徴量エンジニアリング** — 合計面積・築年数・バス数・各種フラグなど(実装済み)。
4. **スタッキング** — XGBoost + LightGBM + GradientBoosting + Lasso + Ridge を
   Lasso メタ学習器でまとめる(実装済み)。単純平均ブレンドより、弱いモデル
   (この問題では LightGBM)をメタ学習器が自動で下げてくれる分わずかに強い。
   `train.py` は blend と stack の CV を比較し、良い方を自動採用する。
5. **ハイパーパラメータ調整** — 学習率を下げて n_estimators を増やす、正則化を効かせる。

変更したら必ず `python src/train.py` で CV スコアの変化を確認してから提出する。
CV が改善しないハイパラ変更は採用しない。

### 試したが不採用(知見)

- **品質グレードの順序エンコード**(ExterQual 等を Po..Ex → 0..5)。
  直感的には効きそうだが、この特徴量セットでは one-hot のままの方が
  線形モデル(Lasso)の CV が良く、全体でも悪化したため不採用。
  `preprocess.py` に `encode_quality()` 関数だけ残してある(呼び出していない)。

## 実績(参考)

| 手法 | CV RMSE(log) | Public LB |
| --- | --- | --- |
| XGBoost 単体ベースライン | 0.1275 | 0.12727 |
| ブレンド + 外れ値除去 + 歪度補正 | 0.1084 | 0.12331 |
| スタッキング(+ LightGBM) | 0.1083 | 0.12280 |
