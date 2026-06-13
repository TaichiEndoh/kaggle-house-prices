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

- `src/preprocess.py` — 外れ値除去・列ごとの欠損補完・近隣別 LotFrontage 補完・
  数値カテゴリの文字列化・ラベルエンコード・特徴量づくり・Box-Cox 変換・
  ワンホットエンコード(train/test を結合して処理し最後に分割)。
- `src/train.py` — 6モデル定義、OOF 予測の作成、NNLS による重み付きブレンド、
  submission 書き出し。
- `src/download_data.py` — Kaggle API でデータ取得。

## スコア改善の定石(伸ばしたいと言われたら)

効果が出やすい順:

1. **外れ値除去** — `GrLivArea > 4000 かつ 価格が安い` 物件を学習から除外(実装済み)。
2. **歪度補正** — 歪んだ数値列を log1p(実装済み)。目的変数も log で学習。
3. **特徴量エンジニアリング** — 合計面積・築年数・バス数・各種フラグなど(実装済み)。
2. **歪度補正** — 歪んだ数値列を Box-Cox(λ=0.15)で変換(実装済み)。目的変数は log。
3. **本格的な前処理**(`preprocess.py`、現行版)— train/test を結合し、列ごとの意味を
   踏まえた欠損補完(設備なし=None / 数量=0 / その他=最頻値)、近隣別 LotFrontage 補完、
   数値カテゴリの文字列化(MSSubClass / OverallCond / YrSold / MoSold)、品質・状態の
   ラベルエンコード、合計面積などの特徴量、Box-Cox、ワンホットまでを一括で行う。
4. **強い6モデルの重み付きブレンド**(`train.py`、現行版)—
   Lasso / ElasticNet / KernelRidge / GradientBoosting / XGBoost / LightGBM の
   OOF 予測(交差検証で作る未知データ相当の予測)を作り、**非負最小二乗(NNLS)で
   最適な重みを学習する線形スタッキング**。等重みブレンドの参考値も併記する。
5. **ハイパーパラメータ調整** — 学習率を下げて n_estimators を増やす、正則化を効かせる。

変更したら必ず `python src/train.py` で CV スコアの変化を確認してから提出する。
CV が改善しないハイパラ変更は採用しない。

### CV が LB より良く出る点に注意(重要)

現行 `preprocess.py` は train/test を**結合してから** Box-Cox や近隣中央値を計算する
(上位ノート流の一般的なやり方)。このため CV に軽いリークが入り、**CV(≈0.108)は
LB(≈0.122)より楽観的に出る**。スコアの最終判断は LB を信じること。CV はあくまで
「改造で良くなったか/悪くなったか」の相対比較に使う。

### 試したが不採用(知見)

House Prices で定番とされる手法でも、このパイプラインでは効かないことが多い。
必ず CV で確認すること。

- **品質グレードの順序エンコード(単独適用)** — one-hot のままの方が lasso の CV が
  良く、全体でも悪化(過去版 stack 0.1083→0.1091)。
  ※現行版では本格前処理の一部としてラベルエンコードを採用済み。
- **多項式・交互作用特徴**(OverallQual² / TotalSF² / Qual×TotalSF など)。
  Box-Cox 済みの元特徴と冗長で CV 悪化(nnls 0.1079→0.1084、lasso 0.1099→0.1116)。
- **数値カテゴリの文字列化 / 近隣別 LotFrontage 補完(単独適用)** — 単独では lasso CV
  0.1098→0.1099 で効果ゼロだった。現行版では本格前処理の一部として採用。

CV ≈ 0.108 / LB ≈ 0.122 で頭打ち。純粋な機械学習での実用上の底に近い。
**LB 0.10 以下は多くが test 答えの「リーク」を使う領域**で、正攻法では到達困難。
さらに伸ばすなら別データ源や、CV からリークを排した厳密な検証基盤づくりが必要。

## 実績(参考)

| 手法 | CV RMSE(log) | Public LB |
| --- | --- | --- |
| XGBoost 単体ベースライン | 0.1275 | 0.12727 |
| ブレンド + 外れ値除去 + 歪度補正 | 0.1084 | 0.12331 |
| スタッキング(+ LightGBM) | 0.1083 | 0.12280 |
| 強い6モデルのブレンド(ENet/KernelRidge 追加) | 0.1080 | 0.12272 |
| 本格前処理 + 6モデル NNLS ブレンド(現行) | 0.1079* | **0.12246** |

\* 現行 CV は train/test 結合によりやや楽観的。LB が実力に近い。

## 関連ドキュメント / スクリプト

- `docs/note_kaggle_house_prices.md` — 初心者向けの解説記事(スコアの読み方など)。
- `docs/top_solutions_analysis.md` — 上位者(正攻法トップ／リーク勢)の手法分析。
- `src/leak_submission.py` — **【リーク実演・教育用】** 外部の Ames 元データで test の
  正解を引く裏ワザ。LB ≈ 0.00463 になるが**モデルの実力ではない**。データリークを
  学ぶ教材として用意したもので、正攻法(`train.py`)とは明確に分離している。
  本来のコンペでは規約違反に当たる点に注意。
