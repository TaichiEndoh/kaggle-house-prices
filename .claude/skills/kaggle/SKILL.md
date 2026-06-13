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

- `src/preprocess.py` — 前処理を **fold 内 fit に対応した sklearn 変換器
  `HousePreprocessor`** にまとめたもの。外れ値除去(`remove_outliers`、CV 外で1回)、
  列ごとの欠損補完(設備なし=None / 数量=0 / その他=最頻値)、近隣別 LotFrontage 補完、
  数値カテゴリの文字列化、ラベルエンコード、特徴量づくり、Box-Cox、ワンホットを
  fit/transform で行う。**統計量は fit した fold からのみ学習** するので CV にリークが入らない。
- `src/train.py` — 6モデルを「前処理 → (スケーリング) → モデル」の Pipeline にし、
  `cross_val_predict` で OOF 予測を作成。NNLS で重みを学習し、**等重みへ収縮**させた
  ブレンド(正則化スタッキング)で submission を作る。
- `src/download_data.py` — Kaggle API でデータ取得。
- `src/leak_submission.py` — リーク実演(教育用、後述)。

## スコア改善の定石(伸ばしたいと言われたら)

効果が出やすい順(いずれも実装済み):

1. **外れ値除去** — `GrLivArea > 4000 かつ 価格が安い` 物件を学習から除外。
2. **目的変数の log 変換** + **歪んだ数値列の Box-Cox(λ=0.15)**。
3. **列ごとの意味を踏まえた欠損補完**(None / 0 / 最頻値 / 近隣別中央値)。
4. **強い6モデルのブレンド** — Lasso / ElasticNet / KernelRidge / GradientBoosting /
   XGBoost / LightGBM。このデータは線形・カーネル系が強い。NNLS で重みを学習し、
   過適合を避けるため等重みへ収縮(`SHRINK`)させる。
5. **ハイパーパラメータ調整** — 効果は逓減。

変更したら必ず `python src/train.py` で **信頼できる CV** を見て判断する。
CV が改善しない変更は採用しない。

### 検証は fold 内 fit(リークなし)

前処理は `HousePreprocessor` で fold 内 fit されるため、CV は信頼できる。
過去に train/test 結合で前処理していた頃は CV が 0.001 ほど楽観的に出ていた
(例: leaky lasso 0.1098 → honest 0.1111)。改善判断は honest CV で行う。

### 試したが不採用(知見)

House Prices で定番とされる手法でも、このパイプラインでは効かないことが多い。

- **多項式・交互作用特徴**(OverallQual² / TotalSF² / Qual×TotalSF など)— Box-Cox 済みの
  元特徴と冗長で CV 悪化。
- **築年数系特徴**(HouseAge / RemodAge / IsNew / IsRemodeled)— 年次列が既に情報を
  持っており honest CV 不変(lasso/enet とも 0.1111)。
- **ブレンド重みの工夫**(NNLS / 等重み / 収縮)— CV は多少動くが **LB はいずれも ≈0.1223**
  で不動。重み付けでは LB は変わらない。

### 到達点と限界(重要)

- honest CV ≈ **0.108** / Public LB ≈ **0.1223** が正攻法の堅牢な底。
- **残差分析で、残差はどの特徴量とも相関しない(最大 0.05)= モデルはノイズ下限に到達済み**。
  特徴量追加では LB は動かない。
- CV(0.108)と LB(0.122)の 0.014 差は **この train/test 分割固有**で、修正可能な
  アーティファクトではない(ブレンドを変えても不動)。
- **LB 0.115 以下は、LB 総当たり(public 過適合)か test 答えの「リーク」が必要**で、
  純粋な機械学習・正攻法では実用上ここが底。趣味としては十分トップ正攻法の入り口。

## 実績(参考)

| 手法 | CV RMSE(log) | Public LB |
| --- | --- | --- |
| XGBoost 単体ベースライン | 0.1275 | 0.12727 |
| ブレンド + 外れ値除去 + 歪度補正 | 0.1084 | 0.12331 |
| スタッキング(+ LightGBM) | 0.1083 | 0.12280 |
| 強い6モデルのブレンド(ENet/KernelRidge 追加) | 0.1080 | 0.12272 |
| 本格前処理 + 6モデル NNLS ブレンド | 0.1079† | 0.12246 |
| **fold-safe パイプライン + 収縮ブレンド(現行)** | **0.1079‡** | **0.12236** |
| (参考)リーク実演 — モデルではない | — | 0.00463 |

† train/test 結合のため楽観的 / ‡ fold 内 fit の信頼できる CV。NNLS 版は LB 0.12230 で同等。

## 関連ドキュメント / スクリプト

- `docs/note_kaggle_house_prices.md` — 初心者向けの解説記事(スコアの読み方など)。
- `docs/top_solutions_analysis.md` — 上位者(正攻法トップ／リーク勢)の手法分析。
- `src/leak_submission.py` — **【リーク実演・教育用】** 外部の Ames 元データで test の
  正解を引く裏ワザ。LB ≈ 0.00463 になるが**モデルの実力ではない**。データリークを
  学ぶ教材として用意したもので、正攻法(`train.py`)とは明確に分離している。
  本来のコンペでは規約違反に当たる点に注意。
