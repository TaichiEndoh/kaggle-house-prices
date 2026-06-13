# House Prices - Advanced Regression Techniques

Kaggle コンペ [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) に取り組むための Python プロジェクトです。

住宅のさまざまな情報(広さ・築年数・立地など)から販売価格(`SalePrice`)を予測します。
まずは **ベースラインのスコアを 1 本出せる状態** をゴールにしています。

## ディレクトリ構成

設定駆動の再利用テンプレートになっています(他の表形式回帰コンペにも転用可)。
コンペ固有なのは `config.py` と `features.py` の2ファイルだけ。詳細は
[`docs/TEMPLATE.md`](docs/TEMPLATE.md)。

```
kaggle-house-prices/
├── data/                  # 取得した CSV を置く場所(.gitignore で除外)
├── src/
│   ├── config.py          # ★コンペ固有の設定(スラッグ/目的変数/欠損ルール 等)
│   ├── features.py        # ★コンペ固有の特徴量づくり
│   ├── preprocess.py      # 汎用前処理エンジン TabularPreprocessor(fold内fit)
│   ├── models.py          # モデル動物園(6モデルの Pipeline)
│   ├── train.py           # 学習 → OOF保存 → ブレンド → submission.csv
│   ├── blend.py           # 保存済みOOFから重みだけ再調整(再学習なし・一瞬)
│   ├── download_data.py   # Kaggle API でデータをダウンロード
│   └── leak_submission.py # 【教育用】データリーク実演(後述)
├── docs/                  # 解説記事・上位手法分析・テンプレート使い方
├── requirements.txt       # 必要なライブラリ
├── .gitignore
└── README.md
```

> **注意**: コンペデータ(`*.csv`)や認証情報(`kaggle.json`)はリポジトリにはコミットされません(`.gitignore` で除外しています)。

## セットアップ

### 1. ライブラリのインストール

```bash
pip install -r requirements.txt
```

### 2. Kaggle API トークンの設定

データの取得には Kaggle API の認証情報が必要です。以下の **どちらか一方** を設定してください。

#### 方法A: 環境変数を使う(クラウド環境におすすめ)

Kaggle の [アカウント設定ページ](https://www.kaggle.com/settings) で「Create New API Token」を押すと `kaggle.json` がダウンロードされます。その中身を環境変数に設定します。

```bash
export KAGGLE_USERNAME="あなたのユーザー名"
export KAGGLE_KEY="あなたのAPIキー"
```

#### 方法B: kaggle.json を置く

ダウンロードした `kaggle.json` を `~/.kaggle/kaggle.json` に配置します。

```bash
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json   # 自分だけが読めるようにする
```

> このコンペに初めて参加する場合は、Kaggle のコンペページで一度 **「Join Competition」(ルールに同意)** しておく必要があります。

## 実行手順

### 1. データのダウンロード

クラウド環境内で Kaggle API を使ってデータを取得します。

```bash
python src/download_data.py
```

成功すると `data/` に `train.csv` / `test.csv` / `sample_submission.csv` が保存されます。

### 2. 学習とベースライン提出ファイルの作成

```bash
python src/train.py
```

実行すると、

- 交差検証による RMSE スコア(log スケール)がコンソールに表示され、
- 予測結果が `submission.csv` として書き出されます。

## Kaggle への提出方法

### 方法A: Web から提出

Kaggle のコンペページの「Submit Predictions」から `submission.csv` をアップロードします。

### 方法B: コマンドラインから提出

```bash
kaggle competitions submit \
  -c house-prices-advanced-regression-techniques \
  -f submission.csv \
  -m "baseline (XGBoost)"
```

提出後、コンペページの「My Submissions」でスコア(Public Leaderboard)を確認できます。

### ブレンドだけ調整する(再学習なし・一瞬)

`train.py` は各モデルの予測を `data/oof_cache.npz` に保存します。ブレンドの重みや
収縮率(`config.BLEND_SHRINK`)を変えて試すだけなら、再学習せず一瞬で回せます。

```bash
python src/blend.py
```

## 中身(現行パイプライン)

- **前処理** (`src/preprocess.py` の `TabularPreprocessor`)
  - 列の意味に応じた欠損補完(設備なし=None / 数量=0 / 最頻値 / グループ別中央値)
  - 数値カテゴリの文字列化、順序カテゴリのラベルエンコード、ワンホット
  - 歪んだ数値列の Box-Cox 変換、外れ値除去
  - **fold 内でのみ fit** するため交差検証にリークが入らない(CV が信頼できる)
- **モデル** (`src/models.py`) — Lasso / ElasticNet / KernelRidge / GradientBoosting /
  XGBoost / LightGBM の6モデルを OOF 予測し、NNLS→等重み収縮でブレンド
- 目的変数は `log1p` 変換してから学習(評価指標に合わせるため)

到達点：honest CV ≈ 0.108 / Public LB ≈ 0.1223。
スコアの読み方や上位手法の分析は [`docs/`](docs/) を参照。
