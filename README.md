# House Prices - Advanced Regression Techniques

Kaggle コンペ [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) に取り組むための Python プロジェクトです。

住宅のさまざまな情報(広さ・築年数・立地など)から販売価格(`SalePrice`)を予測します。
まずは **ベースラインのスコアを 1 本出せる状態** をゴールにしています。

## ディレクトリ構成

```
kaggle-house-prices/
├── data/                  # 取得した CSV を置く場所(.gitignore で除外)
├── src/
│   ├── download_data.py   # Kaggle API でデータをダウンロード
│   ├── preprocess.py      # 前処理(欠損値・エンコード・特徴量づくり)
│   └── train.py           # 学習 → 交差検証 → submission.csv 作成
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

## このベースラインの中身

- **前処理** (`src/preprocess.py`)
  - 欠損値: 数値列は中央値、カテゴリ列は `"None"` で補完
  - 特徴量づくり: 合計床面積・築年数・バスルーム合計数を追加
  - カテゴリ変数: ワンホットエンコーディング
- **モデル** (`src/train.py`)
  - XGBoost(勾配ブースティング)による回帰
  - 目的変数は `log1p` 変換してから学習(評価指標に合わせるため)
  - 5 分割の交差検証でスコアを確認

ここからさらに、特徴量の追加やハイパーパラメータ調整、複数モデルのアンサンブルなどでスコアの改善を目指せます。
