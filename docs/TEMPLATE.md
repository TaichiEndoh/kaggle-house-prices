# 表形式回帰コンペ用テンプレートの使い方

このリポジトリは House Prices で作ったパイプラインを、**他の表形式回帰コンペにも
転用できるテンプレート**として整理したもの。コンペ固有なのは2ファイルだけで、
あとは汎用エンジンが設定を読んで動く。

## 全体像（どこを触るか）

```
src/
  config.py          ← ★コンペ固有の設定(ここを書き換える)
  features.py        ← ★コンペ固有の特徴量づくり(ここを書き換える)
  preprocess.py      汎用前処理エンジン TabularPreprocessor(fold内fit・触らない)
  models.py          モデル動物園 build_models()(必要なら調整)
  train.py           学習ドライバ(触らない)
  blend.py           OOFキャッシュから重み再調整(触らない)
  download_data.py   Kaggle API でデータ取得(触らない)
  leak_submission.py House Prices 専用のリーク実演(教育用・転用時は無視)
```

設計のキモは **前処理を fold 内でだけ fit する `TabularPreprocessor`**。
これにより交差検証(CV)にリークが入らず、CV が信頼できる＝改善判断を提出に頼らず
手元でできる。

## 新しいコンペに使う手順

1. **`config.py` を書き換える**
   - `COMPETITION` … Kaggle の URL 末尾のスラッグ
   - `TARGET` / `ID` … 目的変数列と ID 列
   - `LOG_TARGET` … 目的変数が右に歪んでいれば `True`(log1p 学習)
   - `outlier_mask(df)` … 外れ値の除外ルール(不要なら全 True を返す)
   - 欠損補完の列ロール：`NONE_COLS`(=設備なし) / `ZERO_COLS`(=数量0) /
     `MODE_COLS`(最頻値) / `CONST_FILL`(固定値) / `GROUP_MEDIAN`(グループ別中央値) /
     `DROP_COLS`(削除)
   - エンコード：`TO_STRING_COLS`(数値だが実体カテゴリ) / `LABEL_COLS`(順序カテゴリ)
   - `SKEW_THRESHOLD` / `BOXCOX_LAMBDA` / `N_SPLITS` / `BLEND_SHRINK`

2. **`features.py` の `add_features` を書き換える**
   - そのコンペ特有の合成特徴量を作る。思いつかなければ中身を空にして `return df`。

3. **回す**
   ```bash
   python src/download_data.py   # データ取得
   python src/train.py           # CV 表示 + OOF 保存 + submission 作成
   ```

4. **ブレンドだけ調整したいとき(再学習なしで一瞬)**
   ```bash
   python src/blend.py           # data/oof_cache.npz から重みを再計算
   ```
   `config.BLEND_SHRINK` を変えて `blend.py` を回せば、重みの効き方を即確認できる。

## 効率化メモ（実測ベース・徹底検証の結論）

このテンプレートは「正しく」かつ「速い」ことを実測で確認している。

- **LightGBM は小データで `n_jobs=1`**(最重要)
  数千行の小さなデータに対し `n_jobs=-1` だとスレッド競合で**激遅**になる。
  実測：CV1回が **392秒 → n_jobs=1 で 2.3秒**(スコアは 0.1180 で不変)。
  小データの LightGBM/XGBoost は素直に `n_jobs=1` が速い。これだけで
  全体が **約7分 → 47秒(9倍速)** になった。

- **OOF をキャッシュ → ブレンド調整を分離**
  `train.py` が各モデルの OOF 予測と test 予測を `data/oof_cache.npz` に保存。
  重み(ブレンド方式・収縮率)の試行錯誤は `blend.py` が**再学習なし・0.8秒**で回せる。
  「重みをいじるたびに全モデル再学習(7分)」という無駄を消せる。

- **HistGradientBoosting は速いが不採用**
  `HistGradientBoostingRegressor` は多スレッドで速い(9.5秒)が、このデータでは
  CV が 0.1257 と悪かったため不採用。速さだけで選ばない(必ず CV で確認)。

- **さらに速くしたい/重くなったら**
  - 開発中は `config.N_SPLITS` を 3 にして素早く回す(本番前に 5 へ戻す)。
  - ブースティングは early stopping を使えば固定 `n_estimators` の無駄を省ける。
  - データが大きいコンペでは逆に木モデルの `n_jobs=-1` が効く(小データの逆)。

## スコアの読み方・到達点

- 評価は RMSE(価格 log)。**小さいほど良い**(正解率ではない)。
- 初心者向け解説：`docs/note_kaggle_house_prices.md`
- 上位手法の分析と限界：`docs/top_solutions_analysis.md`
- House Prices での到達点：honest CV ≈ 0.108 / Public LB ≈ 0.1223(正攻法の実用上の底)。
