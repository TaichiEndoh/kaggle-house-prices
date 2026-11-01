# House Prices(house-prices-advanced-regression-techniques)の記録

入門コンペ(賞金なし)。住宅の約80項目から SalePrice を予測。CSV 提出。
評価指標は **log(価格) の RMSE**(小さいほど良い)。目的変数は `log1p` で学習し `expm1` で戻す。

## 結果

| 手法 | CV RMSE(log) | Public LB |
| --- | --- | --- |
| XGBoost 単体ベースライン | 0.1275 | 0.12727 |
| 外れ値除去 + 歪度補正 + 4モデルブレンド | 0.1084 | 0.12331 |
| スタッキング(+ LightGBM) | 0.1083 | 0.12280 |
| 強い6モデルのブレンド(ENet/KernelRidge 追加) | 0.1080 | 0.12272 |
| 本格前処理 + 6モデル NNLS ブレンド | 0.1079† | 0.12246 |
| **fold-safe パイプライン + 収縮ブレンド(最終)** | **0.1079‡** | **0.12236** |
| (参考)リーク実演 — 実力ではない | — | 0.00463 |

† train/test 結合で前処理していたため楽観的 / ‡ fold 内 fit の信頼できる CV。
最終順位: 約852位 / 5209(上位16%)。正攻法トップの相場は 0.115〜0.118。

## 効いたこと

1. 外れ値除去(`GrLivArea > 4000` かつ安い物件)。
2. 目的変数 log + 歪んだ数値列の Box-Cox(λ=0.15)。
3. 列の意味に応じた欠損補完(設備なし=None / 数量=0 / 最頻値 / 近隣別中央値)。
4. Lasso / ElasticNet / KernelRidge / GradientBoosting / XGBoost / LightGBM の6モデルを
   OOF で NNLS 重み付けし、過適合を避けるため等重みへ収縮(このデータは線形系が強い)。
5. 前処理を `TabularPreprocessor`(fold 内 fit)にして CV を信頼できる値にした
   (leaky lasso 0.1098 → honest 0.1111)。

## 効かなかったこと(不採用)

- 多項式・交互作用特徴(Box-Cox 済みの元特徴と冗長で CV 悪化)。
- 築年数系特徴(年次列が既に情報を持っていて CV 不変)。
- ブレンド重みの工夫(NNLS / 等重み / 収縮で CV は動くが LB はどれも ≈0.1223)。
- HistGradientBoosting(速いが CV 0.1257 で悪化)。

## 到達点

残差がどの特徴量とも相関しない(最大 0.05)= ノイズ下限。正攻法ではここが実用上の底。
CV と LB の差(0.108 vs 0.122)はこの分割に固有で、重み付けでは消えなかった。

## コード

- テンプレート: `src/`(コンペ固有は `config.py` と `features.py` の2ファイルだけ)。
  使い方は `docs/TEMPLATE.md`。`python src/train.py`(OOF 保存)→ `python src/blend.py`(再学習なしで重み調整)。
- リーク実演(教育用・正攻法と分離): `src/leak_submission.py`。
- 記事: `docs/note_kaggle_house_prices.md`, `docs/note_full_journey.md`,
  上位手法の分析: `docs/top_solutions_analysis.md`。
