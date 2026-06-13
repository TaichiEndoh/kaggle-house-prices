# House Prices 上位者アプローチの徹底分析

このコンペの「上位」は、性質のまったく違う **2つの層** に分かれている。
まずこの構造を理解するのが重要。

| 層 | だいたいの LB | 正体 |
| --- | --- | --- |
| リーク勢 | 0.00〜0.05 | 外部データ(元の Ames データ)で test の正解を引いている。**機械学習ではない** |
| 正攻法トップ | 0.105〜0.120 | 丁寧な前処理＋強力なアンサンブル。**ここが純ML の実質的な頂点** |
| 一般的な提出 | 0.13〜0.15 | モデルを1〜数個動かしたレベル |

> 実際、トップ100〜200の多くはリークを使っていることが知られている。
> （我々もリーク実演で **LB 0.00463** を確認済み。詳細は本リポジトリの `src/leak_submission.py`）

したがって「正攻法でどこまで行けるか」を学ぶなら、見るべきは **LB 0.105〜0.120 の
正攻法トップ** であり、その代表が Serigne の "Stacked Regressions"。

---

## 1. 正攻法の金字塔：Serigne「Stacked Regressions: Top 4%」(LB ≈ 0.1177)

House Prices で最も参照される正攻法ノート。構成要素を分解する。

### (1) 外れ値除去
- `GrLivArea > 4000 かつ SalePrice < 300000` の物件（広いのに激安）を学習から除外。
  → 我々も採用済み。

### (2) 目的変数の変換
- `SalePrice` を `log1p` 変換。右に偏った価格分布を正規分布に近づけ、評価指標
  （価格 log の RMSE）とも整合させる。→ 採用済み。

### (3) 列ごとの意味を踏まえた欠損補完（ここが丁寧）
- 「設備が無い」を意味する欠損 → `"None"`：PoolQC, MiscFeature, Alley, Fence,
  FireplaceQu, Garage系, Bsmt系, MasVnrType など。
- 数量ゼロを意味する欠損 → `0`：Garage/Bsmt の面積・本数、MasVnrArea など。
- `LotFrontage` → **近隣(Neighborhood)ごとの中央値** で補完。
- 残りカテゴリ → 最頻値（MSZoning, Electrical, KitchenQual, Exterior, SaleType）。
- `Functional` → `"Typ"`、`Utilities` → 情報量が無いので列ごと削除。
  → 我々も現行版で全面採用。

### (4) 歪度補正：Box-Cox
- 数値列のうち歪度 |skew| > 0.75 のものを **Box-Cox 変換（λ=0.15）** で正規化。
  log1p より少しだけ強力。→ 採用済み。

### (5) エンコード
- 一部の順序カテゴリ（品質・状態・MSSubClass・YrSold・MoSold など26列）を
  **ラベルエンコード**（整数化）。
- 残りは `pd.get_dummies` でワンホット。→ 採用済み。

### (6) ベースモデル（具体ハイパラ）
| モデル | 主要パラメータ |
| --- | --- |
| Lasso | alpha=0.0005（RobustScaler を前段に） |
| ElasticNet | alpha=0.0005, l1_ratio=0.9 |
| KernelRidge | kernel=polynomial, degree=2, coef0=2.5, alpha=0.6 |
| GradientBoosting | n=3000, lr=0.05, max_depth=4, max_features='sqrt', loss='huber' |
| XGBoost | n=2200, lr=0.05, max_depth=3, subsample=0.52, colsample=0.46, gamma=0.047 |
| LightGBM | n=720, lr=0.05, num_leaves=5, feature_fraction=0.23, bagging=0.8 |

→ 我々の `train.py` はこの値をベースに採用。

### (7) アンサンブル構造（Serigne の肝）
2段構え：
1. **StackingAveragedModels**：ベース = [ElasticNet, GradientBoosting, KernelRidge]、
   メタ学習器 = Lasso。ベースの out-of-fold 予測をメタが学習。→ CV ≈ 0.1081。
2. **最終ブレンド**：`0.70 × スタッキング + 0.15 × XGBoost + 0.15 × LightGBM`。
   → **LB ≈ 0.1177**。

### 我々との違い
我々は (7) を「6モデルの OOF 予測を **非負最小二乗(NNLS)** で重み最適化する線形
スタッキング」に置き換えた。CV 0.1079 と Serigne の CV 0.1081 はほぼ同等。
**つまり前処理・モデルの“中身”はトップ正攻法とほぼ同等まで到達している。**

---

## 2. 他の正攻法トップに共通するテクニック

複数の上位ノート／ブログを横断すると、共通項がはっきりしている。

- **目的変数の log 変換**（ほぼ全員）。
- **歪んだ特徴量の log/Box-Cox 変換**。
- **外れ値の除去**（GrLivArea の2件は定番）。
- **ドメイン知識ベースの欠損補完**（"None"/0/最頻値の使い分け）。
- **順序カテゴリのラベルエンコード**＋残りワンホット。
- **複合特徴量**：TotalSF（合計床面積）はほぼ全員が作る最重要特徴。
- **正則化線形モデル中心のアンサンブル**：このデータは Lasso/ENet/KernelRidge が
  木モデルより強い。木（XGB/LGB/GBoost）は多様性要員として少量ブレンド。
- **スタッキング or 重み付きブレンド**で最後にまとめる。
- **残差分析**で外し方のクセを見て特徴量を追加（上級者の差がつく所）。

実例：トップ100入り（**LB 0.11229, 84位/4742**）のブログでは、
sklearn パイプライン整備、`BsmtQual` 等の順序エンコード、残差分析による特徴量追加、
StackingRegressor、目的変数 log1p を使用。リークには言及していない＝正攻法。

---

## 3. リーク勢（LB ≈ 0）の実態

- test は学術データ **Ames Housing (Dean De Cock, 2011)** の一部。
- 物件の特徴量で元データに突き合わせると、伏せられた SalePrice をそのまま引ける。
- これで LB はほぼ 0 になるが、**モデルの実力とは無関係**。常設の学習用コンペ
  （賞金なし）なので実害は小さいが、本来のコンペでは規約違反。
- 「LB 上位＝強いモデル」ではないことの典型例であり、**データリークという概念を
  学ぶ格好の教材**。我々の実演で LB 0.00463 を確認。

---

## 4. 我々の現在地と、0.115 台を詰めようとした結果

現在：**正攻法 Public LB ≈ 0.1223 / honest CV ≈ 0.108**（リーク実演は 0.00463）。
前処理・モデルはトップ正攻法とほぼ同等。

「0.115 台を本気で狙う」として、分析で挙げた2つのレバーを実際に実行した：

### レバー1：CV からリークを排した厳密な検証基盤(実行済み)
前処理を `HousePreprocessor`(sklearn 変換器)にまとめ、**fold 内でのみ fit** する形に
作り替えた。結果、CV が信頼できる値になった(leaky lasso 0.1098 → honest 0.1111)。
ブレンドの honest CV は 0.1076〜0.1086。**改善判断を LB 提出に頼らず CV でできる**ように
なったのが最大の収穫。

### レバー2：残差分析 → 特徴量追加(実行済み・効果なしと判明)
honest な OOF 残差を全特徴量と相関させたところ、**最大でも相関 0.05**。
つまり残差はどの特徴量とも結びついておらず、**モデルは既にノイズ下限に到達**している。
大きく外す物件は Abnorml/Family/Partial といった非通常売買の個別ケースで、本質的に
予測不能。試した追加特徴量(築年数系・多項式)はいずれも honest CV を動かさなかった。

### ブレンド重みの調整(実行済み・LB 不動)
NNLS / 等重み / 収縮ブレンドを比較。CV は多少動くが **LB はいずれも ≈0.1223** で不動。
CV(0.108)と LB(0.122)の 0.014 差は **この train/test 分割固有**で、重み付けでは消えない。

### 結論
**純ML・正攻法での実用上の底は LB ≈ 0.1223** と分かった。
ここから 0.115 以下へ行くには、(a) public LB を総当たりで過適合させる(private には
効かず、実質ギャンブル＝gaming)、(b) test 答えの「リーク」を使う、のどちらかしかない。
どちらも「強いモデル」とは別物。

趣味としては **0.12 前後 = トップ正攻法の入り口**に、特別な裏技なしで到達できている。
「基本を丁寧にやり切る」ことの価値と、「これ以上は意味のある改善ではない」見極めの
両方を、実際に手を動かして確認できたのが本当の成果。

---

## 出典

- [Serigne — Stacked Regressions: Top 4% on Leaderboard (Kaggle)](https://www.kaggle.com/code/serigne/stacked-regressions-top-4-on-leaderboard)
- [#1 House Prices Solution [top 1%] (Kaggle, jesucristo)](https://www.kaggle.com/code/jesucristo/1-house-prices-solution-top-1)
- [How I Cracked the Top 100 in the Kaggle House Prices Competition (Finxter, LB 0.11229 / 84位)](https://blog.finxter.com/how-i-cracked-the-top-100-in-the-kaggle-house-prices-competition/)
- [Top 3% rank: Kaggle House Prices using Bagging Ensemble (Medium)](https://parisrohan.medium.com/top-3-rank-kaggle-house-prices-advanced-regression-techniques-using-bagging-ensemble-ff2a3f9b70cb)
- [House Prices: Advanced Regression 'solution' file (Kaggle dataset, リーク議論)](https://www.kaggle.com/datasets/carlmcbrideellis/house-prices-advanced-regression-solution-file)
- 元データ：Dean De Cock, "Ames, Iowa: Alternative to the Boston Housing Data" (JSE, 2011) — http://jse.amstat.org/v19n3/decock.pdf
