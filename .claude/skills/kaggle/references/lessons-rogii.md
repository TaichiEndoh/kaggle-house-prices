# ROGII Wellbore Geology Prediction の記録

賞金 $50,000 の Featured・**コードコンペ**(非公開 test でノートブックを再実行して採点)。
約 6,125 チーム。エントリー締切 2026-07-29。

## 経緯と結果

| 提出 | Public | Private |
| --- | --- | --- |
| 2026-06-20(本番中のベスト、別セッション) | 12.507 | 12.762 |
| **2026-11-01 Late: 物理特徴 + LightGBM** | **11.746** | **10.470** |

- 6月に3回提出したあと改善が止まり、そのまま締切を迎えた(反省点: 間が空くと締切を逃す)。
- 公開順位表で 12.507 は 4,318位、11.746 は約 4,171位相当(この帯はスコアが密集)。
- メダル目安(Public): 金 ≈ 5.5、銀 ≈ 6.3、銅 ≈ 6.4、中央値 ≈ 8.0。1位 4.6。

## データと評価

- 坑井ごとに `{id}__horizontal_well.csv` と `{id}__typewell.csv`。train は 773 坑井。
- horizontal(train): `MD, X, Y, Z, ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA(地層面の深さ), TVT, GR, TVT_input`
- horizontal(test): `MD, X, Y, Z, GR, TVT_input` だけ。**地層面と TVT が無い。**
- typewell: `TVT, GR, Geology`。
- 予測対象: `TVT_input` が NaN の行(PS=予測開始点より後)。坑井の約74%が予測区間。
- 提出: `id = {坑井ID}_{行番号(0始まり)}`, `tvt`。
- 評価: 予測 TVT と正解のずれの **RMSE(ft)**(配布 pptx の最終スライドに記載)。
- GR は予測区間の約32%が欠損。

## 分かったこと(核心)

- **オラクル**: 自坑井の地層面を使うと `TVT = K − (Z − 地層面)` で RMSE 0.006。
  → TVT は地層面からの高さで決まる。問題は「test で地層面(傾き)をどう推定するか」。
- 坑井は地層に沿って掘られるので「PS の TVT を保つ(const)」が強いベースライン。

## 検証(坑井単位 5 分割)の結果

| 手法 | CV RMSE |
| --- | --- |
| const(PS の TVT を維持) | 15.91 |
| Z を使った水平地層の仮定 / 直線外挿 | 107〜124 |
| 近傍坑井の地層面を IDW 補間 | 約26(不採用) |
| Viterbi で GR を typewell と照合 | const とほぼ同じ(不採用) |
| 近傍の平面で傾き推定 × 0.5 | 13.93 |
| **物理特徴(IDW・平面・自坑井トレンド) + LightGBM** | **11.53**(Public 11.75 / Private 10.47 と整合) |

LightGBM の重要度: IDW の地層面変化 0.32、平面(半径3000ft)0.16。
学習用の近傍特徴は「自分の分割を除いた坑井」だけで作り、test 側も同じ5本の木で作って予測を平均した。

## 次の課題

- 上位解法が使っていた **typewell との GR 正規化相互相関**(局所的な TVT の上下を GR 波形で補正)。
- 層別のグループ CV(stratified group CV)。
- 参考: Zenn「Kaggle ROGIIコンペ振り返り―192nd Place Solution」、GitHub `mycarta/rogii-geosteering-toolkit`。

## コード

- 提出用カーネル: `rogii/kernel/rogii_geosteer.py` + `kernel-metadata.json`
  (ローカルでは `python3 -I rogii/kernel/rogii_geosteer.py data_rogii` で約1分)。
- 検証スクリプト: `rogii/experiments/`(`build_cache.py` → `build_oof_feats.py` → `train_gbm.py` など)。
- データは `data_rogii/`(.gitignore 済み、約1.3GB)。
