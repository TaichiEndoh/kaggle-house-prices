# ROGII Wellbore Geology Prediction — 下調べ & 叩き台

House Prices で作ったテンプレを、賞金付きコンペ
[ROGII - Wellbore Geology Prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction)
($50,000 / 締切 2026-08-05) に転用するための調査メモと足場(スキャフォールド)。

> ⚠️ この時点ではデータ本体を取得できていない(下記「最初の一歩」参照)。
> 列名・評価指標の細部は **TODO** で、データ取得後に確定させる。

## このコンペの中身(調査で判明した範囲)

- **目的**：横坑井(horizontal well)に沿った地層を予測し、地質ステアリングを自動化する。
  具体的には各坑井の **評価ゾーンの TVT(True Vertical Thickness=真垂直層厚, ft)** を予測。
- **データ単位は「坑井(well)」**。各坑井につき2つの CSV:
  - `{id}__horizontal_well.csv` … 軌道(trajectory)、地層面、ログデータ
  - `{id}__typewell.csv` … 地質対比に使う垂直参照ログ
  - train にはさらに `{id}.png`(可視化)もある。
- **主な列(判明分)**：
  | 列 | 意味 |
  | --- | --- |
  | `MD` | Measured Depth 測定深度 (ft) |
  | `X` | Easting 東距 (ft) |
  | `GR` | Gamma Ray ガンマ線 (API) |
  | `Geology` | 地層ラベル(カテゴリ) |
  | `TVT_input` | TVT のコピー。**評価ゾーンだけ NaN** ＝ここを予測する |
- **予測対象**：`TVT_input` が NaN の行(評価ゾーン)の TVT。
- **評価指標**：未確定。公開ノートのスコアが ~9.25 / ~9.54 など ft オーダーなので
  **MAE か RMSE(小さいほど良い)** と推定。→ **規約/Evaluation で要確定(TODO)**。
- **提出形式**：`sample_submission.csv` の列に合わせる(取得後に確認)。

## House Prices との決定的な違い(設計に効く)

1. **1行 = 1物件ではなく「坑井内の深度点の系列」**。同じ坑井の行は強く相関する。
   → **CV は必ず坑井単位の GroupKFold**。ランダム KFold だと同一坑井が train/val に
   跨って **情報漏れ(リーク)** し、CV が当てにならなくなる。**ここが最重要の適応点。**
2. **複数ファイルを1表に束ねる前処理が要る**(horizontal + typewell のペアを読み込み、
   坑井IDを付けて縦結合 → 1行=1深度点の表に平坦化)。`data_loader.py` が担当。
3. **typewell(垂直参照)をどう特徴量に落とすか**が腕の見せ所。
   GR の対比、深度方向の移動平均・勾配、評価ゾーンまでの距離など。
4. 評価指標が RMSE(log) ではない可能性大 → 目的変数の log 変換はしない想定で開始。

## 再利用できるもの / 作り替えるもの

| 部品 | 流用 | メモ |
| --- | --- | --- |
| `src/models.py`(6モデル動物園) | ◎ ほぼ流用 | 回帰なのでそのまま使える(lgb は n_jobs=1) |
| OOF + NNLS 収縮ブレンド(`blend.py`の発想) | ◎ | 評価指標に合わせて誤差関数だけ差し替え |
| fold 内 fit の前処理思想 | ◎ | ただし CV は GroupKFold に変更 |
| `src/preprocess.py` の TabularPreprocessor | △ | 列ロールが House Prices 専用。ROGII 用に簡易版を用意 |
| `data_loader.py`(ペアCSVの平坦化) | ★新規 | このコンペ特有。最初の山場 |
| `features.py`(坑井ログの特徴量) | ★新規 | 移動平均/勾配/距離など |

## 最初の一歩(データ取得には同意が必要)

データDLは現在 **403(ルール未同意)**。次の手順で解禁してから本格化する:

1. [コンペページ](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction)で
   **「Join Competition」→ ルール同意**(賞金条件・**ブログ報告可否/NDA有無**もここで確認)。
2. データ取得:
   ```bash
   kaggle competitions download -c rogii-wellbore-geology-prediction -p data_rogii
   unzip -o data_rogii/*.zip -d data_rogii
   ```
3. `python rogii/inspect_data.py` で実際の列名・行数・評価ゾーンの形を確認。
4. 確認結果をもとに `config.py` / `data_loader.py` / `features.py` の **TODO を埋める**。

## TODO チェックリスト

- [ ] ルール同意してデータ取得、`inspect_data.py` で実構造を確認
- [ ] **評価指標を確定**(MAE? RMSE? 重み付き? lower better?)→ `config.METRIC`
- [ ] horizontal/typewell の **正確な列名** を `config.py` に反映
- [ ] train の TVT 正解の所在を確認(TVT_input が full か、別列か)
- [ ] `data_loader.py` の平坦化を実データで検証(行数・NaN ゾーン)
- [ ] **GroupKFold(groups=well_id)** で 1本ベースライン提出 → LB と CV の整合確認
- [ ] typewell 由来の特徴量を `features.py` に追加(効いたものだけ採用)

## 進め方の原則(House Prices で確立した型)

1. まず **正しい CV(GroupKFold)** で素朴な1本を出す。
2. 残差を見て、効く特徴だけを CV で確認して採用。効かない改造はやらない。
3. 速度はボトルネックを計測してから対処(小データの lgb は n_jobs=1)。
4. LB は確認用。CV を信じて回す。引き際も意識する。
