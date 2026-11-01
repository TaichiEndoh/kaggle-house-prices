---
name: kaggle
description: >-
  Kaggle コンペを進めるための実戦プレイブック(表形式・コードコンペ・GPU 学習を含む)。
  安全確認、コンペ選定、データ点検、検証設計、ベースライン、改善、提出(通常/コードコンペ/
  Late Submission)、スコア確認、振り返りの記録までを扱う。House Prices と ROGII で得た
  教訓と数値を参照ファイルに持つ。ユーザーが「Kaggle」「コンペ」「ベースライン出して」
  「提出して」「スコア/順位確認」「CV」「LB」「Late Submission」「次のコンペ」「GPU で学習」
  「Windows の GPU」「ROGII」「House Prices」などと言ったら使う。
---

# Kaggle 実戦プレイブック

House Prices(入門・表形式回帰)と ROGII(賞金付き・コードコンペ)を通して確立した型。
コンペ固有の詳細は `references/` を読む。

## 0. 安全(最優先・毎回)

- **鍵の値を絶対に表示しない。** `env | grep kaggle` や `cat kaggle.json` は禁止
  (2026-06 にこれで KAGGLE_KEY が会話ログに漏れた)。有無だけ確認する:
  ```bash
  [ -n "$KAGGLE_USERNAME" ] && echo "USERNAME: set"; [ -n "$KAGGLE_KEY" ] && echo "KEY: set"
  [ -f ~/.kaggle/kaggle.json ] && echo "kaggle.json: あり"
  kaggle competitions list -p 1 | head -2   # 認証が通るかだけ見る
  ```
- 鍵が漏れたら: kaggle.com/settings で **Expire** → 新しく作成 → 環境変数/設定欄に直接入れる
  (チャットに貼らせない)。新しいセッションで反映を確認する。
- ダウンロードしたデータは専用ディレクトリに置き、`.gitignore` に入れる。データ・鍵・
  重いモデルはコミットしない。

## 1. コンペ選定と着手前チェック

| 確認項目 | どこで | メモ |
| --- | --- | --- |
| 締切・残り日数 | `kaggle competitions list` | **間が空くと締切を過ぎる**(ROGII は6月に3回提出後、止まったまま締切) |
| 賞金・カテゴリ | 同上 | 現金は Featured/Research。Getting Started は賞金なし、Playground は Swag 程度 |
| 提出方式 | Overview / Code タブ | **コードコンペ**(ノートブック再実行・インターネット不可)か CSV 提出か |
| 評価指標 | Evaluation タブ、配布資料(pptx 等) | 小さい方が良いのか大きい方か。単位も |
| ルール | Rules タブ | 外部データ、ブログ公開可否/NDA、チーム、提出回数 |
| 参加 | ページで Join → 同意 | 未同意だとデータ DL が **403** |

候補を出すときは「表形式で手元の型が効くか」「GPU が要るか」「締切まで何週あるか」を並べる。

## 2. 標準ワークフロー

1. **データ点検**: 列、行数、欠損、ID と提出形式。**train と test の列の差**を必ず見る
   (ROGII では test に地層面の列が無かった = それが本当の難所)。
2. **評価指標を確定**し、手元で同じ式を実装する。
3. **test と同じ状況を再現する検証を作る**。行が独立でないなら、まとまり単位で分割する
   (物件→KFold、坑井・患者・ユーザー→GroupKFold)。前処理の統計量は fold 内で fit する。
4. **自明なベースラインを測る**(定数、直前値の延長など)。これを超えない改造は無意味。
5. **上限値(オラクル)を測る**: train にしかない情報を使ったら何点出るか。問題の本質が分かる
   (ROGII: 地層面を使うと RMSE 0.006 → 地層面を推定する問題だと判明)。
6. **ドメイン知識で特徴量を作り、GBM に「どれだけ信じるか」を学習させる**
   (物理的な推定値をそのまま使うと大げさになりがち)。特徴量も OOF(自分を含まない)で作る。
7. **OOF をキャッシュ**してブレンドや後処理を再学習なしで試す。
8. 提出 → **CV と LB を比べる**。近ければ検証は健全。大きく離れたらリークか分割の誤りを疑う。

## 3. 原則(実測で確かめた教訓)

- **CV で良くならない変更は採用しない。** 定番テクでも効かないことが多い(HP で多数不採用)。
- **リークに注意**: train/test を結合した前処理は CV を楽観的にする。外部データで答えを引く
  「リーク」は実力ではない(HP で実演 0.00463。学習目的の非賞金コンペ以外ではやらない)。
- **速さは計測してから判断**: 小データの LightGBM は `n_jobs=-1` で激遅(392s→`n_jobs=1` で
  2.3s)。GPU も同じで、小データなら CPU の方が速いことがある。
- **引き際**: 残差がどの特徴量とも相関しなければノイズ下限。労力に対して伸びない。
- **改造の記録**: 何を試して何点だったか(不採用含む)を残す。次のコンペで効く。

## 4. 提出とスコア確認

- CSV 提出: `kaggle competitions submit <slug> -f submission.csv -m "<手法 + CV>"`
- **コードコンペ / Late Submission**: カーネルを push → 実行完了を待つ → `-k` で提出。
  手順・注意点は `references/code-competition.md`。
- スコア: `kaggle competitions submissions -c <slug> | head`
- 順位: `kaggle competitions leaderboard <slug> --download` は **Public** の順位表。
  終了後の Private 順位は Web で確認する。
- 提出は外向きの操作。ユーザーが明示したときだけ行う。

## 5. GPU(Windows PC)を使うとき

クラウドのセッションからは PC の GPU に届かない。GPU 学習は PC 上の Claude Code で回し、
コード共有は git、データは各自 `kaggle` CLI で取得する。セットアップと役割分担は
`references/gpu-windows.md`。

## 6. 振り返りの残し方(コンペごと)

コンペが一段落したら `references/lessons-<コンペ名>.md` を作り、以下を書いてこのファイルの
表に1行足す: データ構造、評価指標、検証方法、試した手法と CV/LB(不採用も)、効いたこと、
次の課題。コードは `rogii/kernel/` のように残す。

## 7. 過去コンペ

| コンペ | 種類 | 結果 | 詳細 |
| --- | --- | --- | --- |
| House Prices | 入門・表形式回帰・CSV 提出 | Public 0.12236(上位16%) | `references/lessons-house-prices.md` |
| ROGII Wellbore Geology | 賞金$50k・コードコンペ・回帰 | 本番 Private 12.762 → Late 10.470 | `references/lessons-rogii.md` |

再利用できるコード: 表形式テンプレート `src/`(使い方 `docs/TEMPLATE.md`)、
コードコンペ用カーネル例 `rogii/kernel/rogii_geosteer.py`。
