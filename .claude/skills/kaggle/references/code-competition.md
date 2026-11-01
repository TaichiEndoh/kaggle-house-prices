# コードコンペの提出手順(Late Submission も同じ)

コードコンペは CSV を直接アップロードできない。ノートブック(カーネル)を Kaggle 上で動かし、
その出力 `submission.csv` を提出する。採点時は非公開 test に差し替えて**再実行**される。

## 見分け方

- ローカルの `test/` がごく少数(ROGII は3坑井で、中身は train と同じ ID)= 本物の test は非公開。
- Overview に「Notebooks only」「Internet disabled」「runtime ≤ N 時間」などの記載。

## スクリプトの作り方

- 1ファイルで完結させる(学習→特徴量→予測→`/kaggle/working/submission.csv`)。
- データの場所は決め打ちせず探す:
  `glob("/kaggle/input/**/sample_submission.csv", recursive=True)` の親で `train/` があるもの。
  ローカル実行用に引数でルートを渡せるようにしておく。
- インターネットは使えない。`pip install` も不可。Kaggle 環境に入っているライブラリだけ使う
  (numpy, pandas, scipy, scikit-learn, lightgbm, xgboost, catboost, torch など)。
- GPU で学習した重みを使う場合は、重みを**非公開 Kaggle Dataset** にして `dataset_sources` で付ける:
  `kaggle datasets init -p <dir>` → `dataset-metadata.json` を編集 → `kaggle datasets create -p <dir>`。
- 提出前に、ローカルの test で最後まで動き、`sample_submission.csv` と ID が一致し、欠損が無いことを確認する。

## kernel-metadata.json(例)

```json
{
  "id": "<username>/<slug>",
  "title": "<title>",
  "code_file": "<script>.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": false,
  "enable_internet": false,
  "dataset_sources": [],
  "competition_sources": ["<competition-slug>"],
  "kernel_sources": []
}
```

- `title` は `id` の slug と一致させる(例: id `user/rogii-geosteer-late` → title `ROGII Geosteer Late`)。
- `enable_gpu` は GPU 推論が必要なときだけ `true`。
- 実例: `rogii/kernel/kernel-metadata.json`。

## 実行と提出

```bash
kaggle kernels push -p <dir>                       # バージョンが1つ増える
# 完了待ち(バックグラウンドで。sleep 連打はしない)
until s=$(kaggle kernels status <user>/<slug>); echo "$s" | grep -qiE "complete|error|cancel"; do sleep 30; done; echo "$s"
# 提出(-v は push したバージョン番号)
kaggle competitions submit <competition> -k <user>/<slug> -v <N> -f submission.csv -m "<手法 + CV>" --wait 7200
kaggle competitions submissions -c <competition> | head
```

- 実行が失敗したら `kaggle kernels output <user>/<slug> -p <dir>` でログを取る。
- 締切後は Late Submission として採点だけされ、順位には入らない(実力の測り直しや記事のネタに使える)。
