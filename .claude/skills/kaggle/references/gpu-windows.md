# Windows PC の GPU を Kaggle に使う

## 前提

- クラウドの Claude Code セッションからは、手元 PC の GPU に届かない。
- GPU を使う学習は **PC 上で動く Claude Code** で行う(Claude デスクトップアプリ、または
  PC のターミナルで作業フォルダに入って `claude remote-control` を起動すると、Claude アプリから
  その PC のセッションとして操作できる)。
- コードは git で共有、データは各環境で `kaggle` CLI から取得、鍵は各環境に個別に置く。

## 役割分担

| どこで | 何をする |
| --- | --- |
| クラウドのセッション | コンペ調査、データ点検、検証設計、CPU で回るベースライン、記事・スキル更新、git |
| **Windows PC(GPU)** | 深層学習、重い GBDT、ハイパーパラメータ探索。OOF と予測を保存 |
| Kaggle Notebook | コードコンペの最終提出(インターネット不可。重みは非公開 Dataset で渡す)。無料 GPU 枠もある |

GPU が効くのは画像・テキスト・音声・長い系列の深層学習と、数十万行以上の GBDT。
数千行規模の表形式は CPU で十分(むしろ速い)。**GPU も必ず CPU と時間を測って比べる。**

## セットアップ(WSL2 推奨)

Python の機械学習ライブラリは Linux の方がつまずきにくいので、WSL2 の Ubuntu で動かす。

1. Windows 側に最新の NVIDIA ドライバを入れる。**WSL の中に Linux 用ドライバは入れない**
   (WSL2 は Windows のドライバ経由で GPU を使う)。
2. PowerShell(管理者)で `wsl --install` → 再起動 → Ubuntu を初期設定。
3. Ubuntu で `nvidia-smi` を実行し、GPU 名と VRAM が見えることを確認。
4. Python 環境(例: `uv` か `conda`)を作る。リポジトリを clone。
5. ライブラリ:
   - PyTorch: pytorch.org の Get Started で「Linux / pip / CUDA」のコマンドを確認して入れる。
     `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`
   - XGBoost: `XGBRegressor(device="cuda", tree_method="hist")`
   - CatBoost: `CatBoostRegressor(task_type="GPU")`
   - LightGBM の GPU 版は環境依存でつまずきやすい。GPU で木モデルを回すなら XGBoost / CatBoost を優先。
6. Claude Code を Ubuntu 内にインストールし、リポジトリのフォルダで起動する。
7. Kaggle の鍵: `~/.kaggle/kaggle.json` に置いて `chmod 600`(または環境変数)。
   **値を画面やチャットに出さない。** 確認は `kaggle competitions list -p 1 | head -2` だけ。
8. このスキルを全プロジェクトで使うなら `~/.claude/skills/kaggle/` にコピーする
   (リポジトリの `.claude/skills/kaggle/` が元)。

## 運用のコツ

- データは WSL 側のディスク(例: `~/kaggle/data`)に置く。`/mnt/c/...` 越しは遅い。
- 長い学習はログをファイルに出し、途中経過(fold ごとの CV)を書く。チェックポイントを保存。
- `nvidia-smi` で VRAM を見ながらバッチサイズを決める。混合精度(`torch.autocast`)で節約。
- 学習済みの重み・OOF・予測は git に入れない。提出用の重みは非公開 Kaggle Dataset に上げる
  (`references/code-competition.md`)。
- 乱数シードを固定し、どの設定で何点だったかを記録する。
