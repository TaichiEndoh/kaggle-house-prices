"""Kaggle API を使ってコンペデータを data/ にダウンロードするスクリプト。

House Prices - Advanced Regression Techniques コンペの
train.csv / test.csv / sample_submission.csv を取得します。

認証方法(どちらか一方でOK):
  1. 環境変数  KAGGLE_USERNAME / KAGGLE_KEY を設定する
  2. ~/.kaggle/kaggle.json に認証情報を置く

使い方:
  python src/download_data.py
"""

import os
import zipfile
from pathlib import Path

import config as cfg

# このファイルから見たプロジェクトのルート(1つ上の階層)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# 対象コンペの識別子(config.py で一元管理)
COMPETITION = cfg.COMPETITION


def ensure_credentials() -> None:
    """Kaggle の認証情報があるかを確認する。

    環境変数 KAGGLE_USERNAME / KAGGLE_KEY が設定されていればそれを使い、
    なければ ~/.kaggle/kaggle.json の存在を確認します。
    どちらも無い場合は分かりやすいエラーメッセージを出して終了します。
    """
    has_env = os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"

    if has_env:
        print("環境変数 (KAGGLE_USERNAME / KAGGLE_KEY) の認証情報を使用します。")
        return

    if kaggle_json.exists():
        print(f"認証ファイルを使用します: {kaggle_json}")
        return

    raise SystemExit(
        "Kaggle の認証情報が見つかりません。\n"
        "次のいずれかを設定してください:\n"
        "  (1) 環境変数 KAGGLE_USERNAME と KAGGLE_KEY\n"
        "  (2) ~/.kaggle/kaggle.json (Kaggle のアカウント設定から取得)\n"
        "詳しくは README.md を参照してください。"
    )


def download() -> None:
    """コンペデータをダウンロードして data/ に展開する。"""
    ensure_credentials()

    # kaggle ライブラリは import 時に認証情報を読み込むため、
    # 認証確認のあとで import する。
    from kaggle.api.kaggle_api_extended import KaggleApi

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    print(f"コンペ '{COMPETITION}' のデータをダウンロードしています...")
    # ZIP でまとめてダウンロードされる
    api.competition_download_files(COMPETITION, path=str(DATA_DIR), quiet=False)

    # ダウンロードした ZIP を展開する
    zip_path = DATA_DIR / f"{COMPETITION}.zip"
    if zip_path.exists():
        print("ZIP ファイルを展開しています...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATA_DIR)
        zip_path.unlink()  # 展開後は ZIP を削除

    print("完了しました。data/ の中身:")
    for f in sorted(DATA_DIR.iterdir()):
        print(f"  - {f.name}")


if __name__ == "__main__":
    download()
