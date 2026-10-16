#!/usr/bin/env python3
"""Download the Video-MME 10% sample (videos + QA) from Hugging Face.

Dataset: https://huggingface.co/datasets/aintropy-ai/video-mme-sample10
(~11 GB of videos). Idempotent: skips files that are already present.
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = ROOT / "data" / "video-mme-sample10"
REPO = "aintropy-ai/video-mme-sample10"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--repo", default=REPO)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from huggingface_hub import snapshot_download

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"downloading {args.repo} → {dest} (large; may take a while) ...")
    snapshot_download(repo_id=args.repo, repo_type="dataset", local_dir=str(dest))
    videos = dest / "videos"
    n = len(list(videos.glob("*.mp4"))) if videos.is_dir() else 0
    print(f"done: {n} videos under {videos}")


if __name__ == "__main__":
    main()
