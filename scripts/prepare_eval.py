#!/usr/bin/env python3
"""Prepare the downloaded Video-MME sample for full evaluation.

Assumes videos live under data/video-mme-sample10/ (see scripts/download_dataset.py).
Uses the shipped 5-class splits in data/splits/ (Multilingual already dropped).

If splits are missing, rebuilds them from qa/test.jsonl (6-class) then drops
Multilingual. Checks that every split video_id has a matching .mp4.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "video-mme-sample10"
SPLITS = ROOT / "data" / "splits"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET)
    parser.add_argument("--split-dir", type=Path, default=SPLITS)
    parser.add_argument("--rebuild-splits", action="store_true",
                        help="Rebuild splits from qa/test.jsonl and drop Multilingual.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_dir = args.dataset_dir / "videos"
    qa_path = args.dataset_dir / "qa" / "test.jsonl"
    if not video_dir.is_dir() or not any(video_dir.glob("*.mp4")):
        raise SystemExit(
            f"no videos under {video_dir}\n"
            "Download first:\n"
            "  python scripts/download_dataset.py\n"
        )
    if not qa_path.is_file():
        raise SystemExit(f"missing {qa_path}")

    if args.rebuild_splits or not (args.split_dir / "all.jsonl").is_file():
        print("building stratified splits ...")
        subprocess.check_call(
            [sys.executable, str(ROOT / "data" / "dataset_split_process.py"),
             "--dataset-dir", str(args.dataset_dir), "--output-dir", str(args.split_dir)],
            cwd=ROOT,
        )
        backup = args.split_dir.parent / "splits_6class"
        if backup.exists():
            # drop_domain refuses to overwrite an existing backup; use a fresh name once.
            print(f"note: {backup} already exists; dropping Multilingual in place without new backup")
            # Inline filter without backup when splits_6class exists
            for name in ("all", "train", "val", "test"):
                path = args.split_dir / f"{name}.jsonl"
                rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                kept = [row for row in rows if row.get("domain") != "Multilingual"]
                path.write_text("".join(json.dumps(row) + "\n" for row in kept))
        else:
            subprocess.check_call(
                [sys.executable, str(ROOT / "data" / "drop_domain.py"), "Multilingual",
                 "--split-dir", str(args.split_dir), "--backup-dir", str(backup)],
                cwd=ROOT,
            )

    rows = [json.loads(line) for line in (args.split_dir / "all.jsonl").read_text().splitlines() if line.strip()]
    missing = [row["video_id"] for row in rows if not (video_dir / f"{row['video_id']}.mp4").is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} split videos missing under {video_dir}: {missing[:5]}...")
    domains = sorted({row["domain"] for row in rows})
    print(f"ready: {len(rows)} videos, domains={domains}")
    print(f"  videos: {video_dir}")
    print(f"  splits: {args.split_dir}")
    print("run full eval:\n  python src/main.py")


if __name__ == "__main__":
    main()
