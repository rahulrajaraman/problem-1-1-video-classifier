"""Remove one domain from every split file, keeping a copy of the original splits.

Usage: python data/drop_domain.py Multilingual
"""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "splits")
    parser.add_argument("--backup-dir", type=Path, default=ROOT / "splits_6class")
    return parser.parse_args()


def filter_jsonl(path: Path, domain: str, keep_ids: set[str] | None = None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if keep_ids is None:
        kept = [row for row in rows if row.get("domain") != domain]
    else:
        kept = [row for row in rows if row["video_id"] in keep_ids]
    path.write_text("".join(json.dumps(row) + "\n" for row in kept))
    return kept


def main() -> None:
    args = parse_args()
    if args.backup_dir.exists():
        raise SystemExit(f"{args.backup_dir} already exists; refusing to overwrite the backup")
    shutil.copytree(args.split_dir, args.backup_dir)

    kept = {name: filter_jsonl(args.split_dir / f"{name}.jsonl", args.domain) for name in ("all", "train", "val", "test")}
    keep_ids = {row["video_id"] for row in kept["all"]}
    questions = args.split_dir / "questions"
    question_rows = {name: filter_jsonl(questions / f"{name}.jsonl", args.domain, keep_ids) for name in ("train", "val", "test")}

    manifest_path = args.split_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["dropped_domains"] = sorted(set(manifest.get("dropped_domains", [])) | {args.domain})
    manifest["n_videos"] = {name: len(kept[name]) for name in ("train", "val", "test")}
    manifest["by_duration"] = {
        duration: {name: sum(row["duration"] == duration for row in kept[name]) for name in ("train", "val", "test")}
        for duration in manifest["by_duration"]
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    manifest_path = questions / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["dropped_domains"] = sorted(set(manifest.get("dropped_domains", [])) | {args.domain})
    manifest["videos"] = [video for video in manifest["videos"] if video["domain"] != args.domain]
    manifest["n_videos"] = {name: sum(video["split"] == name for video in manifest["videos"]) for name in question_rows}
    manifest["n_questions"] = {name: len(rows) for name, rows in question_rows.items()}
    manifest["counts"] = {
        field: {
            value: {name: sum(video[field] == value and video["split"] == name for video in manifest["videos"]) for name in question_rows}
            for value in sorted({video[field] for video in manifest["videos"]})
        }
        for field in manifest["counts"]
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"backup: {args.backup_dir}")
    for name in ("train", "val", "test", "all"):
        counts = Counter(row["domain"] for row in kept[name])
        print(f"{name:>5} {len(kept[name]):>3}  " + ", ".join(f"{d} {c}" for d, c in sorted(counts.items())))


if __name__ == "__main__":
    main()
