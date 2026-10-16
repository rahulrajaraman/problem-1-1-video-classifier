#!/usr/bin/env python3
"""Split videos into train, validation, and test.

Each duration bucket (short, medium, long) is split on its own at 50% train,
20% validation, and 30% test, stratified by domain. Rows keep only video_id,
domain, sub_category, and duration.

The question file at video-mme-sample10/qa/test.jsonl is left unchanged.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

SPLITS = ("train", "val", "test")
DURATIONS = ("short", "medium", "long")
FIELDS = ("video_id", "domain", "sub_category", "duration")
# On a tied remainder, give the extra video to test, then validation.
TIE_BREAK = {"test": 3, "val": 2, "train": 1}

DEFAULT_FRACTIONS = {"train": 0.50, "val": 0.20, "test": 0.30}
DATA_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = DATA_DIR / "video-mme-sample10"
DEFAULT_OUTPUT = DATA_DIR / "splits"


def apportion(n: int, fractions: dict[str, float]) -> dict[str, int]:
    """Turn a count into integer split sizes that sum to n."""
    raw = {split: n * fractions[split] for split in SPLITS}
    sizes = {split: int(raw[split]) for split in SPLITS}
    leftover = n - sum(sizes.values())
    order = sorted(
        SPLITS,
        key=lambda split: (raw[split] - sizes[split], TIE_BREAK[split]),
        reverse=True,
    )
    for split in order[:leftover]:
        sizes[split] += 1
    return sizes


def load_videos(dataset_dir: Path) -> list[dict]:
    qa_path = dataset_dir / "qa" / "test.jsonl"
    by_video: dict[str, dict] = {}
    for line in qa_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        video = by_video.setdefault(
            row["video_id"],
            {field: row[field] for field in FIELDS},
        )
        for field in FIELDS:
            if video[field] != row[field]:
                raise ValueError(f"{row['video_id']} has inconsistent {field}")
    videos = list(by_video.values())
    videos.sort(key=lambda video: video["video_id"])
    return videos


def split_bucket(videos: list[dict], fractions: dict[str, float], rng: random.Random) -> dict[str, str]:
    """Assign one duration bucket, stratified by domain, within the bucket quotas."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for video in videos:
        groups[video["domain"]].append(video)
    for group in groups.values():
        rng.shuffle(group)

    bucket_target = apportion(len(videos), fractions)
    domain_target = {domain: apportion(len(group), fractions) for domain, group in groups.items()}
    bucket_count: Counter[str] = Counter()
    domain_count: dict[str, Counter[str]] = defaultdict(Counter)
    assignment: dict[str, str] = {}

    # Larger domains go first so their 50/20/30 split is met. Smaller domains
    # fill the bucket slots that remain.
    for domain in sorted(groups, key=lambda name: (-len(groups[name]), name)):
        for video in groups[domain]:
            candidates = [split for split in SPLITS if bucket_count[split] < bucket_target[split]]
            if not candidates:
                raise RuntimeError(f"no split left for {video['video_id']}")

            def score(split: str, domain: str = domain) -> tuple[int, int, int]:
                return (
                    domain_target[domain][split] - domain_count[domain][split],
                    bucket_target[split] - bucket_count[split],
                    TIE_BREAK[split],
                )

            split = max(candidates, key=score)
            assignment[video["video_id"]] = split
            bucket_count[split] += 1
            domain_count[domain][split] += 1
    return assignment


def assign_splits(videos: list[dict], fractions: dict[str, float], seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    assignment: dict[str, str] = {}
    for duration in DURATIONS:
        bucket = [video for video in videos if video["duration"] == duration]
        assignment.update(split_bucket(bucket, fractions, rng))
    return assignment


def write_jsonl(path: Path, rows: list[dict]) -> None:
    lines = [json.dumps({field: row[field] for field in FIELDS}, ensure_ascii=False) for row in rows]
    path.write_text("\n".join(lines) + "\n")


def write_outputs(
    videos: list[dict],
    assignment: dict[str, str],
    fractions: dict[str, float],
    seed: int,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(videos, key=lambda video: video["video_id"])
    write_jsonl(output_dir / "all.jsonl", ordered)
    for split in SPLITS:
        rows = [video for video in ordered if assignment[video["video_id"]] == split]
        write_jsonl(output_dir / f"{split}.jsonl", rows)

    manifest = {
        "seed": seed,
        "fractions": fractions,
        "n_videos": {split: sum(assignment[video["video_id"]] == split for video in videos) for split in SPLITS},
        "by_duration": {
            duration: {
                split: sum(
                    video["duration"] == duration and assignment[video["video_id"]] == split
                    for video in videos
                )
                for split in SPLITS
            }
            for duration in DURATIONS
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def print_summary(videos: list[dict], assignment: dict[str, str]) -> None:
    print("videos", len(videos))
    for duration in DURATIONS:
        counts = {
            split: sum(video["duration"] == duration and assignment[video["video_id"]] == split for video in videos)
            for split in SPLITS
        }
        print(f"  {duration:6} " + "  ".join(f"{split}={counts[split]}" for split in SPLITS))
    print("domain within each duration")
    for duration in DURATIONS:
        print(f"  {duration}")
        domains = sorted({video["domain"] for video in videos if video["duration"] == duration})
        for domain in domains:
            counts = {
                split: sum(
                    video["duration"] == duration
                    and video["domain"] == domain
                    and assignment[video["video_id"]] == split
                    for video in videos
                )
                for split in SPLITS
            }
            total = sum(counts.values())
            print(f"    {domain:22} n={total:<2} " + "  ".join(f"{split}={counts[split]}" for split in SPLITS))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train", type=float, default=DEFAULT_FRACTIONS["train"])
    parser.add_argument("--val", type=float, default=DEFAULT_FRACTIONS["val"])
    parser.add_argument("--test", type=float, default=DEFAULT_FRACTIONS["test"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fractions = {"train": args.train, "val": args.val, "test": args.test}
    if abs(sum(fractions.values()) - 1.0) > 1e-6:
        raise SystemExit(f"split fractions must sum to 1, got {sum(fractions.values()):.4f}")
    if any(fraction <= 0 for fraction in fractions.values()):
        raise SystemExit("each split fraction must be positive")

    videos = load_videos(args.dataset_dir)
    assignment = assign_splits(videos, fractions, args.seed)
    write_outputs(videos, assignment, fractions, args.seed, args.output_dir)
    print_summary(videos, assignment)
    print(f"\nWrote {args.output_dir / 'all.jsonl'} and train/val/test.jsonl")


if __name__ == "__main__":
    main()
