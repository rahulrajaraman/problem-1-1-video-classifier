#!/usr/bin/env python3
"""Classify video domains zero-shot with the Qwen2.5-Omni Thinker.

Each clip's frames and audio go to the Thinker with a question that lists the
domain names. The answer is read from the first answer token: every domain
name starts with a different token, so the domain whose first token has the
highest log-probability wins. With several clips, log-probabilities are
averaged over clips. Nothing is trained, so every split is evaluated.

Results are appended per video to a JSONL file, so an interrupted run resumes
where it stopped.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from clip_sampling import ClipSampling, sample_clips
from dataset import label_map, read_jsonl
from omni_encoder import OmniSettings
from train import DEFAULT_SPLIT_DIR, DEFAULT_VIDEO_DIR, ROOT, pick_device

SPLITS = ("train", "val", "test")
DURATIONS = ("short", "medium", "long")


def build_question(names: list[str]) -> str:
    return (
        "Which category best describes this video? "
        f"Options: {', '.join(names)}. "
        "Answer with the category name only."
    )


def first_token_ids(tokenizer, names: list[str]) -> list[int]:
    ids = [tokenizer.encode(name, add_special_tokens=False)[0] for name in names]
    if len(set(ids)) != len(ids):
        pieces = {name: tokenizer.convert_ids_to_tokens(token) for name, token in zip(names, ids)}
        raise SystemExit(f"domain names share a first token, so first-token scoring is ambiguous: {pieces}")
    return ids


def accuracy_line(records: list[dict]) -> str:
    hits = sum(record["prediction"] == record["domain"] for record in records)
    return f"{hits}/{len(records)} = {hits / len(records):.3f}" if records else "n/a"


def report(records: list[dict], names: list[str]) -> None:
    print("\naccuracy")
    print(f"  all:  {accuracy_line(records)}")
    for split in SPLITS:
        print(f"  {split}: {accuracy_line([r for r in records if r['split'] == split])}")
    print("by duration (all splits)")
    for duration in DURATIONS:
        print(f"  {duration}: {accuracy_line([r for r in records if r['duration'] == duration])}")

    test = [r for r in records if r["split"] == "test"]
    print("by class: test | all")
    for name in names:
        print(f"  {name}: {accuracy_line([r for r in test if r['domain'] == name])} | "
              f"{accuracy_line([r for r in records if r['domain'] == name])}")

    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in records:
        confusion[record["domain"]][record["prediction"]] += 1
    print("confusion (rows are true domains, columns are predictions, all splits)")
    short = [name.split()[0][:8] for name in names]
    print("  " + " " * 22 + " ".join(f"{label:>8}" for label in short))
    for name in names:
        print(f"  {name:<22}" + " ".join(f"{confusion[name][column]:>8}" for column in names))
    mass = sum(record["label_mass"] for record in records) / len(records)
    print(f"mean probability on the six domain tokens: {mass:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "zero_shot")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--clip-len", type=int, default=30)
    parser.add_argument("--num-clips", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--omni-max-tokens", type=int, default=64, help="Upper bound on Omni vision tokens per frame pair.")
    parser.add_argument("--omni-no-audio", action="store_true", help="Feed Omni the frames only.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-videos", type=int, default=0, help="Use the first N videos of each split. 0 means all.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    sampling = ClipSampling(args.clip_len, args.num_clips, args.seed)
    settings = OmniSettings(
        max_pixels=args.omni_max_tokens * 28 * 28,
        min_pixels=min(args.omni_max_tokens, 32) * 28 * 28,
        use_audio=not args.omni_no_audio,
    )

    names = list(label_map(read_jsonl(args.split_dir / "all.jsonl"), "domain"))
    rows = []
    for split in args.splits:
        split_rows = read_jsonl(args.split_dir / f"{split}.jsonl")
        if args.max_videos > 0:
            split_rows = split_rows[: args.max_videos]
        rows.extend({**row, "split": split} for row in split_rows)

    audio = "av" if settings.use_audio else "v"
    out_path = args.out_dir / f"qwen25-omni-3b_{audio}_p{args.omni_max_tokens}_{sampling.tag()}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {record["video_id"]: record for record in read_jsonl(out_path)} if out_path.is_file() else {}
    pending = [row for row in rows if row["video_id"] not in done]
    question = build_question(names)
    print(f"device={device} clips={sampling.tag()} audio={settings.use_audio} videos={len(rows)} pending={len(pending)}")
    print(f"question: {question}")

    if pending:
        from omni_encoder import OmniClipEncoder

        encoder = OmniClipEncoder(settings, device, keep_lm_head=True)
        token_ids = torch.tensor(first_token_ids(encoder.processor.tokenizer, names))
        with out_path.open("a") as handle:
            for index, row in enumerate(pending, start=1):
                video_path = args.video_dir / f"{row['video_id']}.mp4"
                clips = sample_clips(video_path, sampling, row["video_id"])
                log_probs = torch.stack([
                    torch.log_softmax(encoder.next_token_logits(video_path, clip, question), dim=-1)[token_ids]
                    for clip in clips
                ])
                if device.type == "mps":
                    torch.mps.empty_cache()
                scores = log_probs.mean(dim=0)
                prediction = names[int(scores.argmax())]
                record = {
                    **row,
                    "prediction": prediction,
                    "scores": {name: round(float(score), 4) for name, score in zip(names, scores)},
                    "label_mass": round(float(log_probs.exp().sum(dim=1).mean()), 4),
                    "clip_starts_sec": [round(clip.start_sec, 2) for clip in clips],
                }
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                done[row["video_id"]] = record
                mark = "ok " if prediction == row["domain"] else "MISS"
                print(f"{index}/{len(pending)} {mark} {row['video_id']} [{row['split']}/{row['duration']}] "
                      f"true={row['domain']} pred={prediction}", flush=True)

    report([done[row["video_id"]] for row in rows if row["video_id"] in done], names)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
