#!/usr/bin/env python3
"""Classify video domains zero-shot with jev-visual (Qwen3.5-0.8B, 4-bit, MLX).

jev-visual judges one image per request, so each sampled clip becomes a grid
of frames spread evenly across the clip. The model answers a multiple-choice
question whose options are the six domains with short descriptions, and its
candidate letter scores are read straight from the logits. Scores are turned
into log-probabilities over the six options and averaged over a video's clips.

Run it with the jev-visual environment, which has MLX but no torch:

    ../jev-visual/.venv/bin/python src/jev_zero_shot.py

Results are appended per video, so an interrupted run resumes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

from clip_sampling import ClipSampling, sample_clips

ROOT = Path(__file__).resolve().parents[1]
JEV_ROOT = ROOT.parent / "jev-visual"
SPLITS = ("train", "val", "test")
DURATIONS = ("short", "medium", "long")

# Descriptions follow the Video-MME taxonomy of domains and their sub-fields.
PROMPTS = {
    "v1": {
        "Artistic Performance": "A stage performance: acrobatics, a magic show, a stage play, or a variety show.",
        "Film & Television": "Produced screen content: an animation, a documentary, a movie or TV show, or a news report.",
        "Knowledge": "Educational content explaining a topic: astronomy, biology, medicine, finance, geography, history, literature, art, technology, or life tips.",
        "Life Record": "Personal everyday footage or vlogs: daily life, exercise, fashion, food, handicraft, or travel.",
        "Multilingual": "A video centred on languages, translation, or speakers of several languages.",
        "Sports Competition": "A sports match or competition: athletics, basketball, esports, football, or other sports.",
    },
    # v1 listed exercise under Life Record and called sports a "match", so
    # practice sessions and athletics went to Life Record.
    "v2": {
        "Artistic Performance": "A stage performance: acrobatics, a magic show, a stage play, or a variety show.",
        "Film & Television": "Produced screen content: an animation, a documentary, a movie or TV show, or a news report.",
        "Knowledge": "Educational content explaining a topic: astronomy, biology, medicine, finance, geography, history, literature, art, technology, or life tips.",
        "Life Record": "A personal vlog of everyday life: a home workout, fashion, cooking, handicraft, or travel.",
        "Multilingual": "A video centred on languages, translation, or speakers of several languages.",
        "Sports Competition": "Any sports footage, including games, practice, and training: athletics, running, basketball, football, esports, or other sports.",
    },
}
QUESTION = "These frames come from one short clip of a video. Which category does the video belong to?"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def split_prompts(prompt: str, split_dir: Path) -> dict[str, str]:
    """Domain prompts restricted to the domains present in the splits."""
    present = {row["domain"] for row in read_jsonl(Path(split_dir) / "all.jsonl")}
    return {name: text for name, text in PROMPTS[prompt].items() if name in present}


def frame_grid(frames: list[Image.Image], side: int, cell_width: int) -> Image.Image:
    """`side` x `side` frames spaced evenly through the clip, in reading order."""
    count = side * side
    picks = [frames[round(i * (len(frames) - 1) / max(1, count - 1))] for i in range(count)]
    cell_height = round(cell_width * picks[0].height / picks[0].width)
    grid = Image.new("RGB", (side * cell_width, side * cell_height))
    for index, frame in enumerate(picks):
        cell = frame.resize((cell_width, cell_height), Image.BILINEAR)
        grid.paste(cell, ((index % side) * cell_width, (index // side) * cell_height))
    return grid


def log_softmax(values: list[float]) -> list[float]:
    top = max(values)
    total = math.log(sum(math.exp(value - top) for value in values)) + top
    return [value - total for value in values]


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
    print("  " + " " * 22 + " ".join(f"{name.split()[0][:8]:>8}" for name in names))
    for name in names:
        print(f"  {name:<22}" + " ".join(f"{confusion[name][column]:>8}" for column in names))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split-dir", type=Path, default=ROOT / "data" / "splits")
    parser.add_argument("--video-dir", type=Path, default=ROOT / "data" / "video-mme-sample10" / "videos")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "zero_shot")
    parser.add_argument("--grid-dir", type=Path, default=ROOT / "data" / "debug" / "jev_grids")
    parser.add_argument("--model-path", type=Path, default=JEV_ROOT / ".models" / "Qwen3.5-0.8B-4bit")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--clip-len", type=int, default=60)
    parser.add_argument("--num-clips", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", choices=sorted(PROMPTS), default="v2", help="Domain descriptions shown to the model.")
    parser.add_argument("--grid", type=int, default=2, help="Frames per clip image are grid x grid.")
    parser.add_argument("--cell-width", type=int, default=384)
    parser.add_argument("--max-videos", type=int, default=0, help="Use the first N videos of each split. 0 means all.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(JEV_ROOT))
    from jev_visual.engine import Engine
    from jev_visual.schema import Question, Request

    sampling = ClipSampling(args.clip_len, args.num_clips, args.seed)
    domains = split_prompts(args.prompt, args.split_dir)
    names = sorted(domains)
    rows = []
    for split in args.splits:
        split_rows = read_jsonl(args.split_dir / f"{split}.jsonl")
        if args.max_videos > 0:
            split_rows = split_rows[: args.max_videos]
        rows.extend({**row, "split": split} for row in split_rows)

    tag = f"jev-qwen35-08b-4bit_g{args.grid}_{sampling.tag()}"
    # v1 results predate the prompt option and keep their original file name.
    out_path = args.out_dir / (f"{tag}.jsonl" if args.prompt == "v1" else f"{tag}_prompt-{args.prompt}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid_dir = args.grid_dir / tag
    grid_dir.mkdir(parents=True, exist_ok=True)
    done = {record["video_id"]: record for record in read_jsonl(out_path)} if out_path.is_file() else {}
    pending = [row for row in rows if row["video_id"] not in done]
    print(f"clips={sampling.tag()} grid={args.grid}x{args.grid} videos={len(rows)} pending={len(pending)}")

    if pending:
        engine = Engine(model_path=args.model_path)
        question = Question(type="choice", instructions=QUESTION, criteria={name: domains[name] for name in names})
        with out_path.open("a") as handle:
            for index, row in enumerate(pending, start=1):
                video_path = args.video_dir / f"{row['video_id']}.mp4"
                clips = sample_clips(video_path, sampling, row["video_id"])
                per_clip = []
                for clip_index, clip in enumerate(clips):
                    image_path = grid_dir / f"{row['video_id']}_c{clip_index}_{clip.start_sec:08.2f}s.jpg"
                    frame_grid(clip.frames, args.grid, args.cell_width).save(image_path, quality=90)
                    result = engine.judge(Request(image=str(image_path), questions={"domain": question}))
                    per_clip.append(log_softmax(result["answers"]["domain"]["candidate_scores"]))
                scores = [sum(clip[i] for clip in per_clip) / len(per_clip) for i in range(len(names))]
                prediction = names[max(range(len(names)), key=scores.__getitem__)]
                record = {
                    **row,
                    "prediction": prediction,
                    "scores": {name: round(score, 4) for name, score in zip(names, scores)},
                    "clip_predictions": [names[max(range(len(names)), key=clip.__getitem__)] for clip in per_clip],
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
