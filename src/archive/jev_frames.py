#!/usr/bin/env python3
"""Run frozen jev-visual (Qwen3.5-0.8B, 4-bit, MLX) on frames sampled across the whole video.

Frames come from the same sampler as the first MobileCLIP experiments:
uniform plus one per interval, capped (preprocessing.FrameSampling). Each run
of grid x grid consecutive sampled frames becomes one image, so the images
cover the video from start to end in order.

One pass per image gives both
  * zero-shot scores: the six option-letter logits as log-probabilities,
    averaged over a video's images, and
  * probe features: float16 [images, 2, layers + 1, hidden] for probe.py.

Run it with the jev-visual environment:

    ../jev-visual/.venv/bin/python src/jev_frames.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from jev_features import image_forward
from jev_zero_shot import JEV_ROOT, PROMPTS, ROOT, frame_grid, log_softmax, read_jsonl, report, split_prompts
from preprocessing import FrameSampling, sample_video_frames

SPLITS = ("train", "val", "test")
FRAMES_QUESTION = "These frames are sampled in order from one video. Which category does the video belong to?"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split-dir", type=Path, default=ROOT / "data" / "splits")
    parser.add_argument("--video-dir", type=Path, default=ROOT / "data" / "video-mme-sample10" / "videos")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "cache" / "embeddings")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "zero_shot")
    parser.add_argument("--grid-dir", type=Path, default=ROOT / "data" / "debug" / "jev_grids")
    parser.add_argument("--model-path", type=Path, default=JEV_ROOT / ".models" / "Qwen3.5-0.8B-4bit")
    parser.add_argument("--prompt", choices=sorted(PROMPTS), default="v2")
    parser.add_argument("--sampling", choices=["uniform", "interval", "both"], default="both")
    parser.add_argument("--num-frames", type=int, default=60)
    parser.add_argument("--interval-sec", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--grid", type=int, default=2, help="Each image is grid x grid consecutive sampled frames.")
    parser.add_argument("--cell-width", type=int, default=384)
    parser.add_argument("--max-videos", type=int, default=0, help="Use the first N videos of each split. 0 means all.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(JEV_ROOT))
    from jev_visual.engine import Engine
    from jev_visual.preprocessing import build_prompts, read_image
    from jev_visual.schema import Question, Request

    sampling = FrameSampling(args.sampling, args.num_frames, args.interval_sec, args.max_frames)
    domains = split_prompts(args.prompt, args.split_dir)
    names = sorted(domains)
    rows = []
    for split in SPLITS:
        split_rows = read_jsonl(args.split_dir / f"{split}.jsonl")
        if args.max_videos > 0:
            split_rows = split_rows[: args.max_videos]
        rows.extend({**row, "split": split} for row in split_rows)

    tag = f"jev-qwen35-08b-4bit_g{args.grid}_{sampling.tag()}_prompt-{args.prompt}"
    feature_dir = args.cache_dir / tag
    feature_dir.mkdir(parents=True, exist_ok=True)
    grid_dir = args.grid_dir / tag
    grid_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{tag}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {record["video_id"]: record for record in read_jsonl(out_path)} if out_path.is_file() else {}
    pending = [row for row in rows if row["video_id"] not in done or not (feature_dir / f"{row['video_id']}.npy").is_file()]
    print(f"run={tag} videos={len(rows)} pending={len(pending)}")

    if pending:
        adapter = Engine(model_path=args.model_path).adapter
        question = Question(type="choice", instructions=FRAMES_QUESTION, criteria={name: domains[name] for name in names})
        cells = args.grid * args.grid
        with out_path.open("a") as handle, adapter.execution_context():
            prompt = letter_ids = None
            for index, row in enumerate(pending, start=1):
                frames = sample_video_frames(args.video_dir / f"{row['video_id']}.mp4", sampling)
                # Pad the last group with its final frame so every image is a full grid.
                frames = frames + [frames[-1]] * (-len(frames) % cells)
                per_image, features = [], []
                for image_index in range(len(frames) // cells):
                    image = frame_grid(frames[image_index * cells:(image_index + 1) * cells], args.grid, args.cell_width)
                    image_path = grid_dir / f"{row['video_id']}_i{image_index:02d}.jpg"
                    image.save(image_path, quality=90)
                    if prompt is None:
                        prefix, plans = build_prompts(adapter.processor, Request(image=str(image_path), questions={"domain": question}))
                        prompt = prefix + plans[0].suffix
                        letter_ids = [target[0] for target in plans[0].targets]
                    pooled, letters = image_forward(adapter, prompt, read_image(str(image_path)), letter_ids)
                    features.append(pooled)
                    per_image.append(log_softmax(letters))
                np.save(feature_dir / f"{row['video_id']}.npy", np.stack(features))

                scores = [sum(image[i] for image in per_image) / len(per_image) for i in range(len(names))]
                prediction = names[max(range(len(names)), key=scores.__getitem__)]
                record = {
                    **row,
                    "prediction": prediction,
                    "scores": {name: round(score, 4) for name, score in zip(names, scores)},
                    "num_frames": len(frames),
                    "image_predictions": [names[max(range(len(names)), key=image.__getitem__)] for image in per_image],
                }
                if row["video_id"] not in done:
                    handle.write(json.dumps(record) + "\n")
                    handle.flush()
                done[row["video_id"]] = record
                mark = "ok " if prediction == row["domain"] else "MISS"
                print(f"{index}/{len(pending)} {mark} {row['video_id']} [{row['split']}/{row['duration']}] "
                      f"images={len(per_image)} true={row['domain']} pred={prediction}", flush=True)

    report([done[row["video_id"]] for row in rows if row["video_id"] in done], names)
    print(f"saved {out_path}")
    print(f"features {feature_dir}")


if __name__ == "__main__":
    main()
