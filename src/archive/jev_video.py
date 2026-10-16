#!/usr/bin/env python3
"""Feed whole clips as video to the frozen jev-visual model (Qwen3.5-0.8B, 4-bit, MLX).

jev-visual itself scores one image per request. This script keeps its model,
prompt and letter scoring but passes every frame of each clip through the
model's native video path (frame pairs, timestamped). There is no audio: the
model has no audio encoder.

One forward pass per clip gives both
  * zero-shot scores: the logits of the six option letters at the answer
    position, as log-probabilities, averaged over a video's clips, and
  * probe features: every layer's hidden state, averaged over the video
    tokens and taken at the answer position, saved as float16
    [clips, 2, layers + 1, hidden] for probe.py.

Timestamps restart at zero for each clip so the model cannot read a clip's
position, and with it the video's length, from the prompt. Run it with the
jev-visual environment:

    ../jev-visual/.venv/bin/python src/jev_video.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from clip_sampling import ClipSampling, sample_clips
from jev_zero_shot import JEV_ROOT, PROMPTS, ROOT, log_softmax, read_jsonl, report, split_prompts

SPLITS = ("train", "val", "test")
VIDEO_QUESTION = "This is one short clip of a video. Which category does the video belong to?"


def resize_frames(frames: list[Image.Image], pixels: int) -> np.ndarray:
    """Scale every frame to about `pixels` in area, keeping the aspect ratio."""
    width, height = frames[0].size
    scale = (pixels / (width * height)) ** 0.5
    size = (max(32, round(width * scale)), max(32, round(height * scale)))
    return np.stack([np.asarray(frame.resize(size, Image.BILINEAR)) for frame in frames])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split-dir", type=Path, default=ROOT / "data" / "splits")
    parser.add_argument("--video-dir", type=Path, default=ROOT / "data" / "video-mme-sample10" / "videos")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "cache" / "embeddings")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "zero_shot")
    parser.add_argument("--model-path", type=Path, default=JEV_ROOT / ".models" / "Qwen3.5-0.8B-4bit")
    parser.add_argument("--prompt", choices=sorted(PROMPTS), default="v2")
    parser.add_argument("--clip-len", type=int, default=90)
    parser.add_argument("--num-clips", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frame-pixels", type=int, default=320 * 180,
                        help="Area each frame is scaled to. 320x180 gives 60 tokens per frame pair.")
    parser.add_argument("--max-videos", type=int, default=0, help="Use the first N videos of each split. 0 means all.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(JEV_ROOT))
    from jev_visual.engine import Engine
    from jev_visual.preprocessing import build_prompts
    from jev_visual.schema import Question, Request
    from mlx_vlm.utils import prepare_inputs

    sampling = ClipSampling(args.clip_len, args.num_clips, args.seed)
    domains = split_prompts(args.prompt, args.split_dir)
    names = sorted(domains)
    rows = []
    for split in SPLITS:
        split_rows = read_jsonl(args.split_dir / f"{split}.jsonl")
        if args.max_videos > 0:
            split_rows = split_rows[: args.max_videos]
        rows.extend({**row, "split": split} for row in split_rows)

    tag = f"jev-qwen35-08b-4bit_video{args.frame_pixels // 1000}k_{sampling.tag()}_prompt-{args.prompt}"
    feature_dir = args.cache_dir / tag
    feature_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{tag}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {record["video_id"]: record for record in read_jsonl(out_path)} if out_path.is_file() else {}
    pending = [row for row in rows if row["video_id"] not in done or not (feature_dir / f"{row['video_id']}.npy").is_file()]
    print(f"run={tag} videos={len(rows)} pending={len(pending)}")

    if pending:
        adapter = Engine(model_path=args.model_path).adapter
        mx, model = adapter.mx, adapter.model
        language = model.language_model
        num_layers = model.config.text_config.num_hidden_layers
        video_token = model.config.video_token_id

        question = Question(type="choice", instructions=VIDEO_QUESTION, criteria={name: domains[name] for name in names})
        # build_prompts needs an image path only to validate the request; the
        # image placeholder it renders is swapped for a video placeholder.
        placeholder = next((ROOT / "data" / "debug" / "jev_grids").rglob("*.jpg"))
        prefix, plans = build_prompts(adapter.processor, Request(image=str(placeholder), questions={"domain": question}))
        if "<|image_pad|>" not in prefix:
            raise SystemExit("unexpected prompt template: no image placeholder to replace")
        prompt = prefix.replace("<|image_pad|>", "<|video_pad|>") + plans[0].suffix
        letter_ids = mx.array([target[0] for target in plans[0].targets])

        with out_path.open("a") as handle, adapter.execution_context():
            for index, row in enumerate(pending, start=1):
                video_path = args.video_dir / f"{row['video_id']}.mp4"
                clips = sample_clips(video_path, sampling, row["video_id"])
                per_clip, features = [], []
                for clip in clips:
                    adapter.reset()
                    frames = resize_frames(clip.frames, args.frame_pixels)
                    metadata = {"total_num_frames": len(frames), "fps": clip.fps, "frames_indices": list(range(len(frames)))}
                    inputs = prepare_inputs(adapter.processor, videos=[frames], prompts=prompt, video_metadata=[metadata])
                    ids = inputs["input_ids"]
                    embedded = model.get_input_embeddings(
                        ids, pixel_values=inputs["pixel_values_videos"],
                        video_grid_thw=inputs["video_grid_thw"], mask=inputs.get("attention_mask"),
                    )
                    out = language(
                        ids, inputs_embeds=embedded.inputs_embeds, cache=language.make_cache(),
                        position_ids=embedded.position_ids, rope_deltas=embedded.rope_deltas,
                        skip_logits=True, return_hidden=True, capture_layer_ids=list(range(num_layers)),
                    )
                    final = out.hidden_states[-1][0, -1]
                    logits = language.model.embed_tokens.as_linear(final[None])[0] if language.args.tie_word_embeddings \
                        else language.lm_head(final[None])[0]
                    video_positions = mx.array(np.flatnonzero(np.array(ids[0]) == video_token))
                    pooled = mx.stack([
                        mx.stack([hidden[0, video_positions].mean(axis=0) for hidden in out.hidden_states]),
                        mx.stack([hidden[0, -1] for hidden in out.hidden_states]),
                    ])
                    letters = logits[letter_ids].astype(mx.float32)
                    mx.eval(pooled, letters)
                    per_clip.append(log_softmax(letters.tolist()))
                    features.append(np.array(pooled.astype(mx.float32)).astype(np.float16))
                np.save(feature_dir / f"{row['video_id']}.npy", np.stack(features))

                scores = [sum(clip[i] for clip in per_clip) / len(per_clip) for i in range(len(names))]
                prediction = names[max(range(len(names)), key=scores.__getitem__)]
                record = {
                    **row,
                    "prediction": prediction,
                    "scores": {name: round(score, 4) for name, score in zip(names, scores)},
                    "clip_predictions": [names[max(range(len(names)), key=clip.__getitem__)] for clip in per_clip],
                    "clip_starts_sec": [round(clip.start_sec, 2) for clip in clips],
                }
                if row["video_id"] not in done:
                    handle.write(json.dumps(record) + "\n")
                    handle.flush()
                done[row["video_id"]] = record
                mark = "ok " if prediction == row["domain"] else "MISS"
                print(f"{index}/{len(pending)} {mark} {row['video_id']} [{row['split']}/{row['duration']}] "
                      f"true={row['domain']} pred={prediction}", flush=True)

    report([done[row["video_id"]] for row in rows if row["video_id"] in done], names)
    print(f"saved {out_path}")
    print(f"features {feature_dir}")


if __name__ == "__main__":
    main()
