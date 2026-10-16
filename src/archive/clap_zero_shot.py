#!/usr/bin/env python3
"""Classify video domains zero-shot from audio alone with CLAP.

Several audio windows are cut from each video, one at a random point inside
each equal segment of the video, seeded on the video id. CLAP embeds every
window and a set of text descriptions of what each domain sounds like. A
domain's text embedding is the normalised mean of its descriptions. Each
window's cosine similarities, scaled by CLAP's learned temperature, become
log-probabilities over domains, which are averaged over the video's windows.

Audio embeddings are cached as [windows, 1, D] so probe.py can train a linear
probe on them as well.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch

from jev_zero_shot import read_jsonl, report

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")
SAMPLE_RATE = 48000

# Written to describe sound, since CLAP never sees the picture.
DOMAIN_PROMPTS = {
    "Artistic Performance": [
        "a live stage performance with music and audience applause",
        "a magician talking to an audience that laughs and claps",
        "actors performing a stage play in a theatre",
        "a variety show with music, a host and a laughing audience",
    ],
    "Film & Television": [
        "a movie soundtrack with dramatic music and dialogue",
        "a news anchor reporting the news",
        "a documentary narrator speaking over background music",
        "cartoon voices and animated sound effects",
    ],
    "Knowledge": [
        "a person explaining a topic in a lecture",
        "an educational narrator explaining science",
        "a calm voice giving a tutorial",
        "a presenter talking about history or technology",
    ],
    "Life Record": [
        "a vlogger talking casually to the camera",
        "everyday sounds at home and cooking in a kitchen",
        "street ambience while travelling",
        "a casual conversation among friends outdoors",
    ],
    "Multilingual": [
        "people speaking in several different languages",
        "a person speaking a foreign language with a translation",
        "a language lesson comparing words in different languages",
        "a speaker switching between two languages",
    ],
    "Sports Competition": [
        "a sports commentator with a cheering crowd",
        "a stadium crowd cheering at a football match",
        "a referee whistle and players shouting",
        "basketball sneakers squeaking and a crowd cheering",
    ],
}


def embedding(output) -> torch.Tensor:
    """transformers 5 returns a model output from get_*_features; older versions return the tensor."""
    tensor = output.pooler_output if hasattr(output, "pooler_output") else output
    return torch.nn.functional.normalize(tensor, dim=-1)


def class_text_embeddings(model, processor, names: list[str], device: torch.device) -> tuple[torch.Tensor, float]:
    """Normalised mean prompt embedding per domain [classes, D] and CLAP's audio logit scale."""
    prompts = [prompt for name in names for prompt in DOMAIN_PROMPTS[name]]
    text_inputs = processor(text=prompts, padding=True, return_tensors="pt").to(device)
    text = embedding(model.get_text_features(**text_inputs))
    per_class = len(DOMAIN_PROMPTS[names[0]])
    classes = torch.nn.functional.normalize(text.view(len(names), per_class, -1).mean(dim=1), dim=-1)
    return classes, float(model.logit_scale_a.exp())


def video_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def window_starts(duration: float, window_sec: float, num_windows: int, seed: int, video_id: str) -> list[float]:
    """One random start inside each of `num_windows` equal segments, in order."""
    rng = random.Random(f"{seed}:{video_id}")
    latest = max(0.0, duration - window_sec)
    segment = latest / num_windows
    return [rng.uniform(index * segment, (index + 1) * segment) for index in range(num_windows)]


def read_audio(path: Path, start: float, window_sec: float) -> tuple[np.ndarray, bool]:
    """Mono 48 kHz window, zero-padded. The flag is False when the video has no audio there."""
    command = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-ss", f"{start:.3f}", "-t", f"{window_sec:.3f}",
        "-i", str(path), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-",
    ]
    samples = np.frombuffer(subprocess.run(command, capture_output=True, check=False).stdout, dtype=np.float32)
    expected = int(round(window_sec * SAMPLE_RATE))
    has_audio = samples.size > 0 and float(np.abs(samples).max()) > 1e-4
    return np.pad(samples, (0, max(0, expected - samples.size)))[:expected].copy(), has_audio


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split-dir", type=Path, default=ROOT / "data" / "splits")
    parser.add_argument("--video-dir", type=Path, default=ROOT / "data" / "video-mme-sample10" / "videos")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "cache" / "embeddings")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "zero_shot")
    parser.add_argument("--model", default="laion/larger_clap_general")
    parser.add_argument("--num-windows", type=int, default=8)
    parser.add_argument("--window-sec", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-videos", type=int, default=0, help="Use the first N videos of each split. 0 means all.")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    from transformers import ClapModel, ClapProcessor

    device = pick_device(args.device)
    present = {row["domain"] for row in read_jsonl(args.split_dir / "all.jsonl")}
    names = sorted(name for name in DOMAIN_PROMPTS if name in present)
    rows = []
    for split in SPLITS:
        split_rows = read_jsonl(args.split_dir / f"{split}.jsonl")
        if args.max_videos > 0:
            split_rows = split_rows[: args.max_videos]
        rows.extend({**row, "split": split} for row in split_rows)

    model_name = args.model.split("/")[-1].replace("_", "-")
    tag = f"clap-{model_name}_a{args.num_windows}x{args.window_sec:g}s_s{args.seed}"
    feature_dir = args.cache_dir / tag
    feature_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{tag}_{len(names)}class.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {record["video_id"]: record for record in read_jsonl(out_path)} if out_path.is_file() else {}
    pending = [row for row in rows if row["video_id"] not in done or not (feature_dir / f"{row['video_id']}.pt").is_file()]
    print(f"run={tag} device={device} videos={len(rows)} pending={len(pending)}")

    if pending:
        processor = ClapProcessor.from_pretrained(args.model)
        model = ClapModel.from_pretrained(args.model).to(device).eval()
        classes, scale = class_text_embeddings(model, processor, names, device)

        with out_path.open("a") as handle:
            for index, row in enumerate(pending, start=1):
                video_path = args.video_dir / f"{row['video_id']}.mp4"
                starts = window_starts(video_duration(video_path), args.window_sec, args.num_windows, args.seed, row["video_id"])
                windows = [read_audio(video_path, start, args.window_sec) for start in starts]
                cached = feature_dir / f"{row['video_id']}.pt"
                if cached.is_file():
                    audio = torch.load(cached, weights_only=True)[:, 0].to(device)
                else:
                    audio_inputs = processor(audio=[samples for samples, _ in windows], sampling_rate=SAMPLE_RATE, return_tensors="pt").to(device)
                    audio = embedding(model.get_audio_features(**audio_inputs))
                    torch.save(audio.cpu()[:, None], cached)

                log_probs = torch.log_softmax(scale * audio @ classes.T, dim=-1).cpu()
                scores = log_probs.mean(dim=0)
                prediction = names[int(scores.argmax())]
                record = {
                    **row,
                    "prediction": prediction,
                    "scores": {name: round(float(score), 4) for name, score in zip(names, scores)},
                    "window_predictions": [names[int(i)] for i in log_probs.argmax(dim=1)],
                    "window_starts_sec": [round(start, 2) for start in starts],
                    "windows_with_audio": sum(flag for _, flag in windows),
                }
                if row["video_id"] not in done:
                    handle.write(json.dumps(record) + "\n")
                    handle.flush()
                done[row["video_id"]] = record
                mark = "ok " if prediction == row["domain"] else "MISS"
                print(f"{index}/{len(pending)} {mark} {row['video_id']} [{row['split']}/{row['duration']}] "
                      f"audio={record['windows_with_audio']}/{len(windows)} true={row['domain']} pred={prediction}", flush=True)

    report([done[row["video_id"]] for row in rows if row["video_id"] in done], names)
    print(f"saved {out_path}")
    print(f"features {feature_dir}")


if __name__ == "__main__":
    main()
