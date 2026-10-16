#!/usr/bin/env python3
"""Zero-shot video domain classification with MobileCLIP-S0, uncalibrated and calibrated.

Pipeline for each video:
  1. Sample frames: 60 spread uniformly plus one per second, capped at 100.
  2. Embed every frame with the MobileCLIP-S0 image encoder and average them
     into one L2-normalised video embedding.
  3. Compare it with one text embedding per domain, built from the visual
     sub-category prompts in text_prompts.py, and turn the scaled cosine
     similarities into log-probabilities.
  4. Uncalibrated: predict the highest-scoring domain.
     Calibrated: first subtract each domain's mean log-probability over the
     calibration videos (train+val), so a domain that scores high on every
     video stops absorbing the others. Calibration uses no labels.

Evaluation mode (default) scores every split and saves predictions, a summary
and the calibration offsets. Prediction mode classifies new files with those
offsets:
    python src/mobileclip_zero_shot.py
    python src/mobileclip_zero_shot.py --predict path/to/video.mp4 ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dataset import read_jsonl
from model import embed_frames, load_mobileclip
from preprocessing import FrameSampling, sample_video_frames
from text_prompts import text_embeddings

ROOT = Path(__file__).resolve().parents[1]
SAMPLING = FrameSampling(mode="both", num_frames=60, interval_sec=1, max_frames=100)


def log_softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=-1, keepdims=True)
    return logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))


def video_embedding(frame_features: torch.Tensor) -> np.ndarray:
    mean = frame_features.float().reshape(-1, frame_features.shape[-1]).mean(dim=0)
    return torch.nn.functional.normalize(mean, dim=0).numpy()


class FrameEncoder:
    """Loads MobileCLIP only when a video is missing from the cache."""

    def __init__(self, cache_dir: Path, device: torch.device):
        self.cache_dir, self.device, self.model = cache_dir, device, None

    def __call__(self, video_path: Path, cache: bool = True) -> torch.Tensor:
        cached = self.cache_dir / f"{video_path.stem}.pt"
        if cache and cached.is_file():
            return torch.load(cached, weights_only=True)
        if self.model is None:
            self.model, self.preprocess = load_mobileclip(device=self.device)
        features = embed_frames(self.model, self.preprocess, sample_video_frames(video_path, SAMPLING), self.device)
        if cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            torch.save(features, cached)
        return features


def evaluate(predictions: np.ndarray, y: np.ndarray, names: list[str]) -> dict:
    confusion = np.zeros((len(names), len(names)), dtype=int)
    for true, pred in zip(y, predictions):
        confusion[true, pred] += 1
    return {
        "accuracy": float((predictions == y).mean()),
        "correct": int((predictions == y).sum()),
        "total": int(len(y)),
        "per_class": {name: f"{confusion[i, i]}/{confusion[i].sum()}" for i, name in enumerate(names) if confusion[i].sum()},
        "confusion": confusion.tolist(),
    }


def print_report(title: str, result: dict, names: list[str]) -> None:
    short = [name.split()[0][:9] for name in names]
    print(f"\n{title}: {result['correct']}/{result['total']} = {result['accuracy']:.3f}")
    print(f"  {'true / pred':<11}" + "".join(f"{name:>10}" for name in short))
    for name, row in zip(short, result["confusion"]):
        if sum(row):
            print(f"  {name:<11}" + "".join(f"{count:>10}" for count in row) + f"   {row[short.index(name)]}/{sum(row)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split-dir", type=Path, default=ROOT / "data" / "splits")
    parser.add_argument("--video-dir", type=Path, default=ROOT / "data" / "video-mme-sample10" / "videos")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "cache" / "embeddings" / SAMPLING.tag())
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "mobileclip_zero_shot")
    parser.add_argument("--calibration-splits", nargs="+", default=["train", "val"])
    parser.add_argument("--predict", type=Path, nargs="+", help="Classify these video files with saved calibration.")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def predict(args: argparse.Namespace) -> None:
    calibration = json.loads((args.out_dir / "calibration.json").read_text())
    names, offsets = calibration["names"], np.array(calibration["offsets"])
    _, domain_text, scale = text_embeddings(names, torch.device("cpu"))
    encoder = FrameEncoder(args.cache_dir, torch.device(args.device))
    for path in args.predict:
        scores = log_softmax(scale * video_embedding(encoder(path, cache=False)) @ domain_text.T)
        raw, calibrated = names[int(scores.argmax())], names[int((scores - offsets).argmax())]
        print(f"{path.name}: uncalibrated={raw}  calibrated={calibrated}")


def main() -> None:
    args = parse_args()
    if args.predict:
        predict(args)
        return

    splits = {name: read_jsonl(args.split_dir / f"{name}.jsonl") for name in ("train", "val", "test")}
    names = sorted({row["domain"] for rows in splits.values() for row in rows})
    rows = [{**row, "split": split} for split, split_rows in splits.items() for row in split_rows]
    y = np.array([names.index(row["domain"]) for row in rows])

    encoder = FrameEncoder(args.cache_dir, torch.device(args.device))
    videos = np.stack([video_embedding(encoder(args.video_dir / f"{row['video_id']}.mp4")) for row in rows])
    _, domain_text, scale = text_embeddings(names, torch.device("cpu"))
    scores = log_softmax(scale * videos @ domain_text.T)

    calibration_mask = np.array([row["split"] in args.calibration_splits for row in rows])
    offsets = scores[calibration_mask].mean(axis=0)
    variants = {"uncalibrated": scores.argmax(axis=1), "calibrated": (scores - offsets).argmax(axis=1)}

    print(f"MobileCLIP-S0 zero-shot  frames={SAMPLING.tag()}  classes={len(names)}  videos={len(rows)}")
    print("calibration offsets (mean log-prob per domain over " + "+".join(args.calibration_splits) + "):")
    for name, offset in zip(names, offsets):
        print(f"  {name:<22} {offset:+.3f}")

    summary = {"sampling": SAMPLING.tag(), "names": names, "calibration_splits": args.calibration_splits, "results": {}}
    groups = {"train+val": np.array([row["split"] != "test" for row in rows]), "test": np.array([row["split"] == "test" for row in rows])}
    for variant, predictions in variants.items():
        summary["results"][variant] = {}
        for group, mask in groups.items():
            result = evaluate(predictions[mask], y[mask], names)
            summary["results"][variant][group] = result
            print_report(f"{variant} | {group}", result, names)

    print("\nsummary")
    print(f"  {'':<14}{'train+val':>12}{'test':>12}")
    for variant in variants:
        cells = [summary["results"][variant][group] for group in groups]
        print(f"  {variant:<14}" + "".join(f"{c['correct']:>5}/{c['total']:<3} {c['accuracy']:.2f}".rjust(12) for c in cells))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "calibration.json").write_text(json.dumps({"names": names, "offsets": offsets.tolist(),
                                                               "calibration_splits": args.calibration_splits}, indent=2))
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with (args.out_dir / "predictions.jsonl").open("w") as handle:
        for index, row in enumerate(rows):
            handle.write(json.dumps({
                **row,
                "uncalibrated": names[int(variants["uncalibrated"][index])],
                "calibrated": names[int(variants["calibrated"][index])],
                "scores": dict(zip(names, np.round(scores[index], 4).tolist())),
            }) + "\n")
    print(f"\nsaved {args.out_dir}")


if __name__ == "__main__":
    main()
