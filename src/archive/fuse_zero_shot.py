#!/usr/bin/env python3
"""Zero-shot domain classification fusing MobileCLIP frames with CLAP audio.

Video: the mean MobileCLIP embedding of a video's sampled frames is compared to
domain text embeddings built from the visual sub-category prompts in
text_prompts.py. Audio: each cached CLAP window is compared to the sound
prompts in clap_zero_shot.py and the window log-probabilities are averaged.

Both give log-probabilities over the same domains, which are combined as
    w * video + (1 - w) * audio.
Optionally each modality is calibrated first by subtracting its mean
log-probability per class over train+val, which removes a model's bias toward
one class without using labels. The weight and calibration choice are picked
by accuracy on train+val; test is scored once with that choice.

Usage: python src/fuse_zero_shot.py --video both_n60_i1_m100 --audio clap-larger-clap-general_a8x10s_s42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from clap_zero_shot import class_text_embeddings
from dataset import read_jsonl
from text_prompts import text_embeddings, video_embeddings

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = [round(w, 1) for w in np.arange(0.0, 1.01, 0.1)]


def log_softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=-1, keepdims=True)
    return logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))


@torch.no_grad()
def audio_log_probs(rows: list[dict], feature_dir: Path, names: list[str], model_name: str) -> np.ndarray:
    from transformers import ClapModel, ClapProcessor

    processor = ClapProcessor.from_pretrained(model_name)
    model = ClapModel.from_pretrained(model_name).eval()
    classes, scale = class_text_embeddings(model, processor, names, torch.device("cpu"))
    classes = classes.numpy()
    scores = []
    for row in rows:
        windows = torch.load(feature_dir / f"{row['video_id']}.pt", weights_only=True)[:, 0].float().numpy()
        scores.append(log_softmax(scale * windows @ classes.T).mean(axis=0))
    return np.stack(scores)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", default="both_n60_i1_m100", help="MobileCLIP frame cache name.")
    parser.add_argument("--audio", default="clap-larger-clap-general_a8x10s_s42", help="CLAP window cache name.")
    parser.add_argument("--clap-model", default="laion/larger_clap_general")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "cache" / "embeddings")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "data" / "splits")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "fusion")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = {name: read_jsonl(args.split_dir / f"{name}.jsonl") for name in ("train", "val", "test")}
    names = sorted({row["domain"] for row in read_jsonl(args.split_dir / "all.jsonl")})
    fit_rows, test_rows = splits["train"] + splits["val"], splits["test"]
    rows = fit_rows + test_rows
    y = np.array([names.index(row["domain"]) for row in rows])
    fit = np.arange(len(rows)) < len(fit_rows)

    _, domain_text, video_scale = text_embeddings(names, torch.device("cpu"))
    video = log_softmax(video_scale * video_embeddings(rows, args.cache_dir / args.video) @ domain_text.T)
    audio = audio_log_probs(rows, args.cache_dir / args.audio, names, args.clap_model)

    variants = {}
    for calibrated in (False, True):
        v = video - video[fit].mean(axis=0) if calibrated else video
        a = audio - audio[fit].mean(axis=0) if calibrated else audio
        for weight in WEIGHTS:
            predictions = (weight * v + (1 - weight) * a).argmax(axis=1)
            variants[(calibrated, weight)] = predictions

    def accuracy(predictions: np.ndarray, mask: np.ndarray) -> float:
        return float((predictions[mask] == y[mask]).mean())

    print(f"video={args.video}  audio={args.audio}  classes={len(names)}  train+val={fit.sum()}  test={(~fit).sum()}")
    print(f"\n  {'calibrated':<11} {'w_video':>7} {'train+val':>10} {'test':>8}")
    for (calibrated, weight), predictions in variants.items():
        label = "video only" if weight == 1.0 else "audio only" if weight == 0.0 else ""
        print(f"  {str(calibrated):<11} {weight:>7} {accuracy(predictions, fit):>10.3f} "
              f"{int((predictions[~fit] == y[~fit]).sum()):>4}/{(~fit).sum()}   {label}")

    best = max(variants, key=lambda key: (accuracy(variants[key], fit), -abs(key[1] - 0.5)))
    predictions = variants[best]
    hits = int((predictions[~fit] == y[~fit]).sum())
    print(f"\nchosen on train+val: calibrated={best[0]} w_video={best[1]}  "
          f"train+val={accuracy(predictions, fit):.3f}  test={hits}/{(~fit).sum()} = {hits / (~fit).sum():.3f}")
    short = [name.split()[0] for name in names]
    print("test confusion (rows=true, cols=pred)", short)
    for index, name in enumerate(short):
        mask = ~fit & (y == index)
        print(f"  {name:<10}", np.bincount(predictions[mask], minlength=len(names)).tolist())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.video}__{args.audio}.jsonl"
    with out_path.open("w") as handle:
        for index, row in enumerate(rows):
            handle.write(json.dumps({
                **row, "split": "test" if not fit[index] else "train+val",
                "prediction": names[int(predictions[index])],
                "video_scores": dict(zip(names, np.round(video[index], 4).tolist())),
                "audio_scores": dict(zip(names, np.round(audio[index], 4).tolist())),
                "calibrated": best[0], "w_video": best[1],
            }) + "\n")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
