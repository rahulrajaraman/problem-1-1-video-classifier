#!/usr/bin/env python3
"""Calibrated MobileCLIP-S0 zero-shot video domain classifier.

Model package under models/mobileclip/:
  mobileclip_s0.pt   — image/text encoder weights (downloaded at setup)
  calibration.json   — per-domain score offsets (shipped; refreshed by eval)

  python src/main.py --predict data/sample/sports_clip.mp4
  python src/main.py                              # full eval (needs dataset videos)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dataset import all_rows, domain_names, load_splits
from model import DEFAULT_CALIBRATION, DEFAULT_CHECKPOINT
from pipeline import SAMPLING, MobileClipZeroShot

ROOT = Path(__file__).resolve().parents[1]


def evaluate(predictions: np.ndarray, y: np.ndarray, names: list[str]) -> dict:
    """Accuracy, per-class F1, and confusion matrix (rows = true, cols = pred)."""
    confusion = np.zeros((len(names), len(names)), dtype=int)
    for true, pred in zip(y, predictions):
        confusion[true, pred] += 1

    per_class_f1 = {}
    f1_values = []
    for index, name in enumerate(names):
        tp = int(confusion[index, index])
        support = int(confusion[index].sum())
        predicted = int(confusion[:, index].sum())
        if support == 0:
            continue
        precision = tp / predicted if predicted else 0.0
        recall = tp / support
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        per_class_f1[name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
            "correct": tp,
        }
        f1_values.append(f1)

    return {
        "accuracy": float((predictions == y).mean()) if len(y) else 0.0,
        "correct": int((predictions == y).sum()),
        "total": int(len(y)),
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "per_class": {
            name: f"{stats['correct']}/{stats['support']}"
            for name, stats in per_class_f1.items()
        },
        "per_class_f1": per_class_f1,
        "confusion": confusion.tolist(),
    }


def print_report(title: str, result: dict, names: list[str]) -> None:
    short = [name.split()[0][:9] for name in names]
    print(f"\n{title}: accuracy {result['correct']}/{result['total']} = {result['accuracy']:.3f}  "
          f"macro-F1 {result['macro_f1']:.3f}")
    print("  per-class F1")
    for name in names:
        stats = result["per_class_f1"].get(name)
        if not stats:
            continue
        print(f"    {name:<22} f1={stats['f1']:.3f}  p={stats['precision']:.3f}  "
              f"r={stats['recall']:.3f}  {stats['correct']}/{stats['support']}")
    print(f"  confusion (rows=true, cols=pred)")
    print(f"  {'':<11}" + "".join(f"{name:>10}" for name in short))
    for name, row in zip(short, result["confusion"]):
        if sum(row):
            print(f"  {name:<11}" + "".join(f"{count:>10}" for count in row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split-dir", type=Path, default=ROOT / "data" / "splits")
    parser.add_argument("--video-dir", type=Path, default=ROOT / "data" / "video-mme-sample10" / "videos")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "cache" / "embeddings" / SAMPLING.tag())
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "mobileclip_zero_shot",
                        help="Eval outputs only (summary + predictions).")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION,
                        help="Calibration offsets (default: models/mobileclip/calibration.json).")
    parser.add_argument("--calibration-splits", nargs="+", default=["train", "val"])
    parser.add_argument("--predict", type=Path, nargs="+", help="Classify these videos (calibrated).")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                        help="MobileCLIP-S0 weights (default: models/mobileclip/mobileclip_s0.pt).")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def run_predict(args: argparse.Namespace) -> None:
    if not Path(args.calibration).is_file():
        raise SystemExit(f"missing calibration file: {args.calibration}")
    classifier = MobileClipZeroShot(args.cache_dir, args.device, args.checkpoint)
    classifier.load_calibration(args.calibration)
    for path in args.predict:
        path = Path(path)
        label = classifier.predict(path, calibrated=True, cache=False)
        print(f"{path.name}: {label}")


def run_eval(args: argparse.Namespace) -> None:
    if not Path(args.video_dir).is_dir():
        raise SystemExit(
            f"missing video dir {args.video_dir}\n"
            "Download and prepare the dataset first:\n"
            "  python scripts/download_dataset.py\n"
            "  python scripts/prepare_eval.py\n"
            "  python src/main.py\n"
        )

    splits = load_splits(args.split_dir)
    rows = all_rows(splits)
    names = domain_names(rows)
    y = np.array([names.index(row["domain"]) for row in rows])

    classifier = MobileClipZeroShot(args.cache_dir, args.device, args.checkpoint)
    classifier.build_text(names)
    score_matrix = np.stack([
        classifier.scores(classifier.encode_video(args.video_dir / f"{row['video_id']}.mp4"))
        for row in rows
    ])

    calibration_mask = np.array([row["split"] in args.calibration_splits for row in rows])
    offsets = classifier.calibrate(score_matrix[calibration_mask])
    variants = {
        "uncalibrated": score_matrix.argmax(axis=1),
        "calibrated": (score_matrix - offsets).argmax(axis=1),
    }

    print(f"MobileCLIP-S0 zero-shot  frames={SAMPLING.tag()}  classes={len(names)}  videos={len(rows)}")
    print(f"checkpoint={args.checkpoint}")
    print(f"calibration={args.calibration}")
    print("calibration offsets (mean log-prob per domain over " + "+".join(args.calibration_splits) + "):")
    for name, offset in zip(names, offsets):
        print(f"  {name:<22} {offset:+.3f}")

    summary = {"sampling": SAMPLING.tag(), "names": names, "calibration_splits": args.calibration_splits, "results": {}}
    groups = {
        "all": np.ones(len(rows), dtype=bool),
        "train+val": np.array([row["split"] != "test" for row in rows]),
        "test": np.array([row["split"] == "test" for row in rows]),
    }
    for variant, predictions in variants.items():
        summary["results"][variant] = {}
        for group, mask in groups.items():
            result = evaluate(predictions[mask], y[mask], names)
            summary["results"][variant][group] = result
            print_report(f"{variant} | {group}", result, names)

    print("\nsummary")
    header = f"  {'':<14}" + "".join(f"{g:>14}" for g in groups)
    print(header)
    for variant in variants:
        cells = []
        for group in groups:
            result = summary["results"][variant][group]
            cells.append(f"{result['correct']}/{result['total']} a={result['accuracy']:.2f}")
        print(f"  {variant:<14}" + "".join(f"{c:>14}" for c in cells))
    print("  macro-F1")
    for variant in variants:
        cells = [f"{summary['results'][variant][group]['macro_f1']:.3f}" for group in groups]
        print(f"  {variant:<14}" + "".join(f"{c:>14}" for c in cells))

    classifier.save_calibration(args.calibration, args.calibration_splits)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with (args.out_dir / "predictions.jsonl").open("w") as handle:
        for index, row in enumerate(rows):
            handle.write(json.dumps({
                **row,
                "uncalibrated": names[int(variants["uncalibrated"][index])],
                "calibrated": names[int(variants["calibrated"][index])],
                "scores": dict(zip(names, np.round(score_matrix[index], 4).tolist())),
            }) + "\n")
    print(f"\nsaved calibration {args.calibration}")
    print(f"saved eval outputs {args.out_dir}")


def main() -> None:
    args = parse_args()
    if args.predict:
        run_predict(args)
    else:
        run_eval(args)


if __name__ == "__main__":
    main()
