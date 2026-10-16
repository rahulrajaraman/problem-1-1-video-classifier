#!/usr/bin/env python3
"""Train a linear probe on cached multi-layer features and pick the layer honestly.

Works on caches with one file per video:
    jev-visual .npy   [clips, pools, layers, D]   pools: 0 image mean, 1 answer position
    Omni all-layer .pt [clips, layers, D]

Each video becomes the mean of its clip vectors at one (pool, layer). The probe
is standardisation plus class-balanced logistic regression. Every combination
of pool, layer and regularisation strength C is scored with repeated
stratified cross-validation on train + val only. The best combination by that
score is then fit on train + val and evaluated once on test, so the test split
never influences the choice.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
POOL_NAMES = {0: "image-mean", 1: "answer-pos"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load_features(path_stem: Path) -> np.ndarray:
    """[clips, pools, layers, D] as float32. Omni caches get a single pool axis."""
    if path_stem.with_suffix(".npy").is_file():
        features = np.load(path_stem.with_suffix(".npy")).astype(np.float32)
    else:
        import torch

        features = torch.load(path_stem.with_suffix(".pt"), weights_only=True).float().numpy()
    if features.ndim == 3:
        features = features[:, None]
    if features.ndim != 4:
        raise SystemExit(f"{path_stem.name}: expected clips x [pools x] layers x D, got shape {features.shape}")
    return features


def make_probe(c: float):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=5000))


def cv_accuracy(x: np.ndarray, y: np.ndarray, c: float, folds: int, repeats: int, seed: int) -> float:
    splitter = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    hits = 0
    for train_index, test_index in splitter.split(x, y):
        model = make_probe(c).fit(x[train_index], y[train_index])
        hits += int((model.predict(x[test_index]) == y[test_index]).sum())
    return hits / (len(y) * repeats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("features", help="Cache folder name under --cache-dir.")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "cache" / "embeddings")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "data" / "splits")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "probe")
    parser.add_argument("--layers", type=int, nargs="*", default=None, help="Layer indices to try. Default: all.")
    parser.add_argument("--C", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", message="The least populated class")
    feature_dir = args.cache_dir / args.features
    splits = {name: read_jsonl(args.split_dir / f"{name}.jsonl") for name in ("train", "val", "test")}
    names = sorted({row["domain"] for rows in splits.values() for row in rows})
    label = {name: index for index, name in enumerate(names)}

    def stack(rows):
        features = np.stack([load_features(feature_dir / row["video_id"]).mean(axis=0) for row in rows])
        return features, np.array([label[row["domain"]] for row in rows])

    fit_rows = splits["train"] + splits["val"]
    x_fit, y_fit = stack(fit_rows)
    x_train, y_train = stack(splits["train"])
    x_val, y_val = stack(splits["val"])
    x_test, y_test = stack(splits["test"])
    num_pools, num_layers = x_fit.shape[1], x_fit.shape[2]
    layers = args.layers if args.layers else list(range(num_layers))
    print(f"features={args.features} videos fit={len(fit_rows)} test={len(y_test)} pools={num_pools} layers={num_layers} dim={x_fit.shape[-1]}")
    print(f"cv: {args.repeats}x repeated {args.folds}-fold on train+val; val column fits on train only")

    results = []
    for pool in range(num_pools):
        pool_name = POOL_NAMES.get(pool, str(pool)) if num_pools > 1 else "clip-mean"
        for layer in layers:
            for c in args.C:
                cv = cv_accuracy(x_fit[:, pool, layer], y_fit, c, args.folds, args.repeats, args.seed)
                val = float((make_probe(c).fit(x_train[:, pool, layer], y_train).predict(x_val[:, pool, layer]) == y_val).mean())
                results.append({"pool": pool_name, "pool_index": pool, "layer": layer, "C": c, "cv_acc": cv, "val_acc": val})
                print(f"  {pool_name:<10} layer {layer:>2}  C={c:<5}  cv={cv:.3f}  val={val:.3f}", flush=True)

    best = max(results, key=lambda r: (r["cv_acc"], r["val_acc"]))
    pool, layer, c = best["pool_index"], best["layer"], best["C"]
    model = make_probe(c).fit(x_fit[:, pool, layer], y_fit)
    predictions = model.predict(x_test[:, pool, layer])
    test_acc = float((predictions == y_test).mean())
    print(f"\nbest by cv: {best['pool']} layer {layer} C={c}  cv={best['cv_acc']:.3f}  val={best['val_acc']:.3f}")
    print(f"test (fit on train+val): {int((predictions == y_test).sum())}/{len(y_test)} = {test_acc:.3f}")
    for index, name in enumerate(names):
        mask = y_test == index
        if mask.any():
            print(f"  {name}: {int((predictions[mask] == index).sum())}/{int(mask.sum())}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.features}.json"
    out_path.write_text(json.dumps({
        "features": args.features,
        "best": {**best, "test_acc": test_acc},
        "test_predictions": {row["video_id"]: names[p] for row, p in zip(splits["test"], predictions)},
        "sweep": results,
    }, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
