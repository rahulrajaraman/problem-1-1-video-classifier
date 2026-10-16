#!/usr/bin/env python3
"""Use MobileCLIP's text encoder to bring class descriptions into the classifier.

Video features are the mean of cached MobileCLIP-S0 frame embeddings,
re-normalised. Every Video-MME sub-category gets a short visual description,
embedded with a few templates and averaged. A domain's text embedding is the
normalised mean of its sub-categories. Four classifiers are compared:

  zero-shot    cosine similarity to each domain's text embedding, no training
  embed-probe  baseline: standardised logistic regression on the 512-d embedding
  sim-probe    logistic regression on the 28 sub-category similarity scores
  text-init    linear head initialised from the domain text embeddings and
               penalised for moving away from them (lambda * ||W - W_text||^2)

Settings are chosen by repeated stratified cross-validation on train + val,
then the chosen setting is fit on train + val and scored once on test.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dataset import read_jsonl
from model import load_mobileclip

ROOT = Path(__file__).resolve().parents[1]

# Visual descriptions of the Video-MME sub-categories, grouped by domain.
SUBCATEGORIES = {
    "Artistic Performance": {
        "Acrobatics": "acrobats performing tricks",
        "Magic Show": "a magician performing a magic trick for an audience",
        "Stage Play": "actors performing a play on a theatre stage",
        "Variety Show": "a TV variety show with a host and performers on stage",
    },
    "Film & Television": {
        "Animation": "an animated cartoon",
        "Documentary": "a nature or history documentary",
        "Movie & TV Show": "a scene from a movie or TV series",
        "News Report": "a TV news broadcast with a news anchor",
    },
    "Knowledge": {
        "Astronomy": "an astronomy video about planets, stars and space",
        "Biology & Medicine": "an educational video about biology or medicine",
        "Finance & Commerce": "a video explaining finance, business or the economy",
        "Geography": "an educational video about geography and maps",
        "Humanity & History": "a history lecture with old pictures and maps",
        "Life Tip": "a how-to tutorial with practical tips",
        "Literature & Art": "a lecture about literature or art",
        "Technology": "a video explaining technology or gadgets",
    },
    "Life Record": {
        "Daily Life": "a vlog of someone's daily life",
        "Exercise": "a person doing a workout",
        "Fashion": "a fashion vlog showing outfits",
        "Food": "a person cooking or eating food",
        "Handicraft": "a person making handicrafts by hand",
        "Travel": "a travel vlog visiting places",
    },
    "Multilingual": {
        "Multilingual": "people speaking different languages with subtitles on screen",
    },
    "Sports Competition": {
        "Athletics": "an athletics race on a running track",
        "Basketball": "a basketball game",
        "Esports": "a competitive video game esports match",
        "Football": "a football match in a stadium",
        "Other Sports": "a sports competition",
    },
}
TEMPLATES = ["{}", "a video frame of {}.", "a screenshot of {}."]


@torch.no_grad()
def text_embeddings(names: list[str], device: torch.device) -> tuple[np.ndarray, np.ndarray, float]:
    """Sub-category [28, D] and domain [6, D] embeddings, both L2-normalised, and CLIP's logit scale."""
    import mobileclip

    model, _ = load_mobileclip(device=device)
    tokenizer = mobileclip.get_tokenizer("mobileclip_s0")
    subcategory = []
    for name in names:
        for description in SUBCATEGORIES[name].values():
            tokens = tokenizer([template.format(description) for template in TEMPLATES]).to(device)
            embedded = torch.nn.functional.normalize(model.encode_text(tokens).float(), dim=-1).mean(dim=0)
            subcategory.append(torch.nn.functional.normalize(embedded, dim=0))
    subcategory = torch.stack(subcategory).cpu()
    sizes = [len(SUBCATEGORIES[name]) for name in names]
    domain = torch.stack([chunk.mean(dim=0) for chunk in torch.split(subcategory, sizes)])
    domain = torch.nn.functional.normalize(domain, dim=-1)
    return subcategory.numpy(), domain.numpy(), float(model.logit_scale.exp())


def video_embeddings(rows: list[dict], feature_dir: Path) -> np.ndarray:
    vectors = []
    for row in rows:
        frames = torch.load(feature_dir / f"{row['video_id']}.pt", weights_only=True).float()
        mean = frames.reshape(-1, frames.shape[-1]).mean(dim=0)
        vectors.append(torch.nn.functional.normalize(mean, dim=0))
    return torch.stack(vectors).numpy()


def logistic(c: float):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=5000))


class TextInitHead:
    """Linear head W = W_text + delta on normalised embeddings, logits scaled like CLIP."""

    def __init__(self, text: np.ndarray, scale: float, penalty: float, steps: int = 300, lr: float = 1e-3):
        self.text = torch.tensor(text, dtype=torch.float32)
        self.scale, self.penalty, self.steps, self.lr = scale, penalty, steps, lr

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TextInitHead":
        features = torch.tensor(x, dtype=torch.float32)
        labels = torch.tensor(y)
        counts = torch.bincount(labels, minlength=len(self.text)).float()
        weights = counts.sum() / (len(self.text) * counts.clamp(min=1))
        self.delta = torch.zeros_like(self.text, requires_grad=True)
        self.bias = torch.zeros(len(self.text), requires_grad=True)
        optimizer = torch.optim.Adam([self.delta, self.bias], lr=self.lr)
        for _ in range(self.steps):
            optimizer.zero_grad()
            logits = self.scale * features @ (self.text + self.delta).T + self.bias
            loss = torch.nn.functional.cross_entropy(logits, labels, weight=weights)
            loss = loss + self.penalty * self.delta.pow(2).sum()
            loss.backward()
            optimizer.step()
        return self

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> np.ndarray:
        logits = self.scale * torch.tensor(x, dtype=torch.float32) @ (self.text + self.delta).T + self.bias
        return logits.argmax(dim=1).numpy()


def cv_accuracy(make_model, x: np.ndarray, y: np.ndarray, folds: int, repeats: int, seed: int) -> float:
    splitter = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    hits = 0
    for train_index, test_index in splitter.split(x, y):
        model = make_model().fit(x[train_index], y[train_index])
        hits += int((model.predict(x[test_index]) == y[test_index]).sum())
    return hits / (len(y) * repeats)


def per_class(predictions: np.ndarray, y: np.ndarray, names: list[str]) -> str:
    parts = []
    for index, name in enumerate(names):
        mask = y == index
        if mask.any():
            parts.append(f"{name.split()[0]} {int((predictions[mask] == index).sum())}/{int(mask.sum())}")
    return ", ".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("features", nargs="+", help="MobileCLIP cache folder names under --cache-dir.")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "cache" / "embeddings")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "data" / "splits")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "text_prompts")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", message="The least populated class")
    torch.manual_seed(args.seed)
    splits = {name: read_jsonl(args.split_dir / f"{name}.jsonl") for name in ("train", "val", "test")}
    names = sorted({row["domain"] for row in read_jsonl(args.split_dir / "all.jsonl")})
    label = {name: index for index, name in enumerate(names)}
    subcategory_text, domain_text, scale = text_embeddings(names, torch.device("cpu"))
    print(f"text: {len(subcategory_text)} sub-category prompts x {len(TEMPLATES)} templates, logit scale {scale:.1f}")

    fit_rows = splits["train"] + splits["val"]
    y_fit = np.array([label[row["domain"]] for row in fit_rows])
    y_test = np.array([label[row["domain"]] for row in splits["test"]])
    cv = lambda make, x: cv_accuracy(make, x, y_fit, args.folds, args.repeats, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for feature in args.features:
        feature_dir = args.cache_dir / feature
        x_fit = video_embeddings(fit_rows, feature_dir)
        x_test = video_embeddings(splits["test"], feature_dir)
        sims_fit, sims_test = x_fit @ subcategory_text.T, x_test @ subcategory_text.T
        results = []

        def record(method: str, setting: str, cv_score: float | None, predictions: np.ndarray) -> None:
            hits = int((predictions == y_test).sum())
            results.append({"method": method, "setting": setting, "cv_acc": cv_score, "test_hits": hits,
                            "test_acc": hits / len(y_test), "per_class": per_class(predictions, y_test, names)})

        zero_fit = (x_fit @ domain_text.T).argmax(axis=1)
        record("zero-shot", "-", float((zero_fit == y_fit).mean()), (x_test @ domain_text.T).argmax(axis=1))

        for method, (train_x, test_x), grid, make in [
            ("embed-probe", (x_fit, x_test), [0.01, 0.1, 1.0], lambda c: (lambda: logistic(c))),
            ("sim-probe", (sims_fit, sims_test), [0.01, 0.1, 1.0, 10.0], lambda c: (lambda: logistic(c))),
            ("text-init", (x_fit, x_test), [0.01, 0.1, 1.0, 10.0],
             lambda p: (lambda: TextInitHead(domain_text, scale, p))),
        ]:
            scored = [(cv(make(value), train_x), value) for value in grid]
            best_cv, best = max(scored, key=lambda item: item[0])
            predictions = make(best)().fit(train_x, y_fit).predict(test_x)
            key = "lambda" if method == "text-init" else "C"
            record(method, f"{key}={best}  (cv by {key}: " + ", ".join(f"{v}:{s:.2f}" for s, v in scored) + ")",
                   best_cv, predictions)

        print(f"\n== {feature}")
        print(f"  {'method':<12} {'cv (train+val)':>14} {'test':>10}   setting / per-class test")
        for result in results:
            cv_text = f"{result['cv_acc']:.3f}" if result["cv_acc"] is not None else "-"
            print(f"  {result['method']:<12} {cv_text:>14} {result['test_hits']:>4}/{len(y_test)} ={result['test_acc']:.2f}   {result['setting']}")
            print(f"  {'':<12} {'':>14} {'':>10}   {result['per_class']}")
        (args.out_dir / f"{feature}.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved {args.out_dir}")


if __name__ == "__main__":
    main()
