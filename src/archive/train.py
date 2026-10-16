#!/usr/bin/env python3
"""Train a domain classifier on frozen video features.

Frames come from one of two samplers. `clips` takes windows of consecutive
frames at random positions. `frames` samples single frames uniformly and on a
time interval.

Two encoders are available. `mobileclip` embeds each frame with MobileCLIP-S0.
`omni` encodes each clip's frames and audio together with the Qwen2.5-Omni
Thinker and needs the clip sampler. Features are averaged and classified with
an MLP. The label is the video domain.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import EmbeddingDataset, collate_embeddings, label_map, read_jsonl
from model import DomainGRU, DomainMLP, embed_frames, load_mobileclip
from clip_sampling import ClipSampling, sample_clips, sample_video_clips
from omni_encoder import OmniSettings
from preprocessing import FrameSampling, sample_video_frames

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_DIR = ROOT / "data" / "splits"
DEFAULT_VIDEO_DIR = ROOT / "data" / "video-mme-sample10" / "videos"


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def class_weights(rows: list[dict], labels: dict[str, int], label_field: str) -> torch.Tensor:
    counts = torch.zeros(len(labels))
    for row in rows:
        counts[labels[row[label_field]]] += 1
    weights = counts.sum() / (len(labels) * counts.clamp(min=1))
    return weights


def build_embedder(encoder_name: str, sampling, args, device: torch.device):
    """Return a function video_path, video_id -> feature tensor.

    MobileCLIP gives [frames, D] for frame sampling and [clips, clip_len, D]
    for clip sampling. Omni gives [clips, D], one vector per clip.
    """
    if encoder_name == "omni":
        from omni_encoder import OmniClipEncoder

        encoder = OmniClipEncoder(omni_settings(args), device)
        return lambda video_path, video_id: encoder.encode_clips(
            video_path, sample_clips(video_path, sampling, video_id)
        )

    encoder, preprocess = load_mobileclip(args.mobileclip, device)
    if isinstance(sampling, ClipSampling):
        return lambda video_path, video_id: torch.stack(
            [embed_frames(encoder, preprocess, clip, device) for clip in sample_video_clips(video_path, sampling, video_id)]
        )
    return lambda video_path, video_id: embed_frames(encoder, preprocess, sample_video_frames(video_path, sampling), device)


def omni_settings(args) -> OmniSettings:
    return OmniSettings(
        max_pixels=args.omni_max_tokens * 28 * 28,
        min_pixels=min(args.omni_max_tokens, 32) * 28 * 28,
        layer=None if args.omni_all_layers else args.omni_layer,
        use_audio=not args.omni_no_audio,
    )


def ensure_embeddings(rows: list[dict], video_dir: Path, embedding_dir: Path, make_embedder) -> None:
    """Embed every video missing from the cache. The encoder loads only if needed."""
    embedding_dir.mkdir(parents=True, exist_ok=True)
    pending = [row for row in rows if not (embedding_dir / f"{row['video_id']}.pt").is_file()]
    if not pending:
        return
    embed = make_embedder()
    for done, row in enumerate(pending, start=1):
        video_path = video_dir / f"{row['video_id']}.mp4"
        features = embed(video_path, row["video_id"])
        torch.save(features, embedding_dir / f"{row['video_id']}.pt")
        shape = "x".join(str(size) for size in features.shape[:-1])
        print(f"embedded {done}/{len(pending)} {row['video_id']} ({shape}, dim {features.shape[-1]})", flush=True)


def run_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    seen = 0
    for features, mask, labels in loader:
        features = features.to(device)
        mask = mask.to(device)
        labels = labels.to(device)
        if training:
            optimizer.zero_grad()
        logits = model(features, mask)
        loss = criterion(logits, labels)
        if training:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * labels.shape[0]
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += labels.shape[0]
    if seen == 0:
        return 0.0, 0.0
    return total_loss / seen, correct / seen


def evaluate(model, loader, device) -> tuple[float, dict[int, tuple[int, int]]]:
    model.eval()
    correct = 0
    seen = 0
    per_class: dict[int, list[int]] = {}
    with torch.no_grad():
        for features, mask, labels in loader:
            logits = model(features.to(device), mask.to(device))
            predictions = logits.argmax(dim=1).cpu()
            correct += (predictions == labels).sum().item()
            seen += labels.shape[0]
            for label, prediction in zip(labels.tolist(), predictions.tolist()):
                hits = per_class.setdefault(label, [0, 0])
                hits[1] += 1
                hits[0] += int(prediction == label)
    accuracy = correct / seen if seen else 0.0
    return accuracy, {label: (hits[0], hits[1]) for label, hits in per_class.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "cache" / "embeddings")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "domain_mlp.pt")
    parser.add_argument("--label-field", default="domain")
    parser.add_argument("--sampler", choices=["clips", "frames"], default="clips")
    parser.add_argument("--clip-len", type=int, default=30, help="Consecutive frames per clip.")
    parser.add_argument("--num-clips", type=int, default=1, help="Clips per video, spread across its length.")
    parser.add_argument("--sampling", choices=["uniform", "interval", "both"], default="both", help="Frame sampler mode.")
    parser.add_argument("--num-frames", type=int, default=60)
    parser.add_argument("--interval-sec", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--head", choices=["mlp", "gru"], default="mlp",
                        help="mlp averages frames first; gru reads the frame sequence in order.")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--gru-layers", type=int, default=1)
    parser.add_argument("--bidirectional", action="store_true", help="Bidirectional GRU.")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--encoder", choices=["mobileclip", "omni"], default="mobileclip")
    parser.add_argument("--mobileclip", type=Path, default=None)
    parser.add_argument("--omni-max-tokens", type=int, default=64, help="Upper bound on Omni vision tokens per frame pair.")
    parser.add_argument("--omni-layer", type=int, default=-1, help="Thinker hidden layer to pool. 0 is the embeddings, -1 the last.")
    parser.add_argument(
        "--omni-all-layers", action="store_true",
        help="Cache every hidden layer, then train on --omni-layer. Other layers reuse the cache.",
    )
    parser.add_argument("--omni-no-audio", action="store_true", help="Feed Omni the frames only.")
    parser.add_argument("--max-videos", type=int, default=0, help="Use the first N videos of each split. 0 means all.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    if args.sampler == "clips":
        sampling = ClipSampling(args.clip_len, args.num_clips, args.seed)
    else:
        sampling = FrameSampling(args.sampling, args.num_frames, args.interval_sec, args.max_frames)
    if args.encoder == "omni" and not isinstance(sampling, ClipSampling):
        raise SystemExit("--encoder omni needs --sampler clips: it encodes continuous frames with their audio")

    splits = {name: read_jsonl(args.split_dir / f"{name}.jsonl") for name in ("train", "val", "test")}
    if args.max_videos > 0:
        splits = {name: rows[: args.max_videos] for name, rows in splits.items()}
    vocabulary_rows = read_jsonl(args.split_dir / "all.jsonl") if (args.split_dir / "all.jsonl").is_file() else splits["train"]
    labels = label_map(vocabulary_rows, args.label_field)
    names = [name for name, _ in sorted(labels.items(), key=lambda item: item[1])]
    feature_tag = sampling.tag() if args.encoder == "mobileclip" else f"{omni_settings(args).tag()}_{sampling.tag()}"
    print(f"device={device} encoder={args.encoder} features={feature_tag} classes={names}")

    embedding_dir = args.cache_dir / feature_tag
    embedder = None

    def make_embedder():
        nonlocal embedder
        if embedder is None:
            embedder = build_embedder(args.encoder, sampling, args, device)
        return embedder

    for rows in splits.values():
        ensure_embeddings(rows, args.video_dir, embedding_dir, make_embedder)
    embedder = None
    if device.type == "mps":
        torch.mps.empty_cache()

    layer = args.omni_layer if args.encoder == "omni" and args.omni_all_layers else None
    train_set = EmbeddingDataset(splits["train"], embedding_dir, labels, args.label_field, layer)
    val_set = EmbeddingDataset(splits["val"], embedding_dir, labels, args.label_field, layer)
    test_set = EmbeddingDataset(splits["test"], embedding_dir, labels, args.label_field, layer)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_embeddings)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_embeddings)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_embeddings)

    embed_dim = train_set[0][0].shape[-1]
    if layer is not None:
        print(f"training on Omni hidden layer {layer}")
    if args.head == "gru":
        model = DomainGRU(embed_dim, len(labels), args.hidden_dim, args.gru_layers, args.dropout, args.bidirectional)
    else:
        model = DomainMLP(embed_dim, len(labels), args.hidden_dim, args.dropout)
    model = model.to(device)
    print(f"head={args.head} parameters={sum(p.numel() for p in model.parameters()):,}")
    weights = class_weights(splits["train"], labels, args.label_field).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_accuracy = -1.0
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, optimizer, device)
        val_accuracy, _ = evaluate(model, val_loader, device)
        print(
            f"epoch {epoch:02d}  train_loss={train_loss:.4f}  train_acc={train_accuracy:.3f}  val_acc={val_accuracy:.3f}",
            flush=True,
        )
        if val_accuracy >= best_accuracy:
            best_accuracy = val_accuracy
            torch.save(
                {
                    "model": model.state_dict(),
                    "labels": labels,
                    "label_field": args.label_field,
                    "sampling": sampling.tag(),
                    "features": feature_tag,
                    "layer": layer,
                    "head": args.head,
                    "head_config": {"hidden_dim": args.hidden_dim, "dropout": args.dropout,
                                    "gru_layers": args.gru_layers, "bidirectional": args.bidirectional},
                    "embed_dim": embed_dim,
                    "val_accuracy": val_accuracy,
                },
                args.checkpoint,
            )

    checkpoint = torch.load(args.checkpoint, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_accuracy, per_class = evaluate(model, test_loader, device)
    print(f"best_val_acc={best_accuracy:.3f}  test_acc={test_accuracy:.3f}")
    for name, index in labels.items():
        hits, total = per_class.get(index, (0, 0))
        if total:
            print(f"  {name}: {hits}/{total}")
    print(f"saved {args.checkpoint}")


if __name__ == "__main__":
    main()
