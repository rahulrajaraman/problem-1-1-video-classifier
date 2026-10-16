# Source code

**Main pipeline:** calibrated MobileCLIP-S0 zero-shot domain classification.

```bash
python scripts/download_mobileclip.py
python src/main.py --predict data/sample/sports_clip.mp4   # default demo

# Full metrics (needs HF videos ~11 GB):
python scripts/download_dataset.py
python scripts/prepare_eval.py
python src/main.py   # accuracy, macro-F1, per-class F1, confusion on all/train+val/test
```

| File | Role |
| --- | --- |
| `main.py` | CLI entry point |
| `pipeline.py` | Encoder, text prompts scoring, calibration |
| `dataset.py` | Load train/val/test jsonl splits |
| `model.py` | MobileCLIP-S0 encoder + default paths under `models/mobileclip/` |
| `preprocessing.py` | Frame sampling (60 uniform + 1/sec, max 100) |
| `prompts.py` | Domain / sub-category text descriptions |
| `archive/` | Older experiments (MLP, GRU, Omni, jev, CLAP, …) |

Write-up: [`docs/README.md`](../docs/README.md). Full history: [`docs/experiments.md`](../docs/experiments.md).
