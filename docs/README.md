# Video domain classification on Video-MME (10% sample)

## Problem

Classify each video into a Video-MME **domain** using a 10% stratified sample.
After dropping Multilingual (only 2 examples), the data is **88 videos / 5 domains**
(train 43 / val 18 / test 27). Splits: `data/splits/`.

## Method

**Calibrated MobileCLIP-S0 zero-shot** (`src/main.py`):

1. Sample ~100 frames (60 uniform + one per second, capped at 100).
2. Embed each frame with MobileCLIP-S0; mean-pool to one video vector.
3. Score domains by cosine similarity to text prompts built from visual
   sub-category descriptions (`src/prompts.py`).
4. **Calibrate** by subtracting each domain’s mean log-probability on train+val
   (no labels). This removes Film & Television’s score bias.

| Variant | All (88) | Train+val | Test |
| --- | --- | --- | --- |
| Uncalibrated | 63/88 (72%), macro-F1 0.72 | 45/61 (74%) | 18/27 (67%) |
| **Calibrated** | **72/88 (82%), macro-F1 0.80** | **51/61 (84%)** | **21/27 (78%)** |

## Quick start (sample prediction)

No Video-MME download required:

```bash
pip install -r requirements.txt
python scripts/download_mobileclip.py
python src/main.py --predict data/sample/sports_clip.mp4
# → sports_clip.mp4: Sports Competition
```

Model package: `models/mobileclip/calibration.json` (shipped) + `mobileclip_s0.pt` (downloaded, ~206 MB).

## Full evaluation (metrics on all 88 videos)

Needs the Hugging Face dataset (~11 GB of videos):

```bash
# 1. Download Video-MME 10% sample
python scripts/download_dataset.py
#    → data/video-mme-sample10/  (from aintropy-ai/video-mme-sample10)

# 2. Verify videos match the shipped 5-class splits (Multilingual already dropped)
python scripts/prepare_eval.py

# 3. Run zero-shot eval (uncalibrated + calibrated)
python src/main.py
```

Eval reports for **all / train+val / test**:

- overall **accuracy**
- **macro-F1** and **per-class precision / recall / F1**
- **confusion matrix** (rows = true domain, cols = prediction)

Outputs: `results/mobileclip_zero_shot/summary.json` and `predictions.jsonl`.
Also refreshes `models/mobileclip/calibration.json`.

To rebuild splits from scratch instead of using the shipped ones:

```bash
python scripts/prepare_eval.py --rebuild-splits
```

## Earlier experiments

MLP/GRU heads, Qwen Omni, jev-visual, CLAP audio, and text-init probes were tried
and archived under `src/archive/`. Calibrated MobileCLIP zero-shot was the best
trade-off of accuracy and cost. Full tables: [`experiments.md`](experiments.md).

## Layout

- `src/` — main pipeline
- `src/archive/` — previous experiment scripts (kept, not deleted)
- `data/splits/` — 5-class splits; `data/sample/` — smoke-test clip
- `models/mobileclip/` — weights (download) + `calibration.json`
- `results/mobileclip_zero_shot/` — eval summary + predictions
- `experiments/` — outputs from earlier methods
