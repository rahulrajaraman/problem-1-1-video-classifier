# Problem 1.1: Video Classifier

**Write-up:** [`docs/README.md`](docs/README.md) · **Full experiment log:** [`docs/experiments.md`](docs/experiments.md) · **Archived outputs:** [`experiments/`](experiments/) · **Code notes:** [`src/README.md`](src/README.md)

Main pipeline: **calibrated MobileCLIP-S0 zero-shot**.

### Benchmark metrics

Full accuracy, macro-F1, per-class F1, and confusion matrices for the Video-MME 10% sample (88 videos, 5 domains) are in:

**[`results/mobileclip_zero_shot/summary.json`](results/mobileclip_zero_shot/summary.json)**

| Variant | All (88) | Train+val (61) | Test (27) |
| --- | --- | --- | --- |
| Uncalibrated | 63/88 (72%), F1 0.72 | 45/61 (74%) | 18/27 (67%) |
| **Calibrated** | **72/88 (82%), F1 0.80** | **51/61 (84%)** | **21/27 (78%)** |

### Quick start

```bash
python scripts/download_mobileclip.py
python src/main.py --predict data/sample/sports_clip.mp4
```

To regenerate metrics after downloading the HF dataset, see [`docs/README.md`](docs/README.md).

## Submission instructions

1. Install the **AIntropy Grader Bot** GitHub App (https://github.com/apps/aintropy-grader-bot) and grant it access to your fork, if you haven't already — we can't evaluate your submission without this.
2. Put your code under `src/`.
3. Put your write-up under `docs/` — markdown, slides, a notebook, or another format of your choosing are all fine, as long as it's committed in this repo.
4. Fill in `run.yaml` with the **exact** commands needed to set up and run your code. We run these commands exactly as written to evaluate your submission — this is the one part of the submission we read literally, not loosely: if `run.yaml` doesn't work, we can't see your code run.
5. Push to your own fork's default branch any time during your test's time limit. No pull request is needed or expected.

Once you start the test you have until your test's time limit to complete the assignment — see your invitation email for the exact duration. We ignore any commits pushed to your repo after that.

Questions? careers@aintropy.ai
