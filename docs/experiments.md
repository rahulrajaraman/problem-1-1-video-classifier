# Experiment log: Video-MME domain classifier

Hardware unless noted: Apple M3 MacBook Air, 16 GB. Main env: Python 3.13, PyTorch (MPS), transformers, scikit-learn. jev-visual runs in `jev-visual/.venv` (MLX / mlx-vlm).

**Constraint:** files were not deleted when repurposed; old 6-class results and caches remain under `results/` and `data/cache/`.

---

## 1. Dataset and splits

| Item | Detail |
| --- | --- |
| Source | Video-MME 10% sample (`data/video-mme-sample10/`) |
| Original | 90 videos, 6 domains, 28 sub-categories |
| Split | Stratified by duration × domain, seed 42 → 45 / 18 / 27 |
| Change | Dropped **Multilingual** (2 videos, both in train) via `data/drop_domain.py` |
| Current | **88 videos**, 5 domains: train 43 / val 18 / test 27 |
| Backup | `data/splits_6class/` (untouched original splits) |

### Domain counts (5-class)

| Split | Artistic | Film & TV | Knowledge | Life Record | Sports |
| --- | --- | --- | --- | --- | --- |
| Train | 4 | 6 | 12 | 11 | 10 |
| Val | 2 | 2 | 5 | 5 | 4 |
| Test | 5 | 3 | 6 | 7 | 6 |
| All | 11 | 11 | 23 | 23 | 20 |

Test never contained Multilingual. With n_test = 27, **each video ≈ 3.7 accuracy points**; gaps of 1–2 videos are within noise. Prefer repeated stratified CV on train+val (61 videos) when available.

---

## 2. Feature caches

Stored under `data/cache/embeddings/<tag>/` as one `.pt` per `video_id`.

| Tag | Encoder | Sampling |
| --- | --- | --- |
| `both_n60_i1_m100` | MobileCLIP-S0 | 60 uniform + 1/sec, max 100 frames → `[T, 512]` |
| `clips_l30_c1_s42` | MobileCLIP-S0 | 1 clip × 30 contiguous frames |
| `clips_l30_c8_s42` | MobileCLIP-S0 | 8 clips × 30 frames |
| `clips_l60_c1_s42` | MobileCLIP-S0 | 1 clip × 60 frames |
| `qwen25-omni-3b_av_p64_l-1_*` | Qwen2.5-Omni-3B Thinker | Clip A/V; last-layer mean over media tokens |
| `qwen25-omni-3b_av_p64_lall_clips_l60_c8_s42` | Omni all layers | Partial (2 videos); long run aborted |
| `jev-qwen35-08b-4bit_g2_clips_l60_c8_s42_prompt-v2` | jev 0.8B-4bit | 8 clips → 2×2 grids; pools × layers `[clips, 2, 25, 1024]` |
| `jev-qwen35-08b-4bit_video57k_clips_l90_c8_s42_prompt-v2` | jev video path | 8 × 90 frames @ 320×180 |
| `clap-larger-clap-general_a8x10s_s42` | LAION CLAP | 8 × 10 s audio windows → `[8, 1, 512]` |
| `clap-larger-clap-general_a24x10s_s42` | LAION CLAP | 24 × 10 s windows |

---

## 3. Zero-shot methods

### 3.1 MobileCLIP-S0 (primary pipeline)

**Script:** `src/mobileclip_zero_shot.py`  
**Outputs:** `results/mobileclip_zero_shot/`

1. Sample frames (`both_n60_i1_m100`).
2. Embed with MobileCLIP-S0; L2-normalise and mean-pool.
3. Domain text = mean of sub-category prompt embeddings (`src/text_prompts.py`), with template ensembling.
4. Logits = `logit_scale * video · textᵀ`.
5. **Uncalibrated:** argmax.  
   **Calibrated:** subtract mean log-prob per domain on train+val, then argmax (label-free).

| Variant | Train+val | Test |
| --- | --- | --- |
| Uncalibrated | 45/61 = 0.738 | 18/27 = 0.667 |
| **Calibrated** | **51/61 = 0.836** | **21/27 = 0.778** |

**Test per-class (calibrated):** Artistic 4/5, Film 3/3, Knowledge 5/6, Life 4/7, Sports 5/6.

**Why calibration helps:** Film & Television has a much higher average score than other domains; subtracting that bias recovers Knowledge (and others) that were absorbed into Film.

Predict new videos: `python src/mobileclip_zero_shot.py --predict video.mp4` (~20 s/video on CPU including model load).

### 3.2 Qwen2.5-Omni-3B Thinker

**Script:** `src/zero_shot.py`  
**Result:** `experiments/zero_shot/qwen25-omni-3b_av_p64_clips_l30_c1_s42.jsonl` (6-class)

First-token scoring of domain names given video+audio clip.

| Split | Hits |
| --- | --- |
| Val | 12/18 |
| Test | **12/27** |

Cannot reuse frozen embedding caches (needs LM head / next-token logits).

### 3.3 jev-visual (Qwen3.5-0.8B-4bit, MLX)

**Scripts:** `src/jev_zero_shot.py`, `src/jev_video.py`, `src/jev_frames.py`  
Image/video + text only (no audio). Letter (A–E/F) scoring from logits. Prompt set `v2` retuned Sports / Life Record wording.

| Setup | Classes | Test (raw) | Notes |
| --- | --- | --- | --- |
| 8 grids, 60-frame clips, prompt v1 | 6 | 13/27 | |
| Same, prompt v2 | 6 | **15/27** | |
| Same, calibrated (“corrected”) | 6 | 19–20/27 | Mean-subtract scores on train |
| 8 × 90-frame video path, v2 | 6 | 12/27 | Sports → Multilingual often |
| Interval frames as grids | — | not run | Aborted |

### 3.4 CLAP (audio only)

**Script:** `src/clap_zero_shot.py` (restored; transformers-5 uses `.pooler_output`)  
**Model:** `laion/larger_clap_general`, 48 kHz, prompt ensemble of sound descriptions.

| Setup | Classes | Train+val | Test |
| --- | --- | --- | --- |
| 8×10s, uncalibrated | 6 | — | 9/27 |
| 8×10s, calibrated | 5 | 0.44 | 16/27 |
| 24×10s, calibrated | 5 | 0.49 | 14/27 |

Life Record is especially weak (often 0 correct): speech ambience looks like Film / Knowledge.

### 3.5 MobileCLIP + CLAP fusion

**Script:** `src/fuse_zero_shot.py`  
Combine log-probs: `w * video + (1−w) * audio`; optionally calibrate each modality; pick `w` and calibration on train+val.

| Setup | Chosen | Train+val | Test |
| --- | --- | --- | --- |
| Video + CLAP 8 windows | cal=True, w_video=0.9 | 0.836 | 21/27 |
| Video + CLAP 24 windows | cal=True, w_video=0.9 | 0.836 | 21/27 |
| Video only, calibrated | — | 0.836 | 21/27 |

**Conclusion:** fusion matches calibrated video alone; audio weight >0.1 hurts.

### 3.6 Quantized Omni GGUF (llama.cpp)

**Script:** `src/omni_gguf_zero_shot.py`  
Q4_K_M + mmproj; 8 jev grids per video ≈ 20 s/video. **Stopped by user** mid-run; partial `experiments/zero_shot/omni3b-gguf-q4km_*.jsonl`.

---

## 4. Trained heads on frozen features

### 4.1 MobileCLIP MLP / GRU (`src/train.py`)

Fit on train, early-stop / pick by val; report test. **6-class** (pre-drop). Heads: `DomainMLP`, `DomainGRU` (`src/model.py`).

| Features | Head | Val (best) | Test |
| --- | --- | --- | --- |
| 1×30 frames | MLP | 0.611 | 17/27 (0.63) |
| 1×30 frames | GRU h=128 | — | 15/27 |
| 1×30 frames | GRU small / bidir | — | 14 / 16 |
| 100 frames | MLP | 0.833 | 17/27 (0.63) |
| 100 frames | GRU / bidir | 0.667 | 17 / 14 |
| 8×30 frames | MLP | — | 17/27 |
| 8×30 frames | GRU | 0.556 | **18/27** (0.67) |
| 8×30 frames | GRU bidir | — | 13/27 |

**Conclusion:** temporal order does not beat mean pooling; GRU gains look like noise.

### 4.2 Omni MLP

| Features | Val | Test | Classes |
| --- | --- | --- | --- |
| Omni last-layer, 1×90-frame clip | 0.722 | **20/27** | 6 |
| Omni last-layer, 1×30-frame clip | 0.778 | (not re-logged here) | 6 |

Strong test score but **not CV’d**, expensive to extract, and includes Multilingual as a class. 8×60 all-layer Omni run aborted (~9 h estimate).

### 4.3 Linear probes (`src/probe.py`)

StandardScaler + class-balanced LogisticRegression; 5×5 RepeatedStratifiedKFold on train+val; fit on train+val; score test once.

| Features | Classes | Best CV | Test |
| --- | --- | --- | --- |
| jev grids (answer-pos, layer ~10–22) | 6 | ~0.76 | **20/27** |
| jev grids | 5 | 0.800 | 18/27 |
| jev video (answer-pos, layer 15) | 6 | ~0.76 | 16/27 |
| jev video | 5 | 0.797 | **19/27** |
| CLAP window mean | 6 | ~0.62 | 15/27 |
| CLAP | 5 | 0.600 | 15/27 |

### 4.4 Text-prompt experiments (`src/text_prompts.py`)

Methods on MobileCLIP video embeddings + text encoder:

1. **zero-shot** — cosine to domain text  
2. **embed-probe** — logistic on 512-d video embedding  
3. **sim-probe** — logistic on 28 (then 27) sub-category similarities  
4. **text-init** — linear head `W = W_text + δ` with ‖δ‖² penalty (CLIP logit scale)

#### 6-class (before Multilingual drop)

| Features | Zero-shot | Embed-probe | Sim-probe | Text-init |
| --- | --- | --- | --- | --- |
| 1×30 | 0.64 / 15 | 0.66 / 15 | 0.74 / 13 | 0.74 / 16 |
| 8×30 | 0.67 / 15 | 0.80 / 15 | 0.73 / 16 | **0.81 / 19** |
| 100 frames | 0.73 / 17 | 0.80 / 16 | 0.72 / 12 | **0.82 / 18** |

(Cells: CV / test hits out of 27.)

#### 5-class (`experiments/text_prompts_5class/`)

| Features | Zero-shot | Embed-probe | Sim-probe | Text-init |
| --- | --- | --- | --- | --- |
| 1×30 | 0.72 / 16 | 0.69 / 15 | 0.76 / 15 | 0.80 / 17 |
| 8×30 | 0.69 / 17 | 0.84 / 15 | 0.77 / 14 | 0.84 / 19 |
| **100 frames** | 0.74 / 18 | 0.83 / 16 | 0.75 / 14 | **0.85 / 19** |

Sim-probe underperforms embed-probe (28 similarities discard information). Text-init is the best **trained** method by CV.

---

## 5. Leaderboard (recommended ranking)

Prefer 5-class numbers; mark 6-class where needed.

### Zero-shot

| Rank | Model | Metric | Score |
| --- | --- | --- | --- |
| 1 | **MobileCLIP-S0, 100 frames, calibrated** | test / train+val | **21/27**, 0.84 |
| =1 | Same + CLAP fusion | test | 21/27 (no gain) |
| 2 | MobileCLIP-S0, 100 frames, uncalibrated | test | 18/27 |
| 3 | jev grids v2 (6-class) | test | 15/27 |
| 4 | Omni Thinker A/V (6-class) | test | 12/27 |
| 5 | CLAP audio calibrated | test | ≤16/27 |

### Trained

| Rank | Model | Metric | Score |
| --- | --- | --- | --- |
| 1 | **Text-init, MobileCLIP 100 frames (5-class)** | CV / test | **0.85**, 19/27 |
| 2 | Text-init, 8×30 clips (5-class) | CV / test | 0.84, 19/27 |
| 3 | jev video probe (5-class) | CV / test | 0.80, 19/27 |
| 4 | jev grid probe (5-class) | CV / test | 0.80, 18/27 |
| 5 | Omni MLP 90-frame (6-class) | val / test | 0.72, **20/27** |
| — | MobileCLIP MLP/GRU | test | 14–18/27 |

**Deploy recommendation:** calibrated MobileCLIP zero-shot (`src/mobileclip_zero_shot.py`) — best test score, no trained weights, ~20 s/video on CPU.

---

## 6. Scripts map

**Current main pipeline** (`src/`):

| Script | Role |
| --- | --- |
| `src/main.py` | CLI: evaluate or `--predict` |
| `src/pipeline.py` | MobileCLIP-S0 encode + text score + calibration |
| `src/dataset.py` | Load split jsonl files |
| `src/model.py` | MobileCLIP-S0 image encoder |
| `src/preprocessing.py` | Uniform / interval / both frame sampling |
| `src/prompts.py` | Domain / sub-category text descriptions |

**Archived experiments** (moved under `src/archive/`, not deleted):

| Script | Role |
| --- | --- |
| `archive/clip_sampling.py` | Contiguous clip sampling |
| `archive/train.py` | Embed cache + train MLP/GRU |
| `archive/model_heads.py` | `DomainMLP`, `DomainGRU` |
| `archive/zero_shot.py` | Omni Thinker zero-shot |
| `archive/omni_encoder.py` | Omni clip encoder |
| `archive/omni_gguf_zero_shot.py` | llama.cpp Omni GGUF zero-shot |
| `archive/jev_*.py` | jev-visual zero-shot / features / video |
| `archive/clap_zero_shot.py` | CLAP audio zero-shot |
| `archive/text_prompts.py` | Zero-shot / probes / text-init |
| `archive/probe.py` | Linear probe sweep |
| `archive/fuse_zero_shot.py` | MobileCLIP + CLAP fusion |
| `archive/mobileclip_zero_shot.py` | Earlier standalone zero-shot script |
| `data/drop_domain.py` | Remove a domain from splits |
| `data/dataset_split_process.py` | Original stratified split builder |

---

## 7. Result artifacts

| Path | Contents |
| --- | --- |
| `results/mobileclip_zero_shot/` | **Current** eval: summary + predictions |
| `models/mobileclip/calibration.json` | Shipped calibration for inference |
| `experiments/` | All earlier run outputs (see `experiments/README.md`) |
| `experiments/text_prompts/` | 6-class text experiments |
| `experiments/text_prompts_5class/` | 5-class text experiments |
| `experiments/probe/` | 6-class linear probes |
| `experiments/probe_5class/` | 5-class linear probes |
| `experiments/fusion/` | Per-video fused scores |
| `experiments/zero_shot/` | Per-video zero-shot JSONL |
| `experiments/logs/` | Full stdout logs for each run |
| `checkpoints/` | Trained MLP/GRU / Omni MLP weights (local) |

---

## 8. Failure modes and open ideas

**Recurring confusions**

- Knowledge ↔ Film & Television (talking heads / documentaries look like TV).
- Life Record ↔ Artistic Performance (home video vs stage-ish footage).
- CLAP: almost all speech → Film / Knowledge; Life Record near zero.

**Tried and weak**

- GRU over frame sequences; denser clips; raw CLAP; uncalibrated Film-biased CLIP; Multilingual as a class.

**Not finished / unused**

- Subtitle / ASR text fused with visual features (78/90 videos had subtitles before the Multilingual drop).
- Resume Omni 8×60 all-layer MLP and GGUF Omni zero-shot.
- `jev_frames.py` on interval-sampled frames.
- Retrain MLP/GRU on 5-class splits (probes and text-init were re-run; `train.py` heads were not).

---

## 9. How to reproduce the best model

```bash
# Evaluate uncalibrated vs calibrated on the current 5-class splits
python -u src/main.py

# Classify a new file (uses results/mobileclip_zero_shot/calibration.json)
python src/main.py --predict path/to/video.mp4
```

Frame features are reused from `data/cache/embeddings/both_n60_i1_m100/` when present.
