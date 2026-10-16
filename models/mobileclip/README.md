# MobileCLIP-S0 model package

Everything the classifier needs at inference time lives here:

| File | In git? | Role |
| --- | --- | --- |
| `calibration.json` | yes | Per-domain score offsets (train+val means) |
| `mobileclip_s0.pt` | no (~206 MB) | Encoder weights |

```bash
python scripts/download_mobileclip.py   # fetches mobileclip_s0.pt
python src/main.py --predict data/sample/sports_clip.mp4
```

`python src/main.py` (eval) refreshes `calibration.json` from the Video-MME splits and writes predictions under `results/`.
