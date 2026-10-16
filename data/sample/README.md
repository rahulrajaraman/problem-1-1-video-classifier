# Sample video for smoke-testing the pipeline

| File | Source | Expected domain |
| --- | --- | --- |
| `sports_clip.mp4` | 4 s / 320p clip from Video-MME `913JhnW28U4` (Sports Competition, short) | Sports Competition |

Uses the model package under `models/mobileclip/` (weights + `calibration.json`):

```bash
python scripts/download_mobileclip.py
python src/main.py --predict data/sample/sports_clip.mp4
# or
python scripts/smoke_test.py
```
