#!/usr/bin/env python3
"""Smoke-test the calibrated MobileCLIP pipeline on the bundled sample clip.

Requires:
  - models/mobileclip/mobileclip_s0.pt   (scripts/download_mobileclip.py)
  - models/mobileclip/calibration.json  (shipped with the repo)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample" / "sports_clip.mp4"
EXPECTED = "Sports Competition"
CHECKPOINT = ROOT / "models" / "mobileclip" / "mobileclip_s0.pt"
CALIBRATION = ROOT / "models" / "mobileclip" / "calibration.json"


def main() -> None:
    if not SAMPLE.is_file():
        raise SystemExit(f"missing sample video: {SAMPLE}")
    if not CALIBRATION.is_file():
        raise SystemExit(f"missing calibration: {CALIBRATION}")
    if not CHECKPOINT.is_file():
        print("downloading MobileCLIP-S0 ...")
        subprocess.check_call([sys.executable, str(ROOT / "scripts" / "download_mobileclip.py")])

    result = subprocess.run(
        [sys.executable, str(ROOT / "src" / "main.py"), "--predict", str(SAMPLE)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    prediction_line = next((line for line in result.stdout.splitlines() if line.startswith("sports_clip")), None)
    if prediction_line is None:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise SystemExit("no prediction line in output")
    print(prediction_line)
    label = prediction_line.split(":", 1)[1].strip()
    if label != EXPECTED:
        raise SystemExit(f"expected {EXPECTED!r}, got {label!r}")
    print(f"ok: predicted {label}")


if __name__ == "__main__":
    main()
