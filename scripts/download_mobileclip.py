#!/usr/bin/env python3
"""Download MobileCLIP-S0 weights into models/mobileclip/ if missing."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = ROOT / "models" / "mobileclip" / "mobileclip_s0.pt"
REPO = "apple/MobileCLIP-S0"
FILENAME = "mobileclip_s0.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--force", action="store_true", help="Re-download even if the file exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dest = Path(args.dest)
    if dest.is_file() and not args.force:
        print(f"already present: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return

    from huggingface_hub import hf_hub_download

    print(f"downloading {REPO}/{FILENAME} ...")
    cached = hf_hub_download(REPO, FILENAME)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached, dest)
    print(f"saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
