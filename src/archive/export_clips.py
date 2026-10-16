#!/usr/bin/env python3
"""Save the clips a training run sampled, for inspection.

Clip starts depend only on the seed, the video id, and the video's frame
count, so the same --clip-len, --num-clips, and --seed reproduce the clips a
run used. For each clip this writes:

- an .mp4 of the clip window with its audio, cut from the source with ffmpeg
- a .jpg contact sheet of the exact decoded frames the encoder received

plus an index.jsonl listing every clip's start time and duration.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

from clip_sampling import Clip, ClipSampling, sample_clips
from dataset import read_jsonl

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")


def write_clip_video(source: Path, clip: Clip, destination: Path) -> None:
    command = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
        "-ss", f"{clip.start_sec:.3f}", "-i", str(source), "-t", f"{clip.duration_sec:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-movflags", "+faststart", str(destination),
    ]
    subprocess.run(command, check=True)


def write_contact_sheet(clip: Clip, destination: Path, columns: int = 6, thumb_width: int = 256) -> None:
    first = clip.frames[0]
    thumb_height = round(first.height * thumb_width / first.width)
    rows = (len(clip.frames) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_width, rows * thumb_height), "black")
    draw = ImageDraw.Draw(sheet)
    for index, frame in enumerate(clip.frames):
        x = (index % columns) * thumb_width
        y = (index // columns) * thumb_height
        sheet.paste(frame.resize((thumb_width, thumb_height)), (x, y))
        timestamp = (clip.start_frame + index) / clip.fps
        draw.text((x + 4, y + 4), f"{clip.start_frame + index}  {timestamp:.2f}s", fill="yellow")
    sheet.save(destination, quality=85)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split-dir", type=Path, default=ROOT / "data" / "splits")
    parser.add_argument("--video-dir", type=Path, default=ROOT / "data" / "video-mme-sample10" / "videos")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "debug" / "clips")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--video-ids", nargs="*", default=None, help="Only these videos. Default is every video in the splits.")
    parser.add_argument("--clip-len", type=int, default=30)
    parser.add_argument("--num-clips", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-video", action="store_true", help="Write contact sheets only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sampling = ClipSampling(args.clip_len, args.num_clips, args.seed)
    run_dir = args.output_dir / sampling.tag()
    wanted = set(args.video_ids) if args.video_ids else None

    index_rows = []
    for split in args.splits:
        rows = read_jsonl(args.split_dir / f"{split}.jsonl")
        if wanted is not None:
            rows = [row for row in rows if row["video_id"] in wanted]
        split_dir = run_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for row in rows:
            video_id = row["video_id"]
            source = args.video_dir / f"{video_id}.mp4"
            for clip_index, clip in enumerate(sample_clips(source, sampling, video_id)):
                stem = f"{video_id}_c{clip_index}_{clip.start_sec:08.2f}s"
                write_contact_sheet(clip, split_dir / f"{stem}.jpg")
                if not args.no_video:
                    write_clip_video(source, clip, split_dir / f"{stem}.mp4")
                index_rows.append(
                    {
                        "split": split,
                        "video_id": video_id,
                        "domain": row["domain"],
                        "duration": row["duration"],
                        "clip": clip_index,
                        "start_frame": clip.start_frame,
                        "start_sec": round(clip.start_sec, 3),
                        "duration_sec": round(clip.duration_sec, 3),
                        "fps": round(clip.fps, 3),
                        "file": f"{split}/{stem}",
                    }
                )
            print(f"{split} {video_id}", flush=True)

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "index.jsonl").write_text("".join(json.dumps(row) + "\n" for row in index_rows))
    print(f"wrote {len(index_rows)} clips to {run_dir}")


if __name__ == "__main__":
    main()
