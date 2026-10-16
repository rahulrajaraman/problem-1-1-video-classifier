"""Sample windows of consecutive frames from a video."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
from PIL import Image


@dataclass(frozen=True)
class ClipSampling:
    """Take `num_clips` windows of `clip_len` consecutive frames per video.

    With one clip, the start frame is drawn at random from the whole video.
    With several clips, the video is cut into `num_clips` equal segments and
    one start is drawn inside each segment, so clips spread across the video
    and do not overlap when the video is long enough.

    Starts are drawn from a generator seeded on `seed` and the video id, so a
    given video always gets the same clips for the same seed.
    """

    clip_len: int = 30
    num_clips: int = 1
    seed: int = 42

    def __post_init__(self) -> None:
        if self.clip_len < 1:
            raise ValueError("clip_len must be at least 1")
        if self.num_clips < 1:
            raise ValueError("num_clips must be at least 1")

    def tag(self) -> str:
        return f"clips_l{self.clip_len}_c{self.num_clips}_s{self.seed}"


def clip_starts(num_frames_total: int, sampling: ClipSampling, video_id: str) -> list[int]:
    """Start frame of each clip, in timestamp order."""
    rng = random.Random(f"{sampling.seed}:{video_id}")
    latest_start = max(0, num_frames_total - sampling.clip_len)
    if sampling.num_clips == 1:
        return [rng.randint(0, latest_start)]

    starts = []
    segment = (latest_start + 1) / sampling.num_clips
    for index in range(sampling.num_clips):
        low = int(index * segment)
        high = max(low, int((index + 1) * segment) - 1)
        starts.append(min(latest_start, rng.randint(low, high)))
    return starts


def _read_clip(capture: cv2.VideoCapture, start: int, clip_len: int, path: Path) -> list[Image.Image]:
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames: list[Image.Image] = []
    for _ in range(clip_len):
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    if not frames:
        raise RuntimeError(f"failed to read a clip at frame {start} from {path}")
    # Videos shorter than clip_len, or a frame count that overstates the
    # decodable length, are padded with the last frame to keep clips equal size.
    while len(frames) < clip_len:
        frames.append(frames[-1])
    return frames


@dataclass
class Clip:
    frames: list[Image.Image]
    start_frame: int
    fps: float

    @property
    def start_sec(self) -> float:
        return self.start_frame / self.fps

    @property
    def duration_sec(self) -> float:
        return len(self.frames) / self.fps


def sample_clips(path: Path, sampling: ClipSampling, video_id: str) -> list[Clip]:
    """Decode each clip with its start frame and the video frame rate."""
    path = Path(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"could not open video: {path}")
    try:
        num_frames_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if num_frames_total <= 0:
            raise RuntimeError(f"video has no readable frames: {path}")
        if fps <= 0:
            fps = 30.0
        starts = clip_starts(num_frames_total, sampling, video_id)
        return [Clip(_read_clip(capture, start, sampling.clip_len, path), start, fps) for start in starts]
    finally:
        capture.release()


def sample_video_clips(path: Path, sampling: ClipSampling, video_id: str) -> list[list[Image.Image]]:
    """Decode each clip as a list of `clip_len` RGB frames."""
    return [clip.frames for clip in sample_clips(path, sampling, video_id)]
