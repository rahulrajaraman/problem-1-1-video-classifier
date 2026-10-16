"""Sample frames from a video, uniformly and at a fixed time interval."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
from PIL import Image


@dataclass(frozen=True)
class FrameSampling:
    """How many frames to take from each video.

    `uniform` spreads `num_frames` across the whole video.
    `interval` takes one frame every `interval_sec` seconds.
    `both` keeps the union of those two sets. If that union is longer than
    `max_frames`, it is subsampled uniformly so long videos stay bounded.
    """

    mode: str = "both"
    num_frames: int = 60
    interval_sec: float = 1
    max_frames: int = 100

    def __post_init__(self) -> None:
        if self.mode not in {"uniform", "interval", "both"}:
            raise ValueError(f"sampling mode must be uniform, interval, or both, got {self.mode}")
        if self.num_frames < 1:
            raise ValueError("num_frames must be at least 1")
        if self.interval_sec <= 0:
            raise ValueError("interval_sec must be positive")
        if self.max_frames < 1:
            raise ValueError("max_frames must be at least 1")

    def tag(self) -> str:
        interval = f"{self.interval_sec:g}"
        return f"{self.mode}_n{self.num_frames}_i{interval}_m{self.max_frames}"


def uniform_indices(num_frames_total: int, num_samples: int) -> list[int]:
    """Indices spread evenly from the first frame to the last."""
    if num_frames_total <= 0 or num_samples <= 0:
        return []
    if num_samples == 1 or num_frames_total == 1:
        return [min(num_frames_total - 1, num_frames_total // 2)]
    if num_samples >= num_frames_total:
        return list(range(num_frames_total))
    last = num_frames_total - 1
    return [min(last, int(round(i * last / (num_samples - 1)))) for i in range(num_samples)]


def interval_indices(num_frames_total: int, fps: float, interval_sec: float) -> list[int]:
    """One index every `interval_sec` seconds, starting at frame 0."""
    if num_frames_total <= 0:
        return []
    if fps <= 0:
        return uniform_indices(num_frames_total, num_samples=1)
    step = max(1, int(round(interval_sec * fps)))
    return list(range(0, num_frames_total, step))


def _subsample(indices: list[int], limit: int) -> list[int]:
    if len(indices) <= limit:
        return indices
    picked = uniform_indices(len(indices), limit)
    return [indices[i] for i in picked]


def frame_indices(num_frames_total: int, fps: float, sampling: FrameSampling) -> list[int]:
    uniform = uniform_indices(num_frames_total, sampling.num_frames)
    interval = interval_indices(num_frames_total, fps, sampling.interval_sec)
    if sampling.mode == "uniform":
        chosen = uniform
    elif sampling.mode == "interval":
        chosen = interval
    else:
        chosen = sorted(set(uniform) | set(interval))
    ordered = list(dict.fromkeys(chosen))
    return _subsample(ordered, sampling.max_frames)


def sample_video_frames(path: Path, sampling: FrameSampling) -> list[Image.Image]:
    """Decode the sampled frames as RGB images, in timestamp order."""
    path = Path(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"could not open video: {path}")
    try:
        num_frames_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if num_frames_total <= 0:
            raise RuntimeError(f"video has no readable frames: {path}")
        indices = frame_indices(num_frames_total, fps, sampling)
        frames: list[Image.Image] = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                ok, frame = capture.read()
            if not ok:
                if frames:
                    frames.append(frames[-1])
                    continue
                raise RuntimeError(f"failed to read frame {index} from {path}")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
        return frames
    finally:
        capture.release()
