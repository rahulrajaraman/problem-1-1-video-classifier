"""Calibrated MobileCLIP-S0 zero-shot domain classifier."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from model import embed_frames, load_mobileclip
from preprocessing import FrameSampling, sample_video_frames
from prompts import SUBCATEGORIES, TEMPLATES

SAMPLING = FrameSampling(mode="both", num_frames=60, interval_sec=1, max_frames=100)


def log_softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=-1, keepdims=True)
    return logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))


def mean_pool(frame_features: torch.Tensor) -> np.ndarray:
    mean = frame_features.float().reshape(-1, frame_features.shape[-1]).mean(dim=0)
    return torch.nn.functional.normalize(mean, dim=0).numpy()


class MobileClipZeroShot:
    """Mean-pooled MobileCLIP-S0 frames vs domain text prompts, with optional calibration."""

    def __init__(
        self,
        cache_dir: Path,
        device: torch.device | str = "cpu",
        checkpoint: Path | None = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.device = torch.device(device)
        self.checkpoint = checkpoint
        self.model = None
        self.preprocess = None
        self.names: list[str] = []
        self.domain_text: np.ndarray | None = None
        self.scale: float = 1.0
        self.offsets: np.ndarray | None = None

    def _ensure_model(self) -> None:
        if self.model is None:
            self.model, self.preprocess = load_mobileclip(self.checkpoint, self.device)

    def encode_video(self, video_path: Path, cache: bool = True) -> np.ndarray:
        """Return one L2-normalised video embedding from sampled frames."""
        video_path = Path(video_path)
        cached = self.cache_dir / f"{video_path.stem}.pt"
        if cache and cached.is_file():
            frames = torch.load(cached, weights_only=True)
        else:
            self._ensure_model()
            frames = embed_frames(self.model, self.preprocess, sample_video_frames(video_path, SAMPLING), self.device)
            if cache:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                torch.save(frames, cached)
        return mean_pool(frames)

    @torch.no_grad()
    def build_text(self, names: list[str]) -> None:
        """Embed domain prompts with MobileCLIP's text encoder."""
        import mobileclip

        self._ensure_model()
        tokenizer = mobileclip.get_tokenizer("mobileclip_s0")
        vectors = []
        for name in names:
            if name not in SUBCATEGORIES:
                raise KeyError(f"no prompts for domain {name!r}")
            parts = []
            for description in SUBCATEGORIES[name].values():
                tokens = tokenizer([template.format(description) for template in TEMPLATES]).to(self.device)
                embedded = torch.nn.functional.normalize(self.model.encode_text(tokens).float(), dim=-1).mean(dim=0)
                parts.append(torch.nn.functional.normalize(embedded, dim=0))
            vectors.append(torch.nn.functional.normalize(torch.stack(parts).mean(dim=0), dim=0))
        self.names = list(names)
        self.domain_text = torch.stack(vectors).cpu().numpy()
        self.scale = float(self.model.logit_scale.exp())

    def scores(self, video_embedding: np.ndarray) -> np.ndarray:
        if self.domain_text is None:
            raise RuntimeError("call build_text() before scoring")
        return log_softmax(self.scale * video_embedding @ self.domain_text.T)

    def calibrate(self, score_matrix: np.ndarray) -> np.ndarray:
        """Set and return per-domain mean log-prob offsets (label-free)."""
        self.offsets = score_matrix.mean(axis=0)
        return self.offsets

    def load_calibration(self, path: Path) -> None:
        data = json.loads(Path(path).read_text())
        self.names = list(data["names"])
        self.offsets = np.array(data["offsets"])
        if self.domain_text is None:
            self.build_text(self.names)

    def save_calibration(self, path: Path, calibration_splits: list[str]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "names": self.names,
            "offsets": self.offsets.tolist(),
            "calibration_splits": calibration_splits,
            "sampling": SAMPLING.tag(),
        }, indent=2))

    def predict_scores(self, score_row: np.ndarray, calibrated: bool = True) -> str:
        if calibrated:
            if self.offsets is None:
                raise RuntimeError("no calibration offsets loaded")
            score_row = score_row - self.offsets
        return self.names[int(score_row.argmax())]

    def predict(self, video_path: Path, calibrated: bool = True, cache: bool = True) -> str:
        return self.predict_scores(self.scores(self.encode_video(video_path, cache=cache)), calibrated=calibrated)
