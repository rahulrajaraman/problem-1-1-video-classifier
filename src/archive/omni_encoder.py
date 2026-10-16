"""Encode a clip's frames and audio together with the Qwen2.5-Omni Thinker.

Only the Thinker (vision tower, audio tower, and language model) is loaded.
The Talker and token2wav speech modules are not used. Each clip becomes one
vector: the chosen hidden layer averaged over the clip's video and audio
tokens.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from clip_sampling import Clip

OMNI_REPO = "Qwen/Qwen2.5-Omni-3B"
AUDIO_SAMPLE_RATE = 16000
PATCH_PIXELS = 28 * 28
PROMPT = "What kind of video is this?"


@dataclass(frozen=True)
class OmniSettings:
    repo: str = OMNI_REPO
    # Vision tokens per frame pair are about pixels / 784, so this bounds
    # the token count. Qwen's default floor is 128 * 28 * 28.
    min_pixels: int = 32 * PATCH_PIXELS
    max_pixels: int = 64 * PATCH_PIXELS
    # None keeps every hidden layer (embeddings first), so one pass serves any layer.
    layer: int | None = -1
    use_audio: bool = True

    def tag(self) -> str:
        name = self.repo.split("/")[-1].lower().replace(".", "")
        audio = "av" if self.use_audio else "v"
        layer = "all" if self.layer is None else self.layer
        return f"{name}_{audio}_p{self.max_pixels // PATCH_PIXELS}_l{layer}"


def clip_audio(video_path: Path, start_sec: float, duration_sec: float) -> np.ndarray:
    """Mono 16 kHz float32 audio for one clip window. Silence if the video has no audio."""
    command = [
        "ffmpeg", "-nostdin", "-loglevel", "error",
        "-ss", f"{start_sec:.3f}", "-t", f"{duration_sec:.3f}",
        "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", str(AUDIO_SAMPLE_RATE), "-f", "f32le", "-",
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    samples = np.frombuffer(result.stdout, dtype=np.float32)
    expected = max(1, int(round(duration_sec * AUDIO_SAMPLE_RATE)))
    if samples.size < expected:
        samples = np.pad(samples, (0, expected - samples.size))
    return samples[:expected].copy()


class OmniClipEncoder:
    def __init__(self, settings: OmniSettings, device: torch.device, keep_lm_head: bool = False):
        from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

        self.settings = settings
        self.device = device
        self.processor = Qwen2_5OmniProcessor.from_pretrained(settings.repo)
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(settings.repo, dtype=torch.bfloat16)
        # The model's forward returns final hidden states instead of logits. Applying
        # the 151k-way head to every position would cost gigabytes for long prompts,
        # so when it is kept it runs on the last position only.
        self.lm_head = model.lm_head.to(device).eval() if keep_lm_head else None
        model.lm_head = nn.Identity()
        self.model = model.to(device).eval()
        tokenizer = self.processor.tokenizer
        self.media_token_ids = torch.tensor(
            [tokenizer.convert_tokens_to_ids(self.processor.video_token),
             tokenizer.convert_tokens_to_ids(self.processor.audio_token)],
        )
        self.hidden_size = self.model.config.text_config.hidden_size

    def _prompt(self, question: str) -> str:
        processor = self.processor
        video = processor.vision_bos_token + processor.video_token + processor.vision_eos_token
        return (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{video}{question}<|im_end|>\n<|im_start|>assistant\n"
        )

    def prepare_inputs(self, video_path: Path, clip: Clip, question: str = PROMPT) -> dict[str, torch.Tensor]:
        settings = self.settings
        audio = clip_audio(video_path, clip.start_sec, clip.duration_sec) if settings.use_audio else None
        inputs = self.processor(
            text=self._prompt(question),
            videos=[clip.frames],
            audio=[audio] if audio is not None else None,
            use_audio_in_video=settings.use_audio,
            fps=clip.fps,
            size={"shortest_edge": settings.min_pixels, "longest_edge": settings.max_pixels},
            return_tensors="pt",
        )
        return {
            key: value.to(self.device, torch.bfloat16) if value.is_floating_point() else value.to(self.device)
            for key, value in inputs.items()
            if isinstance(value, torch.Tensor)
        }

    @torch.no_grad()
    def next_token_logits(self, video_path: Path, clip: Clip, question: str) -> torch.Tensor:
        """Vocabulary logits for the first token of the Thinker's answer: [vocab]."""
        if self.lm_head is None:
            raise RuntimeError("create the encoder with keep_lm_head=True to score answers")
        inputs = self.prepare_inputs(video_path, clip, question)
        outputs = self.model(**inputs, use_audio_in_video=self.settings.use_audio, use_cache=False)
        return self.lm_head(outputs.logits[0, -1]).float().cpu()

    @torch.no_grad()
    def encode_clip(self, video_path: Path, clip: Clip) -> torch.Tensor:
        """[hidden_size] for one layer, or [layers + 1, hidden_size] when layer is None."""
        settings = self.settings
        inputs = self.prepare_inputs(video_path, clip)
        outputs = self.model(
            **inputs,
            use_audio_in_video=settings.use_audio,
            output_hidden_states=True,
            use_cache=False,
        )
        input_ids = inputs["input_ids"][0]
        media = torch.isin(input_ids, self.media_token_ids.to(input_ids.device))
        if not media.any():
            media = torch.ones_like(input_ids, dtype=torch.bool)
        layers = outputs.hidden_states if settings.layer is None else [outputs.hidden_states[settings.layer]]
        pooled = torch.stack([hidden[0][media].float().mean(dim=0) for hidden in layers]).cpu()
        return pooled if settings.layer is None else pooled[0]

    def encode_clips(self, video_path: Path, clips: list[Clip]) -> torch.Tensor:
        """[clips, hidden_size], or [clips, layers + 1, hidden_size] when layer is None."""
        features = [self.encode_clip(video_path, clip) for clip in clips]
        if self.device.type == "mps":
            torch.mps.empty_cache()
        return torch.stack(features)
