"""MobileCLIP-S0 image encoder."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import Compose

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "mobileclip"
DEFAULT_CHECKPOINT = MODEL_DIR / "mobileclip_s0.pt"
DEFAULT_CALIBRATION = MODEL_DIR / "calibration.json"


def mobileclip_checkpoint(path: Path | None = None) -> str:
    """Resolve the MobileCLIP-S0 weight file.

    Prefers an explicit path, then the vendored copy under models/mobileclip/,
    then a Hugging Face download as a last resort.
    """
    if path is not None:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"MobileCLIP checkpoint not found: {path}")
        return str(path)
    if DEFAULT_CHECKPOINT.is_file():
        return str(DEFAULT_CHECKPOINT)
    from huggingface_hub import hf_hub_download

    return hf_hub_download("apple/MobileCLIP-S0", "mobileclip_s0.pt")


def load_mobileclip(checkpoint: Path | None = None, device: torch.device | None = None):
    """Load MobileCLIP-S0 and the official image transform (resize, center crop, ToTensor)."""
    import mobileclip

    device = device or torch.device("cpu")
    model, _, preprocess = mobileclip.create_model_and_transforms(
        "mobileclip_s0",
        pretrained=mobileclip_checkpoint(checkpoint),
        reparameterize=True,
        device=device,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, preprocess


@torch.no_grad()
def embed_frames(
    encoder,
    preprocess: Compose,
    frames: list[Image.Image],
    device: torch.device,
) -> torch.Tensor:
    """Run MobileCLIP-S0 on each frame; return L2-normalised embeddings [T, D]."""
    batch = torch.stack([preprocess(frame) for frame in frames]).to(device)
    return encoder.encode_image(batch, normalize=True).detach().cpu()
