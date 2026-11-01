"""Strict pretrained encoders and a presence-aware fusion head."""

import contextlib
import hashlib
import importlib
import importlib.util
import io
import sys
from pathlib import Path

import torch
from torch import nn
from torchvision.models import mobilenet_v3_large

from .pretrained_assets import verify
from .pretrained_inputs import AudioFrontend
from .settings import _bool_env


class Encoders(nn.Module):
    def __init__(self, assets):
        """Verify pinned artifacts and load frozen evaluation-mode EfficientAT and MobileNetV3 feature encoders."""
        super().__init__()
        root = Path(assets).resolve()
        self.identity = verify(root)
        namespace = "_harbour_efficientat_" + hashlib.sha256(str(root).encode()).hexdigest()[:16]
        if namespace not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                namespace, root / "efficientat/__init__.py", submodule_search_locations=[str(root / "efficientat")]
            )
            package = importlib.util.module_from_spec(spec)
            sys.modules[namespace] = package
            spec.loader.exec_module(package)
        module = importlib.import_module(namespace + ".model")
        with contextlib.redirect_stdout(io.StringIO()):
            audio = module.get_model(pretrained_name=None, width_mult=1.0)
        audio.load_state_dict(torch.load(root / "mn10_as.pt", map_location="cpu", weights_only=True), strict=True)
        self.audio = nn.Sequential(audio.features, nn.AdaptiveAvgPool2d(1), nn.Flatten(1))
        image = mobilenet_v3_large(weights=None)
        image.load_state_dict(
            torch.load(root / "mobilenet_v3_large.pth", map_location="cpu", weights_only=True), strict=True
        )
        self.image = nn.Sequential(image.features, nn.AdaptiveAvgPool2d(1), nn.Flatten(1))
        self.frontend = AudioFrontend()
        self.requires_grad_(False)
        self.eval()

    def place(self, device):
        """Run CNNs on the requested backend; keep the complex STFT on CPU for MPS."""
        device = torch.device(device)
        self.audio.to(device)
        self.image.to(device)
        self.frontend.to("cpu" if device.type == "mps" else device)
        return self

    def forward(self, waveforms, images):
        with torch.inference_mode():
            mel = self.frontend(waveforms.to(self.frontend.window.device)).to(images.device)
            return self.audio(mel), self.image(images)


class Fusion(nn.Module):
    def __init__(self, classes, dimension=960):
        super().__init__()
        self.dimension = dimension
        self.audio_norm, self.image_norm = nn.LayerNorm(dimension), nn.LayerNorm(dimension)
        self.classifier = nn.Sequential(
            nn.Linear(2 * dimension + 2, 256), nn.ReLU(), nn.Dropout(0.15), nn.Linear(256, classes)
        )

    def forward(self, audio, images, presence=None):
        """Normalize and mask modality embeddings, optionally drop a training modality and compute class logits."""
        if presence is None:
            presence = torch.stack(
                [torch.isfinite(value).all(1) & (value.abs().sum(1) > 0) for value in (audio, images)], dim=1
            ).to(audio.dtype)
        if self.training and _bool_env("HARBOUR_FL_ADAPTATION"):
            choice = torch.rand(len(audio), device=audio.device)
            presence = presence.clone()
            presence[choice < 0.15, 0] = 0
            presence[(choice >= 0.15) & (choice < 0.3), 1] = 0
        audio = torch.nan_to_num(audio, nan=0, posinf=0, neginf=0)
        images = torch.nan_to_num(images, nan=0, posinf=0, neginf=0)
        # Mask after normalization so missing branches cannot leak LayerNorm offsets.
        audio = self.audio_norm(audio) * presence[:, :1]
        images = self.image_norm(images) * presence[:, 1:]
        return self.classifier(torch.cat((audio, images, presence), dim=1))
