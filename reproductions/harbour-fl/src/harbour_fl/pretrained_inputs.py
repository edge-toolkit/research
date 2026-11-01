"""Raw-waveform EfficientAT frontend and deterministic input corruption."""

import hashlib
import math
import wave

import numpy as np
import torch
from PIL import Image
from scipy.signal import resample_poly
from torch import nn
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

CONDITIONS = {
    "clean": {},
    "audio-snr20": {"snr_db": 20},
    "audio-snr10": {"snr_db": 10},
    "audio-snr0": {"snr_db": 0},
    "image-noise005": {"image_std": 0.05},
    "image-noise010": {"image_std": 0.1},
    "image-noise020": {"image_std": 0.2},
    "missing-audio": {"missing": "audio"},
    "missing-image": {"missing": "image"},
}


def sample_seed(segment_id, condition, seed):
    return int(hashlib.sha256(f"{seed}:{segment_id}:{condition}".encode()).hexdigest()[:15], 16)


def decode_audio(row):
    """Decode a bounded PCM WAV segment, mix to mono, resample to 32 kHz and reject invalid or silent audio."""
    with wave.open(row["audio"], "rb") as source:
        rate, channels, width = source.getframerate(), source.getnchannels(), source.getsampwidth()
        start, count = round(row["start"] * rate), round((row["end"] - row["start"]) * rate)
        if start + count > source.getnframes():
            raise ValueError(f"segment exceeds recording duration: {row['segment_id']}")
        source.setpos(start)
        raw = source.readframes(count)
    if width == 3:
        values = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        audio = values[:, 0] | (values[:, 1] << 8) | (values[:, 2] << 16)
        audio = ((audio ^ 0x800000) - 0x800000).astype(np.float32) / 8388608
    elif width in (1, 2, 4):
        dtype = {1: np.uint8, 2: "<i2", 4: "<i4"}[width]
        audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        audio = (audio - 128) / 128 if width == 1 else audio / 2 ** (8 * width - 1)
    else:
        raise ValueError("supported WAV encoding: 8/16/24/32-bit PCM")
    audio = audio.reshape(-1, channels).mean(1)
    divisor = math.gcd(rate, 32000)
    audio = resample_poly(audio, 32000 // divisor, rate // divisor).astype(np.float32)
    if not np.isfinite(audio).all() or len(audio) < 1024 or np.mean(audio**2) <= 0:
        raise ValueError(f"invalid or silent clean audio: {row['segment_id']}")
    return torch.from_numpy(audio.copy())


class AudioFrontend(nn.Module):
    """EfficientAT evaluation frontend: Kaldi triangles, 128 mel bins, 32 kHz, no augmentation.

    Equivalent evaluation operations to pinned EfficientAT AugmentMelSTFT without a torchaudio dependency.
    Complex STFT replaces the legacy real/imaginary output representation.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("window", torch.hann_window(800, periodic=False))
        frequencies = torch.arange(512, dtype=torch.float32) * (32000 / 1024)
        mel = 1127 * torch.log1p(frequencies / 700)
        edges = torch.linspace(0, 1127 * math.log1p(15000 / 700), 130)
        lower = (mel[None] - edges[:-2, None]) / (edges[1:-1, None] - edges[:-2, None])
        upper = (edges[2:, None] - mel[None]) / (edges[2:, None] - edges[1:-1, None])
        self.register_buffer("bank", nn.functional.pad(torch.minimum(lower, upper).clamp_min(0), (0, 1)))

    def forward(self, waveforms):
        emphasized = waveforms[:, 1:] - 0.97 * waveforms[:, :-1]
        spectrum = torch.stft(
            emphasized, 1024, hop_length=320, win_length=800, window=self.window, center=True, return_complex=True
        )
        return ((self.bank @ spectrum.abs().square() + 0.00001).log() + 4.5).div(5).unsqueeze(1)


def materialize(row, condition, seed):
    """Apply deterministic waveform or image corruption before producing encoder-ready paired inputs."""
    options = CONDITIONS[condition]
    generator = torch.Generator().manual_seed(sample_seed(row["segment_id"], condition, seed))
    audio = decode_audio(row)
    if "snr_db" in options:
        noise = torch.randn(audio.shape, generator=generator)
        noise *= audio.square().mean().sqrt() / noise.square().mean().sqrt() / 10 ** (options["snr_db"] / 20)
        audio = audio + noise
    with Image.open(row["image"]) as picture:
        picture = picture.convert("RGB")
        picture = TF.center_crop(TF.resize(picture, 232, interpolation=InterpolationMode.BILINEAR), 224)
        image = TF.to_tensor(picture)
    if "image_std" in options:
        image = (image + torch.randn(image.shape, generator=generator) * options["image_std"]).clamp(0, 1)
    image = TF.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return audio, image
