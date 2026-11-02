"""Pretrained model identity shared by training, evaluation and checkpoint restoration."""

from __future__ import annotations

import os
import platform
from dataclasses import asdict, dataclass

from .pretrained_assets import fingerprint


@dataclass(frozen=True)
class Workload:
    """Identify frozen encoders and fusion-head geometry independently of the execution device."""

    model_profile: str = "pretrained"
    camera_size: int = 224
    audio_size: int = 128
    schema_version: int = 1


def workload(profile=None):
    """Reject obsolete configuration rather than silently selecting a different model."""
    name = os.environ.get("HARBOUR_FL_MODEL_PROFILE", "pretrained") if profile is None else profile
    if name != "pretrained":
        raise ValueError("HARBOUR_FL_MODEL_PROFILE must be pretrained")
    return Workload()


def signature():
    """Record checkpoint-compatible model and preprocessing identities without hardware or batch size."""
    return {
        **asdict(workload()),
        "pretrained_assets": fingerprint(),
        "image_encoder": "imagenet-1k-v2-mobilenet-v3-large",
        "audio_encoder": "audioset-efficientat-mn10",
        "trainable_scope": "fusion-head-only-frozen-cached-encoders",
        "fusion": "layernorm-mask-concatenate-960+960+2-linear-256-relu-dropout-0.15",
        "audio_preprocessing": "raw-pcm-32khz-kaldi-128mel-stft1024-hop320-window800-v1",
        "image_preprocessing": "resize232-bilinear-centercrop224-imagenet-normalization",
        "feature_dimensions": {"audio": 960, "image": 960},
        "federation_algorithm": "sample-weighted-fedavg",
    }


def validate_checkpoint(checkpoint):
    """Require the recorded pretrained identity; incompatible checkpoints need a fresh job."""
    if checkpoint.get("workload") != signature():
        raise ValueError("checkpoint workload does not match the pretrained configuration; start a fresh job")


def describe(model, torch, device):
    """Record head size and execution identity; encoder extraction is a separate preparation step."""
    result = {
        **signature(),
        "device": str(device),
        "torch_version": str(torch.__version__),
        "platform": platform.system(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "torch_threads": torch.get_num_threads(),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "head_parameter_groups": {
            name: sum(p.numel() for p in branch.parameters()) for name, branch in model.named_children()
        },
        "state_tensor_bytes": sum(t.numel() * t.element_size() for t in model.state_dict().values()),
        "optimizer": "adamw",
        "optimizer_weight_decay": 0.01,
        "loss": "global-training-class-weighted-cross-entropy",
        "optimizer_scope": "reset-at-each-client-round",
        "learning_rate": float(os.environ.get("HARBOUR_FL_LEARNING_RATE", "0.001")),
        "encoder_execution": "offline-frozen-feature-preparation",
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result.update(accelerator=properties.name, accelerator_memory_bytes=properties.total_memory)
    elif device.type == "mps":
        result["accelerator"] = "Apple Metal Performance Shaders"
    return result
