"""Split-learning parameter server for `et-ws-pyo3-runner`.

The training half of `scripts/server.py` from the `frontend-train-tensorflow` branch of
https://github.com/jayvdb/split-learning-demo, on the pyo3 runner's hook contract. Upstream, the browser connects to a
FastAPI WebSocket endpoint; here it connects to `et-ws-server` instead, the hub relays its binary frames to this agent,
and the frames this agent returns are relayed back. The wire format is unchanged -- `base64(json(WSMessage))` binary
frames -- so the upstream web app only has to be pointed at the hub's `/ws`.

Only the messages a browser needs to train and then use the model are handled:

- `request_batch` -> `batch`: the next MNIST training batch, for the browser to run its half of the model over.
- `activations_and_labels` -> `grads`: one training step of the server half, returning the cut-layer gradients.
- `reset_server`: re-initialise the server half, which the browser sends before each fresh run.
- `activations` -> `logits`: inference through the server half, for a digit drawn in the browser.

The rest of upstream's server -- storing the browser's client half, logging statistics -- is left out. The trained
server half is stored on shutdown as `server_mnist.onnx` in this agent's ws-server storage bucket, and loaded back
from there on connect, so a restarted agent still answers inference with what it learned.

Configuration, through the runner's environment:

- `SPLIT_LEARNING_LEARNING_RATE` -- SGD learning rate, default `0.01` (must match the browser's).
- `SPLIT_LEARNING_BATCH_SIZE` -- batch size streamed to the browser, default `128`.
- `SPLIT_LEARNING_ACCELERATOR` -- Lightning Fabric accelerator, default `auto`.

`split_learning` is the upstream package, imported from its source tree via `PYO3_PYTHONPATH`; it also locates the
MNIST parquet files, relative to that tree, under `data/external/mnist/`.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import lightning as L
import onnx
import torch
from onnx import numpy_helper
from split_learning.models.vision.cnn_2d import CNN2D, CNN2DServer
from split_learning.schemas.message import MessageType, WSMessage
from split_learning.utils import datasets
from split_learning.utils.serde import decode_message_b64, deserialize_tensor, encode_message_b64, serialize_tensor
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

_logger = logging.getLogger(__name__)

SERVER_WEIGHTS_KEY = "server_mnist.onnx"

# The post-cut activation shape the server half consumes, less the batch dimension.
ACTIVATION_SHAPE = (16, 7, 7)

# Module-level state, held in a container rather than rebindable names -- the runner contract is one module instance
# per process, and `init` only has to populate this.
_state: dict[str, Any] = {}


def _mnist_batches(batch_size: int):
    """Yield MNIST training batches forever, one pass over the loader per browser epoch.

    The transform matches upstream's, so the browser trains on the same augmentation as upstream's Python client.
    """
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(28, padding=4),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    dataset = datasets.mnist(split="train", transform=train_transform)
    # No worker processes: an embedded interpreter has no `python` executable for a spawned worker to start from.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    epoch = 0
    while True:
        yield from loader
        epoch += 1
        _logger.info("browser epoch %d complete, mean loss %.4f", epoch, _take_epoch_loss())


def _take_epoch_loss() -> float:
    """Return the mean training loss since the last call, and start a new window."""
    total, count = _state["epoch_loss"]
    _state["epoch_loss"] = (0.0, 0)
    return total / max(count, 1)


def _build_server_half() -> None:
    """(Re)build the server half with random weights and a fresh optimizer."""
    backbone = CNN2D(in_channels=1, dim_out=10, img_size=28, dropout=0.15)
    model = CNN2DServer(in_channels=1, dim_out=10, img_size=28, model=backbone)
    optimizer = torch.optim.SGD(model.parameters(), lr=_state["learning_rate"], momentum=0.9)
    _state["unwrapped_model"] = model
    _state["model"], _state["optimizer"] = _state["fabric"].setup(model, optimizer)
    _state["trained"] = False
    _state["epoch_loss"] = (0.0, 0)


def init(_send, storage) -> None:
    """Build the untrained server half."""
    _state["storage"] = storage
    _state["learning_rate"] = float(os.environ.get("SPLIT_LEARNING_LEARNING_RATE", "0.01"))
    _state["batch_size"] = int(os.environ.get("SPLIT_LEARNING_BATCH_SIZE", "128"))
    accelerator = os.environ.get("SPLIT_LEARNING_ACCELERATOR", "auto")
    _logger.info(
        "split-learning server: learning_rate=%s batch_size=%s accelerator=%s",
        _state["learning_rate"],
        _state["batch_size"],
        accelerator,
    )
    # One device, whatever the accelerator finds: with several GPUs, Fabric's own `devices="auto"` would pick them
    # all and launch a process per device, which an embedded interpreter -- with no `python` executable to start them
    # from -- cannot do.
    fabric = L.Fabric(accelerator=accelerator, devices=1, precision="32-true")
    fabric.launch()
    _state["fabric"] = fabric
    _state["criterion"] = nn.CrossEntropyLoss()
    _state["batches"] = None
    _build_server_half()


def on_connect(agent_id: str) -> None:
    """Load the server half a previous run stored, if there is one."""
    blob = _state["storage"].get(agent_id, SERVER_WEIGHTS_KEY)
    if blob is None:
        _logger.info("no stored weights at %s/%s; starting from random", agent_id, SERVER_WEIGHTS_KEY)
        return
    # `onnx.load` takes a path, so the blob goes through a temporary file.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / SERVER_WEIGHTS_KEY
        path.write_bytes(bytes(blob))
        initializers = {
            init.name: torch.from_numpy(numpy_helper.to_array(init).copy())
            for init in onnx.load(str(path)).graph.initializer
        }
    model = _state["unwrapped_model"]
    state_dict = model.state_dict()
    missing = [key for key, value in state_dict.items() if getattr(initializers.get(key), "shape", None) != value.shape]
    if missing:
        _logger.warning(
            "stored weights at %s/%s do not fit the model (%s); starting from random",
            agent_id,
            SERVER_WEIGHTS_KEY,
            missing,
        )
        return
    model.load_state_dict({key: initializers[key] for key in state_dict})
    _logger.info("loaded %d-byte server weights from %s/%s", len(blob), agent_id, SERVER_WEIGHTS_KEY)


def on_binary_frame(frame: bytes) -> bytes | None:
    """Dispatch one relayed frame from the browser; return the reply frame, if the message has one."""
    try:
        message = decode_message_b64(bytes(frame))
    except Exception as exc:  # noqa: BLE001 -- an undecodable frame is some other agent's, not a fault here
        _logger.debug("ignoring undecodable binary frame: %s", exc)
        return None
    if message.type == MessageType.REQUEST_BATCH:
        return _next_batch()
    if message.type == MessageType.ACTIVATIONS_AND_LABELS:
        return _train_step(message)
    if message.type == MessageType.ACTIVATIONS:
        return _infer(message)
    if message.type == MessageType.RESET_SERVER:
        _build_server_half()
        _logger.info("server half re-initialised (random weights, fresh optimizer)")
        return None
    _logger.debug("ignoring %s message", message.type)
    return None


def on_shutdown() -> None:
    """Store the trained server half as ONNX in this agent's storage bucket."""
    if not _state.get("trained"):
        return
    example_input = torch.zeros(1, *ACTIVATION_SHAPE, device=_state["fabric"].device)
    model = _state["unwrapped_model"]
    model.eval()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / SERVER_WEIGHTS_KEY
        # The legacy TorchScript exporter, as upstream: the dynamo path pulls in onnxscript's torchlib registry.
        torch.onnx.export(
            model,
            example_input,
            str(path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            dynamo=False,
        )
        blob = path.read_bytes()
    storage = _state["storage"]
    storage.put(SERVER_WEIGHTS_KEY, blob)
    _logger.info("stored %d-byte server weights at %s/%s", len(blob), storage.agent_id, SERVER_WEIGHTS_KEY)


def _reply(message_type: MessageType, data: dict[str, Any], raw: dict[str, bytes]) -> bytes:
    return encode_message_b64(WSMessage(type=message_type, data=data, raw=raw))


def _next_batch() -> bytes:
    if _state["batches"] is None:
        _logger.info("initialising MNIST batch stream for browser training")
        _state["batches"] = _mnist_batches(_state["batch_size"])
    data = next(_state["batches"])
    images, labels = data["image"], data["label"]
    return _reply(
        MessageType.BATCH,
        {"images_shape": list(images.shape), "labels_shape": list(labels.shape)},
        {"images": serialize_tensor(images.cpu()), "labels": serialize_tensor(labels.cpu())},
    )


def _train_step(message: WSMessage) -> bytes:
    fabric: L.Fabric = _state["fabric"]
    model = _state["model"]
    optimizer = _state["optimizer"]

    activations = deserialize_tensor(message.raw["tensor"], dtype=torch.float32)
    activations = activations.to(fabric.device).reshape(*message.data["tensor_shape"])
    labels = deserialize_tensor(message.raw["labels"], dtype=torch.int64).to(fabric.device)

    optimizer.zero_grad()
    model.train()
    activations.requires_grad = True
    loss = _state["criterion"](model(activations), labels)
    fabric.backward(loss)
    optimizer.step()

    _state["trained"] = True
    total, count = _state["epoch_loss"]
    _state["epoch_loss"] = (total + loss.item(), count + 1)
    grads = activations.grad
    return _reply(
        MessageType.GRADS,
        {"tensor_shape": list(grads.shape), "loss": loss.item()},
        {"tensor": serialize_tensor(grads.detach().clone().cpu())},
    )


def _infer(message: WSMessage) -> bytes | None:
    shape = tuple(message.data.get("tensor_shape", []))
    # Anything but a post-cut activation means the wrong tensor was dispatched; reject it rather than fail the step.
    if len(shape) != 4 or shape[1:] != ACTIVATION_SHAPE:
        _logger.error("ignoring activations with unexpected shape %s (expected (B, 16, 7, 7))", list(shape))
        return None
    activations = deserialize_tensor(message.raw["tensor"], dtype=torch.float32)
    activations = activations.to(_state["fabric"].device).reshape(*shape)
    model = _state["model"]
    model.eval()
    with torch.no_grad():
        logits = model(activations).detach().clone()
    _logger.info("inference: predicted %s", logits.argmax(dim=-1).tolist())
    return _reply(MessageType.LOGITS, {"tensor_shape": list(logits.shape)}, {"tensor": serialize_tensor(logits.cpu())})
