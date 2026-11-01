# Pretrained vessel classification

Harbour-FL adapts the boat-classification application of Correira, Ferreira and Mišković (2024) to multimodal
federated learning. Its foundations are [FedMultimodal, MobileNetV3, EfficientAT and FedAvg][research]. It does not
reproduce the paper's YOLO comparison or distribute experimental accuracy or infrastructure results.

## Model and training

- Audio uses the official AudioSet-pretrained EfficientAT MN10 weights and a pinned upstream implementation.
- Images use the official ImageNet-1K V2 MobileNetV3-Large weights.
- Both encoders remain frozen and in evaluation mode. Preparation caches their 960-dimensional embeddings per sample.
- The trainable fusion head applies separate LayerNorm layers, masks unavailable embeddings, and concatenates both
  embeddings with two modality-presence indicators. A 256-unit ReLU layer, dropout and a new vessel classifier follow.
- Clients train this head with AdamW, resetting optimizer state each round. Cross-entropy uses inverse-frequency
  weights computed from the global training partition. The coordinator applies sample-weighted FedAvg to head updates.

The new classifier learns the dataset's mapped vessel labels. The initial weights are general audio/image encoders,
not vessel-specific classifiers or weights from Correira's study. The default deployment selects
`HARBOUR_FL_MODEL_PROFILE=pretrained`. This is the only supported configuration; the encoders remain frozen during
all federated training. Checkpoints must carry this configuration and its pretrained weight identities.

Feature extraction runs before the federated job. Training, evaluation, checkpoint restoration and network transfers
therefore concern the fusion head, not repeated encoder execution. The frozen weights are unnecessary inside training
containers after preparation. These timings must not be presented as full audio/image encoder-training measurements.

## Verified pretrained artifacts

From the project directory, after configuring the temporary Core release described in the [README](README.md):

```sh
mise install
mise run setup:python
mise run setup:weights
```

The setup task downloads artifacts to the ignored `models/pretrained/` directory:

```text
models/pretrained/
  mn10_as.pt
  mobilenet_v3_large.pth
  efficientat/
  EfficientAT-LICENSE
  assets.json
```

The application records the exact upstream source revision and SHA-256 hashes in
`src/harbour_fl/pretrained_assets.json`. Downloads and subsequent loads are verified against those committed identities.
The pinned EfficientAT source is namespaced with package-relative imports; its license is retained. Models load with
`weights_only=True` and strict parameter matching. There is no random-initialization fallback for missing weights.

For an existing artifact directory, import the same verified files without downloading:

```sh
.venv/bin/python -m harbour_fl.pretrained_assets --source /absolute/path/to/pretrained-artifacts
```

`HARBOUR_FL_PRETRAINED_ASSETS` can select another local artifact directory. Runtime training loads the cached features;
it does not download weights. Weight binaries and downloaded upstream source are not committed to Git.

## Use a prepared dataset

Supply a compatible prepared directory through `HARBOUR_FL_SOURCE_DATA` in `.env.local`, then run `mise run setup`
as described in the [README](README.md). No prepared dataset is distributed in this repository.

The directory must contain `preparation.json`, `feature_manifest.csv`, `class_summary.csv`, feature archives and
the three client partitions under `clients/`. Features contain the frozen encoders' 960-dimensional outputs for
the supported clean and noisy conditions. Preparation identities and file hashes are checked before use; a directory
of arbitrary embeddings or log-mel arrays is insufficient. Original WAV files and images are not needed at runtime.

The following sections describe optional preparation from raw inputs when a compatible dataset is not already available.

## Prepare paired raw inputs

Supply a directory containing `segment_manifest.csv` and the raw Croatia WAV/image pairs. Mixed manifests are filtered
by `source_dataset=Croatia` before label mapping. Manifests without a source column are treated as Croatia-only;
that assumption is recorded. Audio/image pairs come from explicit manifest paths; no pairing is inferred from filenames.

```sh
.venv/bin/python -m harbour_fl.prepare \
  --dataset /absolute/path/to/source-manifests \
  --raw-root /absolute/path/to/raw-recordings \
  --output /absolute/path/to/prepared-dataset \
  --device auto
```

Preparation reads the requested WAV segments, converts PCM to mono and resamples to 32 kHz. EfficientAT receives a
128-bin Kaldi-style log-mel frontend with a 1024-point FFT, 800-sample window and 320-sample hop. Images are resized to
232 pixels on the shorter side, center-cropped to 224 × 224 and normalized using ImageNet statistics. Standardized
64-bin log-mel arrays cannot replace the required waveform preprocessing for these pretrained weights.

CPU, CUDA and native Apple MPS extraction are supported. Explicit unavailable devices fail. On MPS, STFT runs on CPU
and the CNN encoders run on MPS. Fixed seeds are used; forced deterministic algorithms are disabled for MPS extraction.
Runtime identities and source hashes are recorded, and changed settings require a new prepared output directory.

The optional YOLO preprocessing detector is not used by this workflow; keep `HARBOUR_FL_DETECTOR_ENABLED=false`.
The separate detector utilities can inspect image crops independently of classification training.

### Labels and partitions

Source categories map to passenger vessels, pleasure craft, sailing vessels, workboats and fishing vessels. A class
requires at least three independent recording/content groups, allowing training, validation and test representation.
Excluded classes, duplicate segments and missing pairs are recorded explicitly. No class is guaranteed inclusion.

Recording/event identities and identical audio/image contents are grouped before splitting. This prevents those groups
from crossing training, validation and test partitions; it does not establish separation by vessel identity or date.
Training groups are assigned to three logical clients. All clients use the same held-out validation partition, while
the coordinator evaluates a separate test partition. Validation and test samples never enter client fitting. If a
client receives no training samples, preparation fails rather than inventing or duplicating training data.

Prepared output contains portable feature archives, class information and three client manifests. Raw files remain
unchanged. The `source/` subdirectory retains import provenance; runtime partitions refer only to project-relative
feature archives, so the prepared directory can be mounted inside containers without the original machine's paths.

Set these values in the ignored `.env.local`, then generate the deployment:

```dotenv
EDGE_TOOLKIT_RELEASE_DIR=/absolute/path/to/core-release
HARBOUR_FL_SOURCE_DATA=/absolute/path/to/prepared-dataset
```

```sh
mise run setup
```

This links the prepared directory at `data/prepared`. Native and k3d execution follow the [README](README.md).
Use a fresh job identity when changing preparation settings. Checkpoints record the pretrained identities and head
architecture; incomplete or incompatible workload metadata cannot resume training.

## Noise and missing inputs

Preparation also caches embeddings from deterministic waveform Gaussian noise at 20/10/0 dB SNR and image Gaussian
noise with standard deviation 0.05/0.10/0.20 on the zero-to-one pixel scale. Corruption occurs before feature
extraction, not by adding noise to the embeddings. Simultaneous audio/image corruption is not prepared.

With `HARBOUR_FL_ADAPTATION=true`, training samples from clean, 10 dB audio-noise and 0.10 image-noise features and
withholds one modality from 30% of training samples. At evaluation, unavailable modalities are explicitly masked after
normalization and their presence flags are cleared. This is augmentation during training, not online retraining or
an automatic claim of improved performance. `HARBOUR_FL_AUDIO_SNR_DB` selects supported waveform noise.
Nonzero `HARBOUR_FL_AUDIO_NOISE_STD` values are rejected to prevent mixing noise definitions.

## Tests and optional experiments

```sh
mise run test
```

Unit tests use temporary fixtures and require no downloaded weights. A native head-training check can use the prepared
partitions and retain its report separately:

```sh
.venv/bin/python -m harbour_fl.calibrate --device cpu \
  --rounds 1 --output results/pretrained-check.json
```

The calibration executes clients sequentially on one host. Use a deployed scenario to investigate transport or recovery.
Benchmark code remains available, but the repository contains no historical result reports or model-quality claims.
New experiment outputs, MLflow data, prepared datasets and trained checkpoints stay ignored and can be shared separately
when needed. Do not combine results from different feature-extraction conditions without identifying them.

[research]: benchmarking/workload.md#research-references
