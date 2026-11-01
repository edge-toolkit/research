# Pretrained workload and recovery experiments

Harbour-FL supports one classification model: frozen AudioSet-pretrained EfficientAT MN10 and ImageNet-1K V2
MobileNetV3-Large encoders, followed by a trainable presence-aware fusion head. The encoders produce separate
960-dimensional embeddings. LayerNorm, missing-input masks and two presence indicators feed a 256-unit ReLU layer,
dropout and the vessel classifier. Clients train the head with class-weighted AdamW and sample-weighted FedAvg.

## Architecture and research basis

Correira, Ferreira and Mišković (2024) provide the boat-detection and classification application reference. Harbour-FL
is an adaptation inspired by that study, extending it to multimodal federated execution and orchestration experiments.
It does not reproduce the paper's YOLO comparison or claim identical datasets, protocols or detection results.

FedMultimodal (Feng et al., 2023) informs the separate modality encoders, feature fusion and evaluation under missing
inputs. MobileNetV3 (Howard et al., 2019) provides the image backbone. EfficientAT (Schmid et al., 2023) provides the
pretrained audio backbone; Harbour-FL uses the published distilled weights without performing teacher distillation.
FedAvg (McMahan et al., 2017) supplies sample-weighted aggregation. The [setup guide](../pretrained.md)
documents the pinned weights, matching waveform/image preprocessing and optional raw-data preparation.

Both encoders stay frozen and in evaluation mode. Dataset preparation caches their embeddings, and all subsequent
federated work uses those features. Only the fusion head, including its LayerNorm parameters, enters client updates
and recovery checkpoints. Checkpoints require the current model and weight identities; incompatible metadata is
rejected.

## Research references

- Correira, A., Ferreira, F., & Mišković, N. (2024).
  [_Comparing Different YOLO Versions for Boat Detection and Classification in Real Datasets_][correira].
  OCEANS 2024 - Singapore.
- Feng, T., et al. (2023). [_FedMultimodal: A Benchmark for Multimodal Federated Learning_][fedmultimodal].
  KDD 2023.
- Howard, A., et al. (2019). [_Searching for MobileNetV3_][mobilenet]. ICCV 2019.
- Schmid, F., Koutini, K., & Widmer, G. (2023).
  [_Efficient Large-scale Audio Tagging via Transformer-to-CNN Knowledge Distillation_][efficientat]. ICASSP 2023.
- McMahan, B., Moore, E., Ramage, D., Hampson, S., & Aguera y Arcas, B. (2017).
  [_Communication-Efficient Learning of Deep Networks from Decentralized Data_][fedavg]. AISTATS, PMLR 54, 1273–1282.

[correira]: https://doi.org/10.1109/OCEANS51537.2024.10682267
[fedmultimodal]: https://arxiv.org/abs/2306.09486
[mobilenet]: https://arxiv.org/abs/1905.02244
[efficientat]: https://arxiv.org/abs/2211.04772
[fedavg]: https://proceedings.mlr.press/v54/mcmahan17a.html

## Optional native calibration

With compatible prepared features linked at `data/prepared`, measure head training on the selected device:

```bash
.venv/bin/python -m harbour_fl.calibrate --device cpu \
  --rounds 2 --output results/calibration/cpu.json
```

Use `--device mps` for native Apple Silicon or `--device cuda` for NVIDIA execution. Explicit unavailable devices fail.
The command defaults to two local epochs, batch size 16 and two CPU threads; keep those settings fixed when comparing
experiments. `HARBOUR_FL_TRAINING_SEED` records the seed, defaulting to 17. `--samples-per-client` and
`--evaluation-samples` can bound a quick check; reports identify limited samples explicitly.

Calibration runs the clients sequentially, then aggregates and evaluates the head on CPU. It records training time,
sample counts, optimizer steps, feature-manifest hashes and serialized head sizes. Encoder extraction happens before
this timing window, so these measurements do not describe CNN execution or training. Full training checkpoints contain
real head tensors and workload metadata; no artificial payload padding is used.

## Deployed recovery experiments

Rebuild the application and regenerate the deployment after implementation changes. The benchmark catalog includes
healthy operation, client outages, coordinator checkpoint restoration, resource pressure and network impairment.
The `representative-*` presets provide longer head-training runs with the same pretrained configuration; they are not
separate model architectures. Scenario deadlines and resource settings are experiment parameters, not measured SLOs.
Calibrate round counts and fault durations for the lightweight head before running pressure or outage experiments.

The benchmark adapter records observed workload settings from all four roles and includes them in comparison identity.
Its stall policy allows the configured round deadline plus a margin, and cleanup restores the manual application
settings. GPU experiments require suitable CUDA allocations. Apple MPS supports native preparation and training;
Kubernetes recovery experiments run in Linux containers.

`round_work_seconds` covers publication, contribution collection, head aggregation and evaluation for one attempt.
Checkpoint commit and recovery are recorded separately. Repeated attempts have distinct timings; useful progress counts
each committed logical round once. Transfer durations describe application-visible storage operations on head tensors,
not maximum network capacity or full-encoder update traffic.

Reports and datasets remain ignored and are generated only when experiments are run. Preserve failed or unknown
observations and record the dataset, cached-input conditions, hardware and resource settings when sharing results.
