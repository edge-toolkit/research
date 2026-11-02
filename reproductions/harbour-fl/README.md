# Harbour-FL

Harbour-FL is a research project built on Edge Toolkit. It demonstrates multimodal federated learning with three
logical vessel clients, a coordinator, MLflow experiment tracking, and operator-side reliability benchmarks.

Potential applications include classifying commercial and fishing vessels alongside recreational boats for harbour
traffic monitoring, coastal observation and fishing-fleet studies. The federated approach could support learning
across monitoring sites while keeping recordings local, and the noise and missing-input experiments explore
performance when sensor conditions vary. These applications require suitable labelled data and validation for the
intended setting.

The application is an adaptation inspired by [Correira, Ferreira and Mišković (2024)][correira],
_Comparing Different YOLO Versions for Boat Detection and Classification in Real Datasets_.
That study provides the boat-detection and classification reference. Harbour-FL extends the application to audio/image
fusion and federated learning to assess orchestration and recovery; it does not reproduce the paper's YOLO comparison
or claim equivalent datasets, experimental conditions or detection results. Its supporting model and learning methods
are described in the
[architecture and research basis](benchmarking/workload.md#architecture-and-research-basis).

[correira]: https://doi.org/10.1109/OCEANS51537.2024.10682267

The default application trains a fusion head on paired audio/image embeddings from frozen pretrained encoders.
Clients exchange head updates over the Edge Toolkit WebSocket hub; input samples and cached features stay local.
The coordinator applies Federated Averaging and
broadcasts the next global model. The engine emits tracking events and artifacts; the generic Core MLflow bridge
records them without making training depend on MLflow server availability.

Standalone YOLOv8 detector utilities are available for inspecting camera crops. They do not participate in the
pretrained classification workflow, which uses complete images.
See [camera detection](models/vessel-detector/README.md) for model placement, configuration and standalone previews.

The [pretrained workflow](pretrained.md) uses AudioSet-trained EfficientAT MN10 and ImageNet-trained MobileNetV3-Large,
with a presence-aware fusion head and sample-weighted FedAvg. Setup downloads verified encoder artifacts into an ignored
model directory. This is the only supported model configuration. Optional
[recovery experiments](benchmarking/workload.md) exercise the same pretrained fusion head. The project includes research
references, setup instructions and tests; experimental results are generated separately and are not distributed here.

## Layout

- `src/harbour_fl/`: installed application package: data, learning, checkpoints, roles, protocol, and instrumentation.
- `modules/`: project-owned Edge Toolkit workflow modules for the client, coordinator, MLflow link, and FL engine.
- `scenario.yaml`: the six-agent Edge Toolkit scenario (three clients, coordinator, bridge, and browser modules).
- `benchmarking/`: Harbour-FL adapter, data faults, metric mapping, and authoritative experiment scenarios.
- `models/vessel-detector/`: frozen camera-detector configuration and locally supplied ONNX weights.
- `experiments/scheduling/`: optional placement experiments, separate from operational recovery.
- `recovery.json`: application identities, mode and policy settings supplied to Core recovery supervision.
- `benchmark-catalog.json`: the safe, browser-visible catalog consumed by the generic benchmark controller.
- `configure_deployment.py`: thin deployment CLI; `deployment_support/` owns release staging, credentials, scenario
  settings, Kubernetes adaptation, and the local MLflow service.

Generated deployments, virtual environments, datasets, and results are intentionally ignored.

## Temporary local Core release

The current setup uses a temporary local Core release bundle. Configure its location and a compatible prepared
dataset in an ignored `.env.local`:

```dotenv
EDGE_TOOLKIT_RELEASE_DIR=/absolute/path/to/core-release
HARBOUR_FL_SOURCE_DATA=/absolute/path/to/prepared-dataset
```

The release must provide these matching runtime artifacts:

```text
bin/et-cli
python/edge_toolkit-*.whl
python/et_ws-*.whl
python/<any additional SDK dependency wheels>
images/core-runtime.tar
modules/@edge-toolkit/et-ws-pyo3-mlflow/
modules/@edge-toolkit/et-ws-server-static/
modules/@edge-toolkit/et-ws-wasm-agent/
```

`core-runtime.tar` supplies `et-ws-server-core:local`, `et-ws-pyo3-runner-core:local`, and
`et-recovery-supervisor:local`. The CLI and Python wheels must include Core's recovery interfaces. No Core source
checkout is used by setup. Setup validates that the required artifacts are present before staging them.

`setup:mlflow-bridge` validates the release before staging modules and Python wheels into ignored `.local-modules/` and
`.local-core-python/`. `setup:python` installs the application package and wheels and verifies recovery imports.
`setup:core-images` loads
the matching server, runner and supervisor images. These are local-release adapters, not copies of Core recovery logic.

## CPU, Apple MPS and NVIDIA CUDA

`HARBOUR_FL_DEVICE=cpu` is the default. Set `HARBOUR_FL_DEVICE=mps` for native Apple Silicon Python execution,
`cuda` for NVIDIA execution, or `auto` to select CUDA, then MPS, then CPU at startup. Explicit GPU requests fail if
unavailable; there is no runtime CPU fallback. Configure a client's environment in `scenario.yaml` to use a different
device from the coordinator. Updates and checkpoints are serialized on CPU; numerical results need not be identical
across different accelerator backends. A failed client retries the unfinished round from the coordinator's published
global model after Core restarts it; this does not provide mid-batch or optimizer-state resume.

For a native GPU smoke test without a dataset, run:

```bash
.venv/bin/python -m unittest discover -s modules/harbour-fl-engine -p test_devices.py -v
```

The GPU test exercises every available CUDA/MPS backend and explicitly skips when neither is exposed. MPS requires
native macOS execution, not the Linux containers in Docker/k3d. The Kubernetes benchmark adapter rejects MPS rather
than silently exercising CPU. Native MPS injection can use the cooperative request protocol, but the current benchmark
engine and Core recovery backend still require Kubernetes; native end-to-end recovery is not claimed. To enable
native injection, set `ET_BENCHMARK_GPU_FAULTS=1` only on the selected client's scenario environment and configure
`HARBOUR_FL_DEVICE=mps`. While it is training, run this with its actual `HARBOUR_FL_JOB_ID`:

```bash
mise run setup:benchmark
.venv/bin/python -m et_experiments.benchmarking.gpu --job-id harbour-fl-demo --mode runtime_error
```

The failed native worker requires a process restart; do not expect the Kubernetes supervisor to restart a native runner.

For CUDA containers, install a PyTorch build matching the NVIDIA platform: set `HARBOUR_FL_TORCH_INDEX_URL` to the
appropriate CUDA-enabled wheel index before `mise run k3d-build`. The default image deliberately remains CPU-only.
Set `HARBOUR_FL_K3S_GPUS_PER_CLIENT=1` and `HARBOUR_FL_K3S_GPU_CLIENTS=vessel-2` before deployment generation to request
`nvidia.com/gpu` only for that client. The GPU benchmark presets use CUDA on vessel-2, with the other clients and
coordinator on CPU, so one schedulable GPU is sufficient. Other scenarios may override the default `device` with a
`client_devices` mapping. Omitting `HARBOUR_FL_K3S_GPU_CLIENTS` allocates GPUs to all three clients instead. This
requires a configured NVIDIA container runtime/device plugin and sufficient schedulable GPU allocations; one physical
GPU cannot satisfy three exclusive allocations without explicitly configured sharing. A normal macOS k3d cluster
does not provide NVIDIA GPUs. These changes do not install drivers or configure GPU sharing.

On a GPU-capable deployment, run these from the project directory:

```bash
mise run setup:benchmark
.venv/bin/python benchmarking/benchmark.py gpu-baseline
.venv/bin/python benchmarking/benchmark.py gpu-runtime-failure
.venv/bin/python benchmarking/benchmark.py gpu-out-of-memory
```

The fault scenarios are opt-in application-boundary simulations on real GPU workers, not driver resets, deliberate
memory exhaustion, or physical GPU damage. They stop the failed worker's training loop, report failure to Core, and
wait within the existing round deadline for a replacement. Completion still requires useful resumed progress.

## Local mise deployment

With `.env.local` configured as above, run from this directory:

```bash
mise install
mise run setup
```

Setup links the supplied prepared dataset at `data/prepared`, installs the runtime, verifies encoder artifacts and
generates the deployment. Prepared data must follow the [feature format](pretrained.md#use-a-prepared-dataset);
arbitrary log-mel arrays are not compatible. Original recordings and images are unnecessary for training with prepared
features. To build those features yourself, follow the optional
[raw-input preparation](pretrained.md#prepare-paired-raw-inputs) instructions before running setup.

Start MLflow in one terminal:

```bash
mise run mlflow
```

Start Edge Toolkit in another:

```bash
mise run server
```

Open the HTTPS address printed by the server, accept the local development certificate if prompted, and select the
three clients and coordinator in separate browser sessions. MLflow is available only on
`http://127.0.0.1:5000`.

After edits to `scenario.yaml`, module metadata, or deployment configuration, regenerate with:

```bash
mise run generate
```

Ordinary Python and JavaScript module edits are read by the generated mise deployment and do not require an image
rebuild. This host deployment is unsupervised: automatic Core supervisor deployment is currently k3s-only. It still uses
Core's checkpoint helper. The reliability benchmarks target the k3d deployment below.

## Local k3d deployment

Docker must be running and `.env.local` must point at the dataset and temporary Core release. The complete happy path
is:

```bash
mise run setup:data
mise run k3d-up
mise run k3d-status
```

`k3d-up` creates or updates the namespace and Kubernetes Secret automatically. The first generation retains its
credentials in ignored `deployment-secrets/secrets.env`; subsequent mise/k3s generations reuse them so scenario edits
do not change the password of a service with an existing database. Keep that file with the deployment's persistent
data. Updating it is not a database password rotation: the database credentials must be changed separately as well.
`Dockerfile.hub` packages the released static frontend, browser agent, workflow modules, and browser dependencies for
this external scenario without requiring a Core source checkout. A successful rollout is required before benchmarking.
Core's scenario generator adds the recovery supervisor, scoped permissions and persistent state automatically. Recovery
is configured once in `recovery.json`, then combined with `scenario.yaml` for k3s generation. Ordinary mise generation
does not receive a recovery block. Change the manual job identity in `recovery.json` for a fresh supervised demo run.

The Edge Toolkit UI is published on port `8443`. Forward MLflow in a separate terminal:

```bash
mise run k3d-mlflow
```

That exposes MLflow at `http://127.0.0.1:5000`. Stop the cluster without deleting its state using:

```bash
mise run k3d-stop
```

Application package, module, or Python-dependency changes require `mise run k3d-up` so the local images are rebuilt
and imported. Dataset
contents are mounted read-only into the k3d server rather than copied into images.

## Benchmarks

Deploy the k3d scenario first. `setup:python` installs only application dependencies; `setup:benchmark` adds the
operator-side controller package for the adapter. This shortcut runs both setup tasks and starts the standalone
benchmark UI with this project's catalog:

```bash
mise run benchmark-ui
```

Open `http://127.0.0.1:8765`, select an allowlisted scenario and bounded parameters, and start the run. The generic
controller owns the UI, validation, progress, cancellation, and process supervision. The project adapter owns
Harbour-FL-specific behavior; the shared package supplies Kubernetes faults and measurements. Browser input cannot
provide shell commands, paths, Kubernetes objects, or environment variables.

UI-launched benchmark outputs are written under `experiments/reliability-benchmarks/results/` at the repository root,
not inside the application. For direct benchmark CLI runs, run `mise run setup:benchmark` first; their outputs default
to local
`results/benchmarks/`. Experimental outputs are created only when those commands run and remain outside Git.
Benchmark summaries are attached to the matching MLflow run when reachable.

The browser supports sequential repetitions and displays recovery/SLO results and attribution. Benchmarks default to
observe-only recovery; controller-assisted mode permits bounded Core-supervisor restarts and records intervention.
Core verifies recovered FL clients through acknowledged accepted updates, and coordinator recovery through new committed
round progress. Loading or republishing an already completed checkpoint alone does not satisfy unfinished-work recovery.
See `benchmarking/README.md`
for the scenarios and [shared controllers](../../experiments/reliability-benchmarks/controllers/README.md) for
architecture and metric limits.

For k3d runs, Core supervision starts with the scenario. `mise run recovery` shows its logs; it does not
start a second controller. A coordinator
restart restores the latest committed round, model, CPU RNG state, and metric history from hub storage. Local AdamW
optimizers are intentionally recreated per round. Interrupted local work may be repeated; this is round-boundary
resume rather than mid-batch recovery. Keep job IDs and configured training settings stable across a recovery; use a
new job ID to start a fresh experiment. Persistent hub storage and stable agent identities are required.
Checkpoint keys use Core's pointer format and a bounded hash of the job identity.

The installed application imports Core's recovery SDK directly, not `et_experiments`. The training image copies only
`src/harbour_fl/`, not the benchmark controller or its fault injection/UI code. Its small instrumentation layer maps
application events to Core lifecycle messages and emits diagnostic logs consumed externally by the benchmark adapter.
The ws-module entry point keeps the same runner hooks and immutable client/coordinator roles. The host deployment
adds `src/` explicitly to `PYO3_PYTHONPATH`, since the runner does not process editable-install path files itself.

The normal three-client mode waits for all clients. Only scenarios explicitly configuring a smaller quorum continue
without a missing or rejected participant. Restart policies are bounded; a persistent fault is reported as unrecovered.
Deployments must be regenerated and the k3d images rebuilt after changing runner-side package code. Host-side
benchmark changes do not require rebuilding images. Restart the benchmark UI after changing its code or catalog.

## Checks

```bash
mise run test
```

Repository-wide formatting and static checks run from the research repository root with `mise run check`.
