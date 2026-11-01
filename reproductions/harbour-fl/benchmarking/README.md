# Harbour-FL benchmark adapter

This directory supplies the Harbour-FL binding for benchmarking against Core recovery supervision. `adapter.py` accepts
a validated request, writes an effective scenario, and executes the small `benchmark.py` entry point. `integration.py`
supplies deployment identities, FL setup/cleanup, data faults, and application progress parsing. `harbour_profile.py`
owns round/MLflow reconciliation, participation and input-I/O metrics, and the mapping to generic work units.
Infrastructure faults, resource sampling, Core recovery-report collection, SLO evaluation, and reports live in the
`experiments/reliability-benchmarks/controllers` package. Engine instances receive adapters directly, without a global
binding.

`scenarios.json` owns scenario descriptions, settings, and defaults. `../benchmark-catalog.json` references it and
defines permitted UI parameters; the browser uses scenario-specific defaults. Run `mise run benchmark-ui` from the
project directory after deploying the k3d scenario.

## Comparable experiment runs

All scenarios use frozen pretrained encoders and federated fusion-head training. See the
[pretrained workload guide](workload.md) for input preparation, calibration and measurement boundaries.

From the application directory, run a repeated, paired checkpoint experiment with:

```bash
mise run setup:benchmark
.venv/bin/python benchmarking/benchmark.py checkpoint-recovery-stale --repetitions 5 --paired-baseline --seed 17
```

The UI provides the same pairing option and increments the timing seed across its repetitions. The training seed remains
fixed unless a scenario changes `training_seed`. Reports capture manifest hashes, source version, training
settings, checkpoint cadence, resources, image identities and node metadata. Review `reference.missing` before
using a paired comparison. Manifest hashes identify the prepared partitions; keep the underlying data immutable as well.

Checkpoint scenarios compare saving every round with saving every three rounds. The pre-commit variant adds a recorded
five-second delay before committing checkpoint bytes. Its trigger is observation-based, so inspect the recorded boundary
and actual injection time when assessing whether the intended window was reached. Real process crash/resume tests verify
that stale checkpoints can repeat work and still produce the same final weights as uninterrupted execution.

Rebuild and redeploy the application image after updating the training instrumentation. Core's recovery implementation
is unchanged. The soak presets run 60 rounds and the `representative-*` presets run 20; elapsed duration depends on
hardware and data volume.

The [experiment guide](../../../experiments/reliability-benchmarks/controllers/experiments.md) defines the metrics,
confidence intervals, comparison groups, service continuity and participant return.

## Scenario defaults

These are the defaults from `scenarios.json`; permitted UI overrides can change their parameters.
Unless noted otherwise, scenarios default to observe-only recovery.

| Scenario                         | Description                                                                |
| -------------------------------- | -------------------------------------------------------------------------- |
| `baseline`                       | Three-client reference run without injected faults.                        |
| `reference-training-outage`      | Continue with two clients, then verify fresh participation from vessel-3.  |
| `reference-client-stage-restart` | Seeded training/update-transfer observation before a client restart.       |
| `checkpoint-recovery-recent`     | Restore a recent checkpoint and verify unfinished processing resumes.      |
| `checkpoint-recovery-stale`      | Save every three rounds; measure retained and repeated work after restart. |
| `checkpoint-recovery-precommit`  | Interrupt a bounded pre-commit delay and restore a complete checkpoint.    |
| `soak-baseline`                  | Sixty-round healthy reference for longer-duration measurements.            |
| `soak-client-outage`             | Sixty rounds with a one-minute outage and verified later participation.    |
| `combined-client-network-outage` | Concurrent outage on one client and degraded connectivity on another.      |
| `client-memory-saturation`       | Ramp client RAM occupancy, release it, then require useful progress.       |
| `client-network-saturation`      | Step competing client traffic across a controlled egress bottleneck.       |
| `gpu-baseline`                   | CUDA training baseline on GPU-capable Kubernetes workers.                  |
| `gpu-runtime-failure`            | One-shot simulated GPU runtime error; Core restarts the client.            |
| `gpu-out-of-memory`              | Simulated GPU OOM exception without excessive memory allocation.           |
| `client-restart`                 | Delete vessel-2's pod; verify reconnection and recovery.                   |
| `client-dropout`                 | Remove vessel-3 for 15 seconds, then restore it.                           |
| `cpu-saturation`                 | Run four CPU burners on the node for 60 seconds.                           |
| `memory-pressure`                | Allocate and touch 512 MiB of node memory for 30 seconds.                  |
| `client-oom`                     | Limit vessel-2 to 96 MiB; detect OOM, restore, assess recovery.            |
| `dataset-read-latency`           | Delay vessel-2 feature-archive reads by 25 ms; measure throughput.         |
| `audio-noise`                    | Add deterministic noise to vessel-2 audio; keep images clean.              |
| `image-noise`                    | Add deterministic noise to vessel-2 images; keep audio clean.              |
| `missing-audio`                  | Zero-mask vessel-2 audio while retaining images.                           |
| `missing-image`                  | Zero-mask vessel-2 images while retaining audio.                           |
| `checkpoint-write-failure`       | Reject a round-1 checkpoint; verify safe failure, recovery disabled.       |
| `mlflow-storage-outage`          | Stop MLflow for 15 seconds; verify ingestion without bridge restart.       |
| `network-latency`                | Add 250 ms delay and 50 ms jitter to vessel-2 for 30 seconds.              |
| `network-bandwidth`              | Limit vessel-2 egress to 256 kbit/s for 30 seconds.                        |
| `network-loss`                   | Apply 10% packet loss to vessel-2 for 30 seconds.                          |
| `network-partition`              | Block all vessel-3 egress for 15 seconds.                                  |
| `degraded-network`               | Combine delay, jitter, loss and a bandwidth cap for 30 seconds.            |
| `ws-server-restart`              | Restart the hub; Core supervision disabled for this experiment.            |
| `coordinator-restart`            | Restart the coordinator; verify checkpoint restoration and progress.       |
| `adaptive-missing-audio`         | Train with augmentation and assess unavailable audio.                      |
| `adaptive-audio-noise`           | Train with augmentation and assess waveform audio noise.                   |
| `adaptive-missing-image`         | Train with augmentation and assess unavailable images.                     |
| `adaptive-image-noise`           | Train with augmentation and assess image noise.                            |
| `recovery-slo`                   | Check coordinator restart/resume against illustrative recovery SLOs.       |
| `invalid-client-update`          | Reject nonfinite vessel-2 updates; allow a two-client quorum.              |
| `quorum-client-outage`           | Allow a two-client quorum while vessel-3 is absent for 25 seconds.         |
| `recovery-slo-assisted`          | Recovery-SLO comparison permitting bounded controller assistance.          |
| `mlflow-storage-outage-assisted` | Restore MLflow and explicitly restart its bridge to aid ingestion.         |

Augmented scenarios train the fusion head with cached clean/noisy features and modality dropout. Compare their
held-out metrics with a separately trained clean-input configuration; they do not enable an inference-time gate.

## Client resource saturation

From the Harbour-FL project directory, on the prepared Kubernetes deployment:

```bash
.venv/bin/python benchmarking/benchmark.py client-memory-saturation
.venv/bin/python benchmarking/benchmark.py client-network-saturation
```

Both default to 12 rounds; calibrate the round count so training continues after the pressure profile.
The memory scenario sets
vessel-2's limit to 768 MiB, aims at 60%, 75% and 90% occupancy, and adds at most 512 MiB while retaining
headroom. The reference limit is restored after measurement. Tune the envelope if the client's normal PyTorch footprint
already approaches it. This is bounded near-limit RAM occupancy, not deliberate OOM or a claim of memory-bandwidth
saturation; `client-oom` remains the separate termination test.

The network scenario uses a 1024 kbit/s client-egress cap and offers 25%, 50%, 85%, 125% and 150% of that rate to a
temporary private receiver. It measures congestion on that controlled virtual/container link, not your physical LAN's
maximum capacity. Profiles that fail to demonstrate the requested pressure, or finish training before post-release
progress can be verified, do not pass. Reports include per-stage completions and diagnostic latency/throughput
measurements; the existing MLflow annotation records compact pressure summaries and return-to-progress time.

The hub-restart scenario explicitly disables Core supervision to assess application/Kubernetes communication recovery
without a Core recovery claim. This scenario setting is separate from the supervisor's configurable reconnection
support. In supervised scenarios, supervisor replacement requires verifiable durable context continuity; a restart
alone must not reset the observation clock or satisfy the recovery gate.

## Recovery and interpretation

The coordinator-restart scenario requires recovery from a committed round checkpoint. The checkpoint-write-failure
scenario deliberately verifies safe failure and disables automatic restart; a passing expected-failure experiment is
not a successful recovery. The `recovery-slo` scenario supplies illustrative recovery thresholds.

Standard scenarios use `observe-only` recovery: Kubernetes and application/Core mechanisms are observed without
additional supervisor restarts. Select `controller-assisted` to permit bounded Edge Toolkit Core intervention.
`recovery-slo-assisted` provides that mode as a named comparison; it does not force a restart if native recovery works.
Expected checkpoint-write failures fix the mode to `disabled` so they cannot be silently rescued.

`mlflow-storage-outage` restores the service and waits for post-fault training metrics and run completion without
restarting the bridge. `mlflow-storage-outage-assisted` explicitly restarts the bridge and labels that assistance.
Both measure the combined system, not Core-only reliability. The tracking gate verifies all expected round-loss
metrics, not every artifact or the bridge's internal queue length.

The browser's attribution panel, JSON reports, `recovery-summary.csv`, and MLflow tags identify the mode, observed
recovery actors, evidence level, and intervention. Fault injection/removal is separate from recovery assistance.
Missing actor evidence stays unattributed. Compare fresh runs within the same mode; older reports are not relabelled.

Coordinator failure logs take precedence over delayed MLflow status. Expected checkpoint-write failures end the
experiment promptly, require the injected failure message, and retain the failure in the report even if MLflow still
shows the run as running. Recovery-enabled scenarios instead keep observing until resumed progress or the experiment
timeout; a Ready pod alone does not establish recovery.

Scenarios requiring recovery keep observing after training finishes, until the affected workload has recovered or
its recovery deadline expires. Coordinator restart tests also require post-fault checkpoint restoration and resume
events. Recovery uses progress from the current pod rather than cached events from its predecessor. If training
finishes during graceful termination, the replacement must restore and republish the completed result; this measures
completed-workload recovery, not continuation of unfinished training. Use more rounds to exercise the latter case.

`quorum-client-outage` allows two-client aggregation after a bounded round wait. `invalid-client-update` injects
nonfinite model parameters, verifies their rejection through recorded events, and permits a two-client quorum.
Validation covers parameter names, shapes, dtypes, finite values, sample counts, labels/rounds, and a configurable
per-parameter delta norm bound. This is limited invalid-update handling, not authenticated identity or Byzantine
robustness against all finite poisoned updates.

The `adaptive-*` scenarios enable noise augmentation and modality dropout during head training. Presence indicators
mask unavailable embeddings after normalization. Evaluation does not retrain the model or toggle a separate gate;
performance changes require comparing trained configurations under equivalent held-out input conditions. Both frozen
encoders run only during preparation, and missing-input experiments mask their cached embeddings.

Baseline data faults affect one client's fitting/validation. Adaptive scenarios also apply the perturbation to the
coordinator's held-out evaluation inputs so classification performance can be measured. Test labels are used only for
evaluation. Noise is deterministic and the source dataset is not modified.

## Application and infrastructure boundary

The generic engine reports work units and infrastructure evidence. This adapter defines one work unit as one completed
federated round and retains the existing `training` JSON block and round-named SLIs for Harbour-FL reports and
scenarios.
FL metrics are supplied through `application_metrics`; the shared MLflow writer no longer interprets client logs.
The scenario name `client-oom` remains unchanged, while its adapter translates the fault to generic `worker_oom`.
Likewise, scenario `after_round` settings become `after_work_unit` before the shared engine executes them.

For direct command-line benchmarks, run `mise run setup:benchmark` in the project directory first. Model runtime
setup (`setup:python`) does not install the benchmark controller. The browser shortcut installs adapter dependencies
automatically; training containers do not contain the benchmark controller.

## Functional availability and communication checks

Enable **Measure functional availability and communication RTT** in the benchmark UI, or add this object to a scenario:

```json
"service_probes": {
  "enabled": true,
  "interval_seconds": 2,
  "timeout_seconds": 5,
  "rtt_payload_bytes": 32
}
```

Rebuild and deploy the updated Harbour-FL image first. The benchmark checks image support and restores probe environment
settings during cleanup. Checks are disabled by default. Healthy and interrupted comparisons must use the same settings.

An independent observer sends a bounded arithmetic task through the coordinator to each client and verifies the returned
result. It continues during coordinator and client replacement. The required client count defaults to `minimum_clients`
(or three); `required_workers` can explicitly set a different quorum. These checks assess application message handling,
task dispatch and result collection. They do not establish training, inference, checkpoint storage or model-quality
availability. Successful checks do not advance training progress or satisfy Core recovery evidence.

`service_probes` in each job report contains per-client success counts, response latency, and a fixed-interval estimate
of
quorum availability. Each interval is classified by the checks issued for that interval, with the final interval
truncated.
A response must arrive within its timeout and contain the expected result. Missing observer data remains unknown;
affected
availability estimates provide lower/upper bounds instead of a single value. The eligible window starts after observer
registration and ends before log collection and cleanup; startup before registration is outside this window. Functional
response latency includes the host observer's Kubernetes port-forward path. Raw check outcomes are retained in the
linked
`*-service-probes.json` file. Pending checks can add up to their response deadline during finalization.

Each client also sends correlated WebSocket Ping/Pong probes directly to the communication hub. `communication_probes`
reports observed attempts, failures, timeouts, payload size and successful RTT distributions separately by client.
Connection setup is excluded from RTT, although connection failures count as unsuccessful attempts. There is one pending
RTT probe per client, so timeouts can extend the configured spacing. Client downtime or missing diagnostic logs are
unobserved by this client-side probe and cannot establish continuous service availability. p95 is withheld below 20
successful samples and p99 below 100; these minima do not establish statistical confidence. Neither measurement
represents
vessel radio-network propagation delay alone.

## Transfer and final-model evaluation measurements

Client model downloads and update uploads emit correlated start/end records, including payload size, SDK operation
duration and successful, failed or timed-out outcomes. `transfer_measurements` reports per-client durations, effective
MiB/s and observed attempt counts. Serialization is excluded. Missing end records remain unknown and orphan end records
are retained; incomplete populations do not receive a success ratio. SDK storage timing does not establish maximum
network capacity, and the instrumentation does not add retries or replace the SDK's own timeout behavior.

Set `evaluate_degraded_inputs: true` in a scenario to evaluate the final model on clean, noisy and missing-modality
held-out inputs without additional training. This also supports models trained with augmentation. Clean reference
inputs are evaluated explicitly, independently of any coordinator input fault.
Prediction-input conditions are passed explicitly and do not inherit client training faults. Results are retained in
`prediction_input_evaluations`, the saved evaluation summary and MLflow metrics. The adapter requires the primary
application's final completion event before cleanup so these evaluations cannot be cut short by the last round commit.

## Recovery supervision measurements

The recovery configuration includes dependency checks, coordinated actions and bounded backoff. Dependencies receive
read-only deployment access; only the configured Harbour-FL workloads can be restarted. Regenerate the deployment with
a matching Core version when changing this configuration.

Experiments can set `recovery_resilience.stability_seconds` to a positive observation window and
`require_recovery_stability: true` together with `require_recovery: true`. The controller waits for Core's separate
stability assessment before cleanup. MTTR still ends at first useful recovery. Pending, incomplete, unstable, disabled
and unreported assessments are distinct; older reports do not establish stability. Matched healthy runs retain the
same resilience settings but clear the fault-specific success gates.

Reports retain observation gaps, the latest failure classification and action decision, and the observer clock basis.
Supervisor replacement is accepted only with verifiable durable context continuity, including preserved incident
history and consumed budgets. Restored clocks and different resilience settings form separate comparison groups.
