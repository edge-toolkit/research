# Core recovery integration and experimental measurements

Edge Toolkit Core owns operational recovery: lifecycle validation, failure/stall detection, grace periods, restart
budgets, recovery deadlines and useful-progress verification. Kubernetes performs the requested Deployment restarts.
The benchmark controller handles fault injection, experiment setup, diagnostic evidence, repetition statistics and SLO
evaluation.
The application owns serialization, safe checkpoint boundaries, participation rules and in-memory state reconstruction.

## Deployment and run ownership

The application declares recovery settings in its Edge Toolkit scenario. Core's `et-cli` generates the supervisor,
ServiceAccount/Role/RoleBinding, projected-token kubeconfig and persistent
state. Each application supplies its data mounts, optional tracking services, images and stable workload identities.

`backends/core_recovery.py` configures the supervisor for each experiment and reads its JSON report.
The supervisor is stopped before intentional application reconfiguration, then started
after application workloads are Ready. This avoids treating deployment setup as the experiment's injected failure.

Each benchmark gets a new job identity and `/app/state/<job-id>.json`. Core preserves each job's action budget in its
state file. Reports are copied into benchmark
results before cleanup. After an ordinary benchmark, manual-start application settings and the default supervisor mode
are restored. Use only one benchmark controller per deployed application.

## Modes

- `observe-only`: Core detects and measures failures but never requests workload restarts. Kubernetes and application
  checkpoint restoration can still operate normally.
- `controller-assisted`: Core may request bounded restarts. Experiment setup selects `get,patch` permissions on exactly
  the configured Deployments; the Core supervisor does not grant itself permissions.
- `disabled`: Core is stopped for the measurement window, for example to test explicit safe failure after a checkpoint
  write error. This is not the same as observe-only.

Fault injection, deliberate fault removal, and experiment cleanup are benchmark actions in every mode. They are not
credited to Core as autonomous recovery. Explicit bridge-restart experiments remain labelled benchmark assistance.

## Application evidence and checkpoints

Applications use `edge_toolkit.recovery.Lifecycle` to send Core's direct CloudEvents through the hub. Applications also
emit structured `ET_EVENT` diagnostic logs for external observation. Both forms of instrumentation stay with the
application and operate independently of the benchmark controller.

Useful progress is application-defined and should represent accepted or committed work, not merely an attempt.
Readiness is distinct from process startup: an interrupted workload should restore usable state before announcing
readiness. Core's Python `Checkpoints` helper handles checkpoint bytes and integrity-checked pointers; the application
supplies serialization and reconstructs its state. Checkpoint granularity determines how much interrupted work may
repeat. Never deserialize untrusted executable model formats.

For federated learning, an application might checkpoint after aggregation and report progress when a round commits.
For inference, it might checkpoint a durable input offset and report completed batches. The adapter defines which
evidence proves that unfinished work resumed after recovery.

## Measurements and limitations

Core's raw report is retained under `recovery.core_report`, alongside normalized benchmark fields. It supplies MTTR,
detection time, incident status, action budgets and checkpoint timings.

Core incident timestamps are monotonic elapsed seconds within one supervisor process. The benchmark controller never
subtracts them from wall-clock fault-injection timestamps. It records a pre-injection incident boundary so earlier
incidents do not satisfy the injected fault's recovery gate. Supervisor restart/replacement invalidates the active
observer session;
the experiment fails rather than merging unrelated clocks or presenting reset history as successful recovery.
An interruption cleared within Core's detection grace can remain transient: it has no measured MTTR and does not satisfy
a required recovery/SLO gate. Such a run finishes as unmeasured failure rather than waiting for a nonexistent incident.
If training finishes without a target incident, the experiment waits up to 30 seconds for late Core evidence before
recording an unmeasured result.

A recovered incident requires Core-verified useful progress, not just a Ready Pod or a loaded checkpoint. Missing MTTR
remains null; API/observation errors are not application failures. Core durations start at its first unhealthy sample,
not at the fault injection command. Checkpoint timings are producer-measured. Sampled Pod readiness and resources
remain separate supporting indicators, not continuous availability guarantees. Resume-efficiency estimates use observed
application work-attempt logs and do not measure partial computation lost within a round.

Core action requests are attributed to `edge-toolkit-core`. Recovery without actor evidence stays `unattributed`;
it is not automatically credited to Core or Kubernetes. The overall experiment still evaluates the combined deployed
system. A single-node k3d deployment does not demonstrate physical-node failover, hub/storage loss recovery or malicious
node detection. Suspected compromise and data-noise experiments do not constitute a security architecture.

Core's initial backend requires one desired replica per supervised Deployment. Temporary scale-to-zero experiments are
therefore observed as unknown deployment configurations, not Core-detected node failures; their application outcomes and
resource samples remain useful, but Core detection/MTTR may be unmeasured. They are not physical-node failure tests.

## Release prerequisite

Recovery experiments require a compatible Core release containing the recovery SDK, scenario-generation CLI and
supervisor image. The application owns release selection, deployment setup and checkpoint/resume tests against that
release. The controller's independent tests use protocol fixtures; the minimal adapter exercises the process interface
with a synthetic workload.
