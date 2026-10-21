# Reliability Benchmark Controller

The Reliability Benchmark Controller runs repeatable experiments against deployed applications. It introduces controlled
faults, collects resource and application observations, evaluates SLO targets and presents results through a browser UI,
JSON/CSV reports and optional MLflow or OpenObserve integration.

The `et-experiment-runtime` package requires Python 3.11 or newer and uses the standard library at runtime. Application
adapters connect workloads to the controller, while Edge Toolkit Core provides operational recovery supervision.

## Install and test

For a runnable UI example, follow the [workspace quick start](../README.md#quick-start).

From this directory, install the development tools and run the controller tests:

```bash
mise install
mise run test
```

To install the package into an existing application environment, activate that environment and run this command from
this directory:

```bash
python -m pip install .
```

Installation provides the `et-benchmark-ui`, `et-benchmark-o2` and `et-benchmark-o2-dashboard` commands, along with the
browser UI assets.
Each application manages its own datasets, training libraries and deployment. Kubernetes-backed experiments require
`kubectl`, access to the configured context and namespace, and a compatible Core deployment with recovery supervision
when enabled. See [recovery integration](recovery.md#release-prerequisite) for the release requirements.

## Guide

- [Structure and ownership](#structure-and-ownership)
- [Cooperative GPU failures](#cooperative-gpu-failures)
- [Client saturation profiles](#client-saturation-profiles)
- [Recovery modes and attribution](#recovery-modes-and-attribution)
- [Application adapters](#application-adapters)
- [Catalog and UI](#catalog-and-ui)
- [Recovery evidence and measurement scope](#recovery-evidence-and-measurement-scope)

## Structure and ownership

| Component                      | Responsibility                                                                     |
| ------------------------------ | ---------------------------------------------------------------------------------- |
| `benchmarking/engine.py`       | Experiment lifecycle, fault timing, recovery gates, and SLO evaluation.            |
| `benchmarking/faults.py`       | Allowlisted infrastructure faults and their teardown.                              |
| `benchmarking/sampling.py`     | Resource observations and summaries.                                               |
| `benchmarking/observations.py` | Work-unit completion and fault-target recovery gates.                              |
| `backends/`                    | Kubernetes commands and recovery observation.                                      |
| `recovery/`                    | Diagnostic parsing, recovery attribution, and measurements.                        |
| `tracking/`                    | MLflow observation/annotation and optional O2 results export/dashboard generation. |
| `web/`                         | UI, validation, supervision, repetitions, and cancellation.                        |
| Application project            | Deployment setup, model/data semantics, and application progress parsing.          |

The controller introduces and removes faults, while Core supervises operational recovery. Application evidence confirms
when useful processing resumes. Applications define their own placement experiments.

## Cooperative GPU failures

`gpu_failure` targets one allowlisted worker, with `mode` set to `runtime_error` or `out_of_memory`. It requires a
cooperating application to publish `/tmp/et-benchmark-gpu/state.json` with `schema_version: 1`, the current `job_id`,
and the selected `device` (`cuda` or `mps`). The injector refuses CPU or another job's state. It executes Python at
`/opt/cpython/bin/python3` inside the selected workload container, using the application's execution environment.

The injector atomically writes `request.json` in that directory with the schema, job, mode and a unique `request_id`.
At a training boundary the application consumes it once, publishes `receipt.json` containing the same fields plus
`device` and `status: injected`, then raises the chosen execution error. No exception should be injected unless the
application explicitly enabled this protocol. The controller requires a matching receipt within the bounded
`detection_timeout_seconds` (default 60); an unconsumed request is removed and reported `not-observed`, not success.
These simulations exercise application-level GPU error handling. The receipt confirms consumption of the injected fault;
recovery gates separately verify that useful work resumes. Physical device failures require hardware-level testing.

For an opted-in native worker, the same injection helper can run locally:

```bash
python -m et_experiments.benchmarking.gpu --job-id example-job --mode runtime_error
```

This command emits an injection receipt. Use the benchmark engine to collect recovery measurements.

## Client saturation profiles

`client_memory_saturation` and `client_network_saturation` apply bounded, increasing pressure within a selected client.
An adapter calls `faults.prepare_fault(fault, started)` before starting application workers.
Targets must be allowlisted workers. The container must provide Python at `/opt/cpython/bin/python3`.

Memory preparation may set a reference `memory_limit_mib`; cleanup restores the complete original resource settings.
The allocator executes inside the client container, reads its finite Linux cgroup v1/v2 memory limit and current usage,
and touches/revisits pages while aiming at increasing occupancy `levels`. Stages include ramp-up within `hold_seconds`;
measured samples determine whether the peak target was reached. Extra allocation is capped at
`max_extra_memory_mib` (at most 1024), while a 93% guard and `reserve_mib` headroom trigger early release. This is a
best-effort safety guard, not an OOM guarantee under concurrent allocations. The OOM benchmark remains separate.
Cgroup occupancy includes charged cache; high occupancy does not itself prove memory-bandwidth saturation or slowdown.
An unmet target or safety stop is `not-observed`, so the experiment cannot pass by merely requesting pressure.

Networking creates a private, unprivileged, time-bounded UDP echo receiver in the same namespace. The existing isolated
NET_ADMIN helper establishes a controlled egress bottleneck, then paced traffic from inside the client increases through
fractions/multiples of `rate_kbit`. This traffic competes with application traffic on the client's egress path. The
profile measures achieved offered rate, echoed payload-equivalent goodput, round-trip loss and p95 RTT. Classification
requires a working low-load reference, offered traffic above the cap, a throughput plateau and latency/loss degradation.
It is not a physical-network capacity test. Existing non-default traffic-control policies are not overwritten. The
receiver and qdisc are removed on completion or failure; the receiver also has an independent lifetime limit.
Hard termination of the controller can bypass cleanup. Verify or redeploy the test environment before reusing it if
the controller was killed, because a traffic-control policy or reference resource limit may remain in place.

JSON reports retain stage times, memory samples and network measurements. Timestamped application completions supply
per-stage throughput, duration p95 and sample counts. Their clocks must be aligned; a unit may span a phase boundary,
and small sample populations are diagnostic, not production SLO evidence. Missing timestamps remain unmeasured.
MLflow receives compact `benchmark.pressure.*` summaries; detailed samples remain in the JSON report.

Scenarios with `require_post_fault_progress` establish a fresh completed-work baseline after release and require a
subsequent increase. A workload that finishes before that baseline is captured fails this gate; increase its workload
length rather than treating prior completion or Pod readiness as recovery. `post_fault_progress_seconds` is an observed
return-to-progress interval, separate from incident MTTR. Pressure experiments need not cause a detected failure at all.

## Recovery modes and attribution

Benchmarks default to `observe-only`: the observer measures recovery but never issues operational restarts.
`controller-assisted` enables Edge Toolkit Core's bounded restart policy. It permits intervention without forcing it;
reports record whether any restart was actually requested. `disabled` stops Core supervision for expected failure
scenarios. Invalid mode/fault combinations fail before deployment changes.

Fault injection and deliberate fault removal remain benchmark actions in every mode. They are recorded separately
from recovery assistance. In particular, a requested rollout is not presented as autonomous Kubernetes recovery.
Application and deployment cleanup after measurement is not counted as operational recovery.

Reports distinguish Pod readiness, observed connectivity, useful application progress, and checkpoint restoration.
Missing actor evidence remains `unattributed`; successful progress does not automatically credit Core. The declared
scope is the combined deployed system: Edge Toolkit, Kubernetes, and the application. Actor labels describe recorded
commands and observations, not exclusive causal responsibility. Recovery of completed state remains distinct from
continuation of unfinished training.

The browser shows modes and attribution; JSON reports and MLflow tags retain the same distinctions. The CSV summary
uses work-unit columns. If a historical `summary.csv` has incompatible headers, new rows go to
`workload-summary.csv` instead. `recovery-summary.csv` retains recovery-specific columns.
Repetition summaries separate modes and omit an unqualified MTTR if results from different modes are combined.
Core and legacy observer results are not pooled into a single MTTR, and transient observations are not counted as
detected failures.
Historical reports without this metadata remain unspecified; they are not retroactively relabelled.

Adapters define how post-fault tracking ingestion is verified for their application.
Service readiness alone does not satisfy this gate. Autonomous variants leave the bridge running; explicit assisted
variants restart it and record the action. This verifies training-metric ingestion, not replay of every artifact.

## Application adapters

`contracts.ApplicationAdapter` defines the interface. Construct `BenchmarkEngine(adapter)` with an adapter instance.
Each instance has its own application configuration.

Adapters supply `WorkloadConfig(context, namespace, primary, workers, supporting, stress_image)` plus setup/cleanup,
Core recovery settings, optional tracking, and application-specific fault handling. `primary` is the progress-driving
Deployment; `workers` are additional application Deployments, such as processing workers or federated clients.

| Method                | Responsibility                                                                        |
| --------------------- | ------------------------------------------------------------------------------------- |
| `prepare_scenario()`  | Translate application options into generic fault settings, such as `after_work_unit`. |
| `recovery_config()`   | Supply Core supervisor configuration for the deployed workload.                       |
| `configure_run()`     | Prepare the deployment and fault helpers, then start the application's work.          |
| `expected_units()`    | Return the target amount of work for the run.                                         |
| `observe()`           | Reconcile logs and optional tracking into the work-unit document below.               |
| `application_fault()` | Handle application-specific faults, such as input perturbations.                      |
| `enrich_report()`     | Add application SLIs and optional `application_metrics` for MLflow annotation.        |
| `tracking_recovery()` | Define evidence of restored tracking ingestion when required by the scenario.         |
| `restore()`           | Restore application configuration after the experiment.                               |

```json
{
  "completed_units": 1,
  "expected_units": 3,
  "failed": false,
  "units": [{ "batch": 0, "metrics": { "accuracy": 0.75 }, "duration_seconds": 1.0 }],
  "unit_duration_seconds": { "mean": 1.0, "p50": 1.0, "p95": 1.0, "max": 1.0 },
  "final_metrics": { "accuracy": 0.75 }
}
```

A work unit may be an inference batch, task, or federated learning round. The adapter defines its measurement boundary.
The engine uses adapter-reported work units to measure progress. Missing measurements remain null. Application metrics
are numeric entries under `report.application_metrics`; the tracker logs them under `benchmark.<name>`.

Domain-specific reconciliation, participation rules, I/O metrics and report aliases belong in the application adapter.
Application runtimes can use the Core SDK directly, while benchmark adapters connect them to this package. The runnable
[minimal adapter](../examples/minimal-adapter/adapter.py) demonstrates the process interface with a synthetic workload.

## Catalog and UI

A trusted project catalog supplies an adapter command, bounded parameters, and one source of scenario definitions:

```json
{
  "schema_version": 1,
  "adapter": { "command": [".venv/bin/python", "benchmarking/adapter.py"] },
  "scenarios_file": "benchmarking/scenarios.json",
  "parameters": {
    "rounds": { "type": "integer", "default": 3, "minimum": 1, "maximum": 20 }
  }
}
```

Paths in the catalog are relative to its directory, which serves as the application project root. The adapter command
runs there by default; an optional `adapter.cwd` selects a subdirectory within that root.

The scenario file defines descriptions, fault settings, SLOs, and defaults. A scenario specifying five rounds overrides
the catalog's shared default of three. Explicit operator choices override scenario defaults. Each experiment saves
its effective configuration. Inline `scenarios` remain supported but cannot be combined with `scenarios_file`.

Run `et-benchmark-ui --catalog path/to/catalog.json --results-dir path/to/outputs serve` from an environment with
this package installed. Omitting `--results-dir` retains the catalog's project-local output location. The UI
listens on `127.0.0.1:8765`. Browser input cannot supply commands, Kubernetes objects, filesystem paths, or environment
variables. Only one experiment runs at a time per controller; use one controller per application deployment.

Adapters receive `BENCHMARK_REQUEST_FILE` and `BENCHMARK_RESULT_FILE`, honor termination, restore injected changes, and
write JSON results. `BENCHMARK_OUTPUT_DIR` is the controller-owned run directory for detailed reports and artifacts;
adapters should write their outputs there. Repetitions are sequential. Reports retain missing measurements and
unrecovered incidents. The controller saves completed evidence before attempting optional MLflow annotation. Core's
generic MLflow bridge records application events; the controller adds experiment-level annotations.

## Recovery evidence and measurement scope

For operational results in Core's existing O2 service, see [OpenObserve setup and dashboards](openobserve.md).
The optional exporter retains local evidence and replayable batches. Benchmark outcomes are independent of O2 delivery.

See [recovery integration](recovery.md) for metric definitions, the event contract, and checkpoint behaviour.
Recovery tests wait independently of training completion. Applications can require post-fault checkpoint restore/resume
events.
Ready pods and cached predecessor progress do not establish recovery. Restoring and republishing completed state is
different from continuing unfinished training after a failure.

Thresholds are experimental targets, not service guarantees. Missing measurements are null; configured objectives
without evidence do not pass. Sampled readiness is not continuous uptime or request-success availability. Request
error rates and inference latency require application observations. Physical-node recovery and non-Kubernetes backends
require further implementation or validation.
