# Reliability benchmarks

The Reliability Benchmark Controller and its browser UI evaluate failure handling, recovery and degraded-input behaviour
in applications built on Edge Toolkit. Application catalogs connect the controller to independently deployed workloads
through trusted adapter commands. Edge Toolkit provides recovery supervision, while applications restore their own state
and processing.

## Quick start

Install [mise](https://mise.jdx.dev/getting-started.html), then run these commands from the repository root:

```bash
mise install
mise -C experiments/reliability-benchmarks install
mise -C experiments/reliability-benchmarks run benchmark-ui -- \
  --catalog examples/minimal-adapter/catalog.json serve
```

Open `http://127.0.0.1:8765`. The included example uses synthetic results to demonstrate scenario selection, experiment
execution and reporting. Replace `serve` with `list` to inspect its scenarios from the terminal.

To benchmark an application, prepare its deployment and dependencies, then supply its catalog:

```bash
mise -C experiments/reliability-benchmarks run benchmark-ui -- \
  --catalog /absolute/path/to/application/catalog.json serve
```

Catalogs are trusted operator configuration. Review their adapter commands before running them. Use one controller per
application deployment; each controller runs one experiment at a time.

## Workspace layout

- [controllers](controllers/README.md): the independently installable package for benchmarking, Core recovery evidence,
  metrics, tracking, and the browser UI. Application code and deployments remain separate.
- [minimal adapter](examples/minimal-adapter/adapter.py): a runnable process-interface example using synthetic results.
- `results/`: ignored controller-owned requests, detailed reports, artifacts and repetition summaries.

Application-specific model behaviour and data perturbations are implemented in the application adapter or module.

For results in Core's existing OpenObserve service, see [O2 setup and dashboard import](controllers/openobserve.md).
For reference conditions, repeated runs and matched baselines, see [experiment design](controllers/experiments.md).

## Run the tests

From the repository root, install the tools and run the controllers' independent tests:

```bash
mise -C experiments/reliability-benchmarks/controllers install
mise run test-experiments
```

## Environments and results

The UI runs in its own Python environment. Adapter commands run in the selected application's directory and can use that
application's separate environment. The workspace's `benchmark-ui` task stores experiment outputs under `results/`,
including reports, artifacts and repetition summaries. Direct application CLI runs retain their own output defaults.

## Connect another module

1. Keep its module, deployment and dependencies in a separate project directory.
2. Supply a catalog with an allowlisted adapter command, supported scenarios and bounded parameters.
3. Have the adapter read `BENCHMARK_REQUEST_FILE`, print progress, and write JSON to `BENCHMARK_RESULT_FILE`.
   Detailed reports and artifacts belong under `BENCHMARK_OUTPUT_DIR`. Honor cancellation and restore injected faults.
4. Launch the UI with that catalog.

The process interface supports non-Python adapters. Python adapters can additionally implement `ApplicationAdapter`
from `controllers/src/et_experiments/contracts.py` and use `BenchmarkEngine(adapter)` for reusable Kubernetes faults,
Core recovery-report collection and SLO evaluation. Adapters define supported measurements and fault capabilities,
including application-specific model semantics, checkpoint reconstruction and input corruption.

Datasets, local configuration, generated deployments and application environments stay with their application.
Applications can use Edge Toolkit's generic MLflow bridge to record training metrics and artifacts.

See the [controller documentation](controllers/README.md#application-adapters) for the Python adapter lifecycle and
[catalog format](controllers/README.md#catalog-and-ui).
