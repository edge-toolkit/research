# Hub failover evaluation

Measure failover between two Core hubs sharing NATS and object storage. The test interrupts the primary and checks
agent reconnection, messaging and stored data through the standby. HAProxy handles hub selection.

## Prepare and run

Load the same local Core release used by Harbour-FL, then run from this directory:

```sh
mise install
mise run hub-failover prepare
ulimit -n 65536
mise run hub-failover check --scenario fleet-25000
mise run hub-failover run --scenario fleet-25000 --output results/fleet-25000
```

Requires a local Docker daemon with a Unix socket, Docker Compose and enough CPU, memory and ports.
Set defaults in `settings.json`;
`fleet-25000.json` defines the large-fleet preset. `prepare` records `et-ws-server-core:local` and pulls supporting
images; `run` includes smoke. Core is not built by this evaluation.
Completed cases are reused when inputs match. Failures stop the suite and preserve reports.
Changed inputs require a new output directory.
Avoid concurrent runs using the same ports or output directory.

Ctrl-C triggers cleanup. After a forced stop, remaining Docker resources may need manual cleanup.

## Output and other commands

The output directory contains `summary.csv`, per-attempt `report.json` files and logs.

From `experiments/reliability-benchmarks`, open the controller UI:

```sh
mise run benchmark-ui -- --catalog evaluations/hub-failover/catalog.json serve
```

Run unit tests with `mise run test`.
