# Hub network evaluation

Test agent communication under bandwidth limits, packet loss and added latency.
Dedicated routers isolate network emulation from the hub's CPU budget.

## Run

Load the same local Core release used by Harbour-FL, then run from this directory:

```sh
mise install
mise run hub-network prepare
mise run hub-network check
mise run hub-network smoke --output results/smoke
mise run hub-network run --output results/current
```

Full runs require Linux, Docker Compose, permission to configure container networks and adequate resources.
Containers must expose cgroup v2 CPU and memory accounting; preflight checks this before starting the fleet.
Protection limits use the Docker host's Linux counters, including the VM on Docker Desktop.
`prepare` adds evaluation tools to `et-ws-server-core:local` without rebuilding the Core hub.
Resource guards stop runs that exceed configured limits. Set fleet size, conditions and durations in `settings.json`.

With `smoke` or `run`, use `--case <id>` to select a case and its baseline. Case IDs are listed in `settings.json`.
Completed cases are reused when inputs match.
Changing source, settings or images requires a new output directory.
Results include logs, measurements and cleanup status.
After interrupted teardown, use `mise run hub-network cleanup --output <results-directory>` to reverify cleanup
on the same Docker daemon before resuming.

Use `mise tasks` to list commands and `mise run test` for unit tests.
