# Harbour-FL participant scaling

Measure round latency, model-data throughput, CPU and memory across client counts.
The runner uses the Harbour-FL adapter and a deployment's images and prepared features to create an isolated fleet.

## Run

Prepare and deploy Harbour-FL, then run `mise run setup:benchmark` in its application directory.
Copy `settings-per-client.json` to `settings.local.json`. Set `application` to that directory and configure
`context`, `source_namespace` and optional `kubeconfig`.

From this directory:

```sh
mise install
mise run scaling plan --settings settings.local.json
mise run scaling check --settings settings.local.json
mise run scaling smoke --settings settings.local.json
mise run scaling run --settings settings.local.json
```

`check` verifies compatibility, data, permissions and metrics. `smoke` runs one client for one round and checks cleanup.
Use the same settings and output directory for smoke and full runs. Completed cases are reused when inputs match;
changed inputs require a new output directory. `--case 6` runs the six-client case.

`settings-per-client.json` holds work per client constant; `settings.json` holds total data constant.
Train, validation and test data stay separate, and results record any sample reuse.
Workloads use private, hash-verified feature snapshots so later data preparation cannot change their training inputs.
CPU statistics use untrimmed Metrics API windows, which can include startup and idle time.
Result archives omit input and pod specifications that can contain credentials.
Use `mise run scaling cleanup --settings settings.local.json` to reverify cleanup on the recorded cluster and namespace.

Use `mise tasks` to list commands and `mise run test` for unit tests.
