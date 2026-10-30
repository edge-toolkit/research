# Benchmark results in OpenObserve

The benchmark controller exports experiment results to the OpenObserve (O2) instance deployed by Edge Toolkit Core.
The optional exporter uses the existing service and Python's standard library. O2 dashboards present recovery and
operational measurements, while MLflow records model histories and artifacts.

## Connect the existing service

Start the application's existing Core deployment. For Kubernetes, forward its O2 service in a separate terminal,
replacing the namespace below with that deployment's namespace and selecting the intended kubectl context:

```bash
kubectl --namespace application-namespace port-forward service/openobserve 5080:5080
```

Open `http://127.0.0.1:5080`. This forwards Core's existing `openobserve` service, not a second instance.
With a local or Compose deployment, use the address
published by that deployment instead. The URL must be reachable from the host running the benchmark controller,
not just from inside Kubernetes.

In the terminal that will launch the benchmark UI or CLI, set:

```bash
export BENCHMARK_O2_URL=http://127.0.0.1:5080
export BENCHMARK_O2_ORG=default
export BENCHMARK_O2_STREAM=reliability_benchmarks
export BENCHMARK_O2_USER='<your O2 ingestion user>'
```

Set `BENCHMARK_O2_PASSWORD` securely to that user's password without committing it. Prefer a dedicated identity with
ingestion access to this stream. For a local demo, the account created by the generated deployment also works; use
its actual credentials, not an assumed default. These are O2 credentials, not MLflow credentials. Launch the UI from
that same terminal with an explicitly selected application catalog. For the included interface example, run from the
repository root:

```bash
mise -C experiments/reliability-benchmarks run benchmark-ui -- \
  --catalog examples/minimal-adapter/catalog.json serve
```

Only `BENCHMARK_O2_URL` opts in. Organization and stream have the defaults above. With no URL, exports are disabled
and the benchmark behaves as before. Credentials are read on the operator side and never accepted from browser
requests or application catalogs. HTTPS is required for remote addresses by default. For an explicitly trusted
private network only, `BENCHMARK_O2_ALLOW_HTTP=1` permits remote HTTP; it does not encrypt credentials or data.
Certificate verification stays enabled. HTTP redirects are rejected rather than forwarding credentials.

## Import the dashboard

From the repository root:

```bash
mise -C experiments/reliability-benchmarks run o2-dashboard
```

In O2, choose **Dashboards -> Import** and select
`experiments/reliability-benchmarks/results/o2-dashboard.json`. For a different stream, append `--stream your_stream`
to the generation command. The generated file uses O2's native v8 format and can be regenerated without credentials.
No dashboard is created or replaced automatically on your existing O2 server.

Run a benchmark, then refresh the dashboard and select a time range covering the experiment. Export happens after
each repetition, not live during training. Summary records use the report completion time; resource samples retain
their observation times. Historical imports therefore need the historical time range. Optional panels need the
corresponding evidence: a baseline cannot supply a failure-recovery duration.

- **Runs and SLOs:** outcomes, fault targets, recovery attribution, objective bounds, SLI values and pass/fail status.
- **Recovery:** detection latency, incident duration, unresolved incidents, Core actions, checkpoints and resume.
- **Infrastructure:** CPU/memory history, Pod readiness/restarts, resource summaries, network deltas and warnings.
- **Application and evidence:** final and per-work-unit model metrics, application SLIs, checkpoint/application events.

Each record includes application, scenario, job identity, repetition, recovery mode and provider. Tables retain these
labels for comparisons. Refine SQL panels by job/application/scenario when comparing runs. Resource charts average
samples by workload across the selected range: select a single experiment's time window for an individual run.
The browser UI shows whether each repetition exported, was disabled or needs replay.
Optional fields also have a JSON-string representation, `record_data`. Dashboard queries use O2's `spath` extraction
to return null for absent fields, even when O2 has not inferred a column for them. No synthetic measurements or schema
bootstrap events are inserted to make empty panels work.

## Meaning of the results

- Recovery durations come from Core's observer-relative clock; those offsets are never presented as epoch timestamps.
  The incident table is the recovery timeline. Generic event offsets retain their own clock basis; do not subtract
  values across clock domains. Summary ingestion time is not the time at which an incident happened.
- Mean recovered-incident duration uses recovered incidents only. Unrecovered and undetected failures remain visible
  in the incident/outcome tables. Do not interpret this mean without those outcomes and the number of observations.
  Compare like-for-like applications, workloads, modes, providers and hardware; do not pool unrelated experiments.
- Missing measurements stay absent/null. Configured SLOs without evidence retain `not-measured`; they are not zero or
  successful. SLO outcomes are experimental assessments, not claims of production compliance.
- Pod sampled readiness is not end-to-end service availability. CPU millicores are consumption, not percentage
  saturation: capacity/limits are needed to make that comparison. Network summaries are observed counter increases,
  not inferred bandwidth or packet-loss measurements. Missing resource collection is not zero consumption.
- The exporter does not invent request error rates, inference latency or model metrics that an adapter has not
  measured. Per-work-unit metrics are generic: an adapter may define work units as rounds, batches or tasks.
- Raw logs, warning messages, dataset files, model binaries and arbitrary error text are not exported. Identifiers
  and numeric application metric names/values are exported; avoid putting secrets in these fields.

## Local evidence, failures and replay

The controller preserves its JSON reports regardless of O2 availability. Before sending it also writes a credential-
free outbox under the run's `o2/` directory. Requests contain up to 200 records, have a maximum three-second socket
timeout and a twelve-second delivery budget checked between batches. Export failure never changes the benchmark's
pass/fail result. Export can add a short delay after a repetition; it is outside the workload's measured run time.
Configuration failures leave the original report available for later export even if no outbox was created.

Replay a failed outbox, using the returned `o2.outbox` path, from the repository root:

```bash
mise -C experiments/reliability-benchmarks run o2-export --replay /absolute/path/to/outbox.json
```

Or export an existing full report, including reports created before the integration:

```bash
mise -C experiments/reliability-benchmarks run o2-export \
  --report /absolute/path/to/benchmark-report.json --application example-app
```

Successful batches have local receipt files and are skipped on replay to the same destination. If O2 accepts records
but its acknowledgement is lost, replay can duplicate those records. Every record has a deterministic `record_id`;
the supplied dashboard groups by this ID before calculating results. Delivery is at least once, not exactly once.
Do not count raw ingested rows as independent experimental observations. Preserve outboxes and receipts together.
Changing the destination causes the retained outbox to be sent there from the beginning. Once exported, receipts
assume O2 retains the data; remove only the relevant receipt deliberately if you need to refill an emptied stream.
Outboxes follow the same operator-managed retention as the existing benchmark reports.

Browser-driven runs export once in the parent controller, with catalog and repetition labels. Engine CLI runs export
directly. Any catalog adapter can receive the basic outcome integration; detailed panels
require the adapter to supply the generic report fields. Full-report paths are read only inside the controller-owned
run directory. The benchmark engine exports failed runs too, including available recovery evidence.

The integration uses the documented [JSON ingestion API](https://openobserve.ai/docs/reference/api/ingestion/logs/json/)
and [native dashboard import](https://openobserve.ai/docs/user-guide/analytics/dashboards/dashboards-in-openobserve/).
